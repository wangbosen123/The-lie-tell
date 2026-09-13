"""Main script for trajectory optimization."""

import os
import random
import pickle

import torch
import torch.optim as optim
import torch.nn as nn
import torch.distributed as dist
from torch_geometric.data import Batch
from matplotlib import pyplot as plt
import numpy as np
import tap
from typing import Tuple, Optional
from pathlib import Path
import plotly.graph_objects as go

from lda.data.calvin import CalvinDataset
from lda.trainer import TrainTester as BaseTrainTester
# from lda.trainer import traj_collate_fn, fig_to_numpy, Arguments
from lda.utils.common import (
    load_instructions, get_gripper_loc_bounds
)

# from graph_beta_diffusion.graph_diffusion_general import GraphBetaDiffusion

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation


class TrainTester(BaseTrainTester):
    """Train/test a trajectory optimization algorithm."""

    def __init__(self, args):
        """Initialize."""
        super().__init__(args)

    def get_datasets(self):
        '''
        1. 載入 instruction embeddings
        2. 建 training CalvinDataset
        3. 建 validation CalvinDataset
        4. 回傳 train_dataset, test_dataset
        '''
        """Initialize datasets."""
        # Load instruction, based on which we load tasks/variations
        # Initialize datasets with arguments
        taskvar = [
            ("A", 0), ("B", 0), ("C", 0), ("D", 0),
            # ("D", 0),
        ]

        train_instruction = load_instructions(
            self.args.instructions, 'training'
        )   
        train_dataset = CalvinDataset(
            root=self.args.dataset,
            instructions=train_instruction,
            taskvar=taskvar,
            max_episode_length=self.args.max_episode_length,
            cache_size=self.args.cache_size,
            max_episodes_per_task=self.args.max_episodes_per_task,
            num_iters=self.args.train_iters,
            cameras=self.args.cameras,
            training=True,
            image_rescale=tuple(
                float(x) for x in self.args.image_rescale.split(",")
            ),
            return_low_lvl_trajectory=True,
            dense_interpolation=bool(self.args.dense_interpolation),
            interpolation_length=self.args.interpolation_length,
            relative_action=bool(self.args.relative_action),
            training_split=self.args.training_split,
        )

        test_instruction = load_instructions(
            self.args.instructions, 'validation'
        )
        test_dataset = CalvinDataset(
            root=self.args.valset,
            instructions=test_instruction,
            taskvar=taskvar,
            max_episode_length=self.args.max_episode_length,
            cache_size=self.args.cache_size_val,
            max_episodes_per_task=self.args.max_episodes_per_task,
            cameras=self.args.cameras,
            training=False,
            image_rescale=tuple(
                float(x) for x in self.args.image_rescale.split(",")
            ),
            return_low_lvl_trajectory=True,
            dense_interpolation=bool(self.args.dense_interpolation),
            interpolation_length=self.args.interpolation_length,
            relative_action=bool(self.args.relative_action),
            training_split=self.args.training_split,
        )
        return train_dataset, test_dataset

    def save_checkpoint(self, model, optimizer, step_id, new_loss, best_loss):
        """Save checkpoint if requested."""
        if new_loss is None or best_loss is None or new_loss <= best_loss:
            best_loss = new_loss
            torch.save({
                "weight": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "iter": step_id + 1,
                "best_loss": best_loss
            }, self.args.log_dir / "best.pth")
        torch.save({
            "weight": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "iter": step_id + 1,
            "best_loss": best_loss
        }, self.args.log_dir / '{:07d}.pth'.format(step_id))
        torch.save({
            "weight": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "iter": step_id + 1,
            "best_loss": best_loss
        }, self.args.log_dir / "last.pth")
        return best_loss

    def get_optimizer(self, model):
        """Initialize optimizer."""
        optimizer_grouped_parameters = [
            {"params": [], "weight_decay": 0.0, "lr": self.args.lr},
            {"params": [], "weight_decay": self.args.wd, "lr": self.args.lr}
        ]
        no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]
        for name, param in model.named_parameters():
            if any(nd in name for nd in no_decay):
                optimizer_grouped_parameters[0]["params"].append(param)
            else:
                optimizer_grouped_parameters[1]["params"].append(param)
        optimizer = optim.AdamW(optimizer_grouped_parameters)
        return optimizer
    
    def process_batch(self, batch):
        """Process a batch of data."""
        # Convert to device
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(self.device)
            elif isinstance(batch[key], list):
                batch[key] = [item.to(self.device) for item in batch[key]]
        return batch
    

class Arguments(tap.Tap):
    # CALVIN uses the static "front" cam + the wrist cam (matches what
    # ``scripts/package_calvin.py`` writes into camera_dicts). The
    # RLBench triple-camera default would fail the camera-list assert in
    # CalvinDataset.__getitem__ on the first batch.
    cameras: Tuple[str, ...] = ("front", "wrist")
    image_size: str = "256,256"
    max_episodes_per_task: int = 100
    instructions: Optional[Path] = "instructions.pkl"
    seed: int = 0
    tasks: Tuple[str, ...]
    variations: Tuple[int, ...] = (0,)
    checkpoint: Optional[Path] = None
    accumulate_grad_batches: int = 1
    val_freq: int = 500
    gripper_loc_bounds: Optional[str] = None
    gripper_loc_bounds_buffer: float = 0.04
    eval_only: int = 0

    # Training and validation datasets
    dataset: Path
    valset: Path
    dense_interpolation: int = 0
    interpolation_length: int = 100

    # Logging to base_log_dir/exp_log_dir/run_log_dir
    base_log_dir: Path = Path(__file__).parent / "train_logs"
    exp_log_dir: str = "exp"
    run_log_dir: str = "run"

    # Main training parameters
    num_workers: int = 1
    batch_size: int = 16
    batch_size_val: int = 4
    cache_size: int = 100
    cache_size_val: int = 100
    lr: float = 1e-4
    end_lr: float = 1e-5
    lr_decay: float = 0.95
    wd: float = 5e-3  # used only for CALVIN
    weight_decay: int = 1e-12
    train_iters: int = 200_000
    val_iters: int = -1  # -1 means heuristically-defined
    max_episode_length: int = 5  # -1 for no limit

    # power noise
    noise_start: float = 1e-8
    noise_end: float = 1.0
    diffusion_timesteps: int = 100
    noise_power: float = 3.0

    # Data augmentations
    image_rescale: str = "0.75,1.25"  # (min, max), "1.0,1.0" for no rescaling

    # Model
    backbone: str = "clip"  # one of "resnet", "clip"
    embedding_dim: int = 120
    num_vis_ins_attn_layers: int = 2
    use_instruction: int = 0
    rotation_parametrization: str = 'quat'
    quaternion_format: str = 'wxyz'
    keypose_only: int = 0
    num_history: int = 0
    num_joints: int = 7
    relative_action: int = 0
    lang_enhanced: int = 0
    fps_subsampling_factor: int = 5

    # Ablation flags (express paper Table 1 rows as configs)
    use_gat: int = 1
    """1 = use GAT encoder (paper main); 0 = ablate (w/o GAT row)."""
    diffusion_space: str = "lie"
    """lie = score matching on SE(3) tangent (paper main); euclidean = DDPM (w/o Lie row)."""
    loss_formulation: str = "new"
    """new = paper's corrected loss; old = earlier formulation (supplementary ckpt only)."""
    training_split: str = "ABC"
    """ABC = train on tasks A/B/C; ABCD = train on merged ABC+D set."""


def traj_collate_fn(batch):
    keys = [
        "trajectory", "trajectory_mask",
        "rgbs", "pcds",
        "curr_gripper", "curr_gripper_history", "action", "instr", 
        "joints_coords",
    ]
    ret_dict = {
        key: torch.cat([
            item[key].float() if key != 'trajectory_mask' else item[key]
            for item in batch
        ]) for key in keys
    }

    ret_dict["task"] = []
    for item in batch:
        ret_dict["task"] += item['task']
    return ret_dict

def load_instructions(instructions, split):
    instructions = pickle.load(
        open(f"{instructions}/{split}.pkl", "rb")
    )['embeddings']
    return instructions

def animate_arm_with_pcd(joints_coords, pcds, interval=200, cam_idx=0, point_subsample=2000):
    """
    Animate the arm motion and point cloud over time.
    joints_coords: shape (traj_len, num_joints, 7)
    pcds: shape (traj_len, n_cam, 3, H, W)
    cam_idx: which camera to visualize (default 0)
    point_subsample: number of points to randomly sample from the point cloud for display
    """
    traj_len, num_joints, _ = joints_coords.shape
    xyz = joints_coords[:, :, :3]  # (traj_len, num_joints, 3)

    # Prepare point cloud for each frame
    pcd_points = []
    for t in range(traj_len):
        # pts = pcds[t, cam_idx].reshape(3, -1).T  # (H*W, 3)
        pts = pcds[t]
        # if pts.shape[0] > point_subsample:
        #     idx = np.random.choice(pts.shape[0], point_subsample, replace=False)
        #     pts = pts[idx]
        pcd_points.append(pts)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    colors = plt.cm.jet(np.linspace(0, 1, num_joints))

    def update(frame):
        ax.cla()
        # Plot point cloud
        pts = pcd_points[frame]
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c='gray', s=0.5, alpha=0.2, label='pcd')
        # Plot joints/arm
        for j in range(num_joints):
            ax.plot(xyz[:frame+1, j, 0], xyz[:frame+1, j, 1], xyz[:frame+1, j, 2], color=colors[j])
            ax.scatter(xyz[frame, j, 0], xyz[frame, j, 1], xyz[frame, j, 2], color=colors[j], marker='o')
        ax.plot(xyz[frame, :, 0], xyz[frame, :, 1], xyz[frame, :, 2], c='k', alpha=0.5)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f"Frame {frame+1}/{traj_len}")
        ax.set_xlim(np.min(xyz[...,0]), np.max(xyz[...,0]))
        ax.set_ylim(np.min(xyz[...,1]), np.max(xyz[...,1]))
        ax.set_zlim(np.min(xyz[...,2]), np.max(xyz[...,2]))

    ani = animation.FuncAnimation(fig, update, frames=traj_len, interval=interval, repeat=False)
    plt.show()

def node_type_to_color(node_type):
    color_map = {
        0: 'red',     # arm
        1: 'blue',    # context
        2: 'green'    # (optional: other type)
    }
    return [color_map.get(int(t), 'gray') for t in node_type]

def plot_3d_graph_with_types(data):
    pos = data.pos.cpu().numpy()                  # (N, 3)
    edge_index = data.edge_index.cpu().numpy()    # (2, E)
    node_type = data.node_type.cpu().numpy()      # (N,)

    # Color by node type
    colors = node_type_to_color(node_type)

    # Plot edges as lines
    edge_trace = []
    for i in range(edge_index.shape[1]):
        src, tgt = edge_index[:, i]
        edge_trace.append(go.Scatter3d(
            x=[pos[src, 0], pos[tgt, 0], None],
            y=[pos[src, 1], pos[tgt, 1], None],
            z=[pos[src, 2], pos[tgt, 2], None],
            mode='lines',
            line=dict(width=2, color='gray'),
            showlegend=False
        ))

    # Plot nodes
    node_trace = go.Scatter3d(
        x=pos[:, 0], y=pos[:, 1], z=pos[:, 2],
        mode='markers',
        marker=dict(size=6, color=colors),
        text=[f"Node {i}, type: {t}" for i, t in enumerate(node_type)],
        hoverinfo='text',
        name='Nodes'
    )

    fig = go.Figure(data=[*edge_trace, node_trace])
    fig.update_layout(
        title='3D Graph Visualization by Node Type',
        scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'),
        margin=dict(l=0, r=0, b=0, t=40),
        showlegend=False
    )
    print(f"showing figure")
    fig.write_html("graph.html")
    fig.show()


if __name__ == '__main__':
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # Arguments
    args = Arguments().parse_args()
    print("Arguments:")
    print(args)
    print("-" * 100)
    if args.gripper_loc_bounds is None:
        args.gripper_loc_bounds = np.array([[-2, -2, -2], [2, 2, 2]]) * 1.0
    else:
        args.gripper_loc_bounds = get_gripper_loc_bounds(
            args.gripper_loc_bounds,
            task=args.tasks[0] if len(args.tasks) == 1 else None,
            buffer=args.gripper_loc_bounds_buffer,
        )
    log_dir = args.base_log_dir / args.exp_log_dir / args.run_log_dir
    args.log_dir = log_dir
    log_dir.mkdir(exist_ok=True, parents=True)
    print("Logging:", log_dir)
    print(
        "Available devices (CUDA_VISIBLE_DEVICES):",
        os.environ.get("CUDA_VISIBLE_DEVICES")
    )
    print("Device count", torch.cuda.device_count())
    args.local_rank = int(os.environ["LOCAL_RANK"])

    # Seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # DDP initialization
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    train_tester = TrainTester(args)
    train_tester.main(collate_fn=traj_collate_fn)