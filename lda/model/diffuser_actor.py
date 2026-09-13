import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
import math
from theseus.geometry import SE3
import numpy as np

from lda.encoder.layers import (
    FFWRelativeSelfAttentionModule,
    FFWRelativeCrossAttentionModule,
    FFWRelativeSelfCrossAttentionModule
)
from lda.encoder.encoder import Encoder
from lda.encoder.layers import ParallelAttention
from lda.encoder.position_encodings import (
    RotaryPositionEncoding3D,
    SinusoidalPosEmb
)
from lda.encoder.utils import (
    compute_rotation_matrix_from_ortho6d,
    get_ortho6d_from_rotation_matrix,
    normalise_quat,
    matrix_to_quaternion,
    quaternion_to_matrix
)
from lda.diffusion.lie.noise import PowerNoiseSchedule
from lda.diffusion.lie.metrics import se3 as lie_metrics
from lda.diffusion.lie.dist.se3 import NormalSE3
from lda.diffusion.lie.utils import ops

from lda.gat.graph_planner import GraphPlanner
from lda.gat.models.gat_encoder import GATEncoder
from torch_geometric.data import Batch


class DiffuserActor(nn.Module):

    def __init__(self,
                 backbone="clip",
                 image_size=(256, 256),
                 embedding_dim=60,
                 num_vis_ins_attn_layers=2,
                 use_instruction=False,
                 fps_subsampling_factor=5,
                 gripper_loc_bounds=None,
                 rotation_parametrization='6D',
                 quaternion_format='xyzw',
                 diffusion_timesteps=100,
                 nhist=3,
                 relative=False,
                 lang_enhanced=False,
                 args=None):
        super().__init__()
        self._rotation_parametrization = rotation_parametrization
        self._quaternion_format = quaternion_format
        self._relative = relative
        self.use_instruction = use_instruction
        # Ablation flag: when False, drop the GAT planner + cross-attention
        # condition path entirely. Source branch: nogat-score-2enc.
        self.use_gat = bool(getattr(args, "use_gat", 1)) if args is not None else True
        # Ablation flag: "lie" runs the SE(3)-tangent score-matching path
        # (default, source branch new_loss); "euclidean" runs vanilla DDPM
        # over (3D position, 6D rotation) (source branch gat-ddpm-2enc).
        self.diffusion_space = getattr(args, "diffusion_space", "lie") if args is not None else "lie"
        # Ablation flag: "new" uses the weighted-MSE loss on the first
        # prediction layer (source branch new_loss); "old" uses per-axis
        # MSE x [20, 10] summed over every prediction layer + per-layer
        # BCE openness (source branch gat-score-2enc, the parent of
        # new_loss). Only affects the lie training path.
        self.loss_formulation = getattr(args, "loss_formulation", "new") if args is not None else "new"
        self.encoder = Encoder(
            backbone=backbone,
            image_size=image_size,
            embedding_dim=embedding_dim,
            num_sampling_level=1,
            nhist=nhist,
            num_vis_ins_attn_layers=num_vis_ins_attn_layers,
            fps_subsampling_factor=fps_subsampling_factor
        )
        self.prediction_head = DiffusionHead(
            embedding_dim=embedding_dim,
            use_instruction=use_instruction,
            rotation_parametrization=rotation_parametrization,
            nhist=nhist,
            lang_enhanced=lang_enhanced,
            use_gat=self.use_gat,
            diffusion_space=self.diffusion_space,
        )
        if self.diffusion_space == "euclidean":
            self.position_noise_scheduler = DDPMScheduler(
                num_train_timesteps=diffusion_timesteps,
                beta_schedule="scaled_linear",
                prediction_type="epsilon"
            )
            self.rotation_noise_scheduler = DDPMScheduler(
                num_train_timesteps=diffusion_timesteps,
                beta_schedule="squaredcos_cap_v2",
                prediction_type="epsilon"
            )
        else:
            self.noise_scheduler = PowerNoiseSchedule(
                alpha_start=args.noise_start,
                alpha_end=args.noise_end,
                timesteps=args.diffusion_timesteps,
                power=args.noise_power
            )
            self.repr_type = 'tan'

        self.n_steps = diffusion_timesteps
        self.gripper_loc_bounds = torch.tensor(gripper_loc_bounds)

        if self.use_gat:
            self.graph_planner = GraphPlanner(
                backbone=args.backbone,
                image_size=tuple(int(x) for x in args.image_size.split(",")),
                embedding_dim=args.embedding_dim,
                njoints=args.num_joints,
                num_vis_ins_attn_layers=args.num_vis_ins_attn_layers,
                fps_subsampling_factor=args.fps_subsampling_factor,
                use_instruction=bool(args.use_instruction),
            )

            self.gat_encoder = GATEncoder(
                in_channels=197,
                out_channels=512,
                edge_dim=60,
            )

        # For Calvin
        self.r0_min = [-1.0520354509353638, -1.1197506189346313, -1.1390202045440674, -3.1401865482330322, -2.7450056076049805, -3.139648199081421]
        self.r0_max = [0.8016011714935303, 0.9660374522209167, 1.2049458026885986, 3.1414663791656494, 3.1119823455810547, 3.1353979110717773]
    
    #  (呼叫不同的encoder，class 寫在一起而已)
    def encode_inputs(self, visible_rgb, visible_pcd, instruction,
                      curr_gripper):
        # Compute visual features/positional embeddings at different scales
        rgb_feats_pyramid, pcd_pyramid = self.encoder.encode_images(
            visible_rgb, visible_pcd
        )
        # Keep only low-res scale
        context_feats = einops.rearrange(
            rgb_feats_pyramid[0],
            "b ncam c h w -> b (ncam h w) c"
        )
        context = pcd_pyramid[0]

        # Encode instruction (B, 53, F)
        instr_feats = None
        if self.use_instruction:
            instr_feats, _ = self.encoder.encode_instruction(instruction)

        # Cross-attention vision to language
        if self.use_instruction:
            # Attention from vision to language
            context_feats = self.encoder.vision_language_attention(
                context_feats, instr_feats
            )

        # Encode gripper history (B, nhist, F)
        adaln_gripper_feats, _ = self.encoder.encode_curr_gripper(
            curr_gripper, context_feats, context
        )

        # FPS on visual features (N, B, F) and (B, N, F, 2)
        fps_feats, fps_pos = self.encoder.run_fps(
            context_feats.transpose(0, 1),
            self.encoder.relative_pe_layer(context)
        )
        return (
            context_feats, context,  # contextualized visual features
            instr_feats,  # language features
            adaln_gripper_feats,  # gripper history features
            fps_feats, fps_pos  # sampled visual features
        )

    # policy forward inputs=> trajectory = 目前帶噪聲的候選動作軌跡 timestep   = diffusion step
    # fixed_inputs = encode_inputs() 得到的 observation condition, condition = GAT 額外 condition
    def policy_forward_pass(self, trajectory, timestep, fixed_inputs, condition):
        # Parse inputs
        (
            context_feats,
            context,
            instr_feats,
            adaln_gripper_feats,
            fps_feats,
            fps_pos
        ) = fixed_inputs

        return self.prediction_head(
            trajectory,
            timestep,
            context_feats=context_feats,
            context=context,
            instr_feats=instr_feats,
            adaln_gripper_feats=adaln_gripper_feats,
            fps_feats=fps_feats,
            fps_pos=fps_pos,
            condition=condition,
        )


    def conditional_sample(self, condition_data, condition_mask, fixed_inputs, GATcondition=None):
        if self.diffusion_space == "euclidean":
            return self._conditional_sample_euclidean(
                condition_data, condition_mask, fixed_inputs, GATcondition
            )
        return self._conditional_sample_lie(
            condition_data, condition_mask, fixed_inputs, GATcondition
        )

    def _conditional_sample_lie(self, condition_data, condition_mask, fixed_inputs, GATcondition=None):
        '''
            1. 先從 SE(3) 上 sample 隨機 pose
            2. 每個 denoising step：
            - 把 pose 轉成 tangent representation
            - 讓模型預測 mu，也就是修正方向
            - 用 exponential map 把 tangent-space 的小位移套回 SE(3)
            3. 最後轉回 quaternion pose
            4. 接上 gripper openness
        '''
        device = condition_data.device
        num_steps = self.noise_scheduler.timesteps

        batch, seq_len = condition_data.shape[:2]
        steps = 100

        time_arr = np.linspace(num_steps - 1, 0, int(steps))
        # TODO: check the batch * seq_len sample then reshape
        poses = lie_metrics.as_mat(NormalSE3._sample_unit(n=(batch * seq_len,)).to(device))

        for t in time_arr:
            tt = torch.tensor(np.full([batch, 1], t, dtype=np.int32)).to(device)
            poses = lie_metrics.as_repr(poses, "tan")
            poses = poses.view(batch, seq_len, -1)


            mu = self.policy_forward_pass(
                poses,
                tt.squeeze(-1),
                fixed_inputs,
                condition=GATcondition,
            )[-1]

            openness = mu[..., 6:]
            mu = mu[..., :6]

            t = np.full([batch * seq_len], t, dtype=np.int32)

            sigma_t = self.noise_scheduler.sqrt_alphas[t]
            sigma_L = np.full([batch * seq_len], (self.noise_scheduler.alpha_start) ** 0.5, dtype=np.float32)
            sigma_t = torch.tensor(sigma_t).unsqueeze(dim=1)
            sigma_L = torch.tensor(sigma_L).unsqueeze(dim=1)

            epsilon = 2e-8
            step_size = (epsilon * 0.5 * (sigma_t ** 2) / (sigma_L ** 2)).to(device)
            noise = (NormalSE3._sample_unit(n=(batch * seq_len,))).to(device)
            poses = poses.view(batch * seq_len, -1)
            poses = lie_metrics.as_mat(poses)
            mu = mu.view(batch * seq_len, -1)
            poses = torch.bmm(poses, lie_metrics.as_mat(step_size * mu / sigma_t.to(device) + 0.01 * torch.sqrt(2 * step_size) * noise))

        # Convert to tangent space for unnormalization
        poses = lie_metrics.as_lie(poses)
        poses = lie_metrics.as_repr(poses, "tan")
        # Unnormalize in tangent space
        poses = self.unnormalize_r0(poses)
        # Back to matrix
        poses = lie_metrics.as_mat(poses)

        poses = lie_metrics.as_lie(poses)
        poses = lie_metrics.as_quat(poses)
        poses = poses.view(batch, seq_len, -1)
        opensess = openness.view(batch, seq_len, -1)
        traj = torch.cat((poses, opensess), -1)
        return traj

    def _conditional_sample_euclidean(self, condition_data, condition_mask, fixed_inputs, GATcondition=None):
        # Vanilla DDPM: position and rotation are denoised on independent
        # schedulers; the gripper-openness logit comes from the model output's
        # tail and is appended after the loop. Source branch: gat-ddpm-2enc.
        self.position_noise_scheduler.set_timesteps(self.n_steps)
        self.rotation_noise_scheduler.set_timesteps(self.n_steps)

        # Random trajectory, conditioned on start-end
        noise = torch.randn(
            size=condition_data.shape,
            dtype=condition_data.dtype,
            device=condition_data.device
        )
        # Noisy condition data
        noise_t = torch.ones(
            (len(condition_data),), device=condition_data.device
        ).long().mul(self.position_noise_scheduler.timesteps[0])
        noise_pos = self.position_noise_scheduler.add_noise(
            condition_data[..., :3], noise[..., :3], noise_t
        )
        noise_rot = self.rotation_noise_scheduler.add_noise(
            condition_data[..., 3:9], noise[..., 3:9], noise_t
        )
        noisy_condition_data = torch.cat((noise_pos, noise_rot), -1)
        trajectory = torch.where(
            condition_mask, noisy_condition_data, noise
        )

        timesteps = self.position_noise_scheduler.timesteps
        for t in timesteps:
            out = self.policy_forward_pass(
                trajectory,
                t * torch.ones(len(trajectory)).to(trajectory.device).long(),
                fixed_inputs,
                condition=GATcondition,
            )
            out = out[-1]  # keep only last layer's output
            pos = self.position_noise_scheduler.step(
                out[..., :3], t, trajectory[..., :3]
            ).prev_sample
            rot = self.rotation_noise_scheduler.step(
                out[..., 3:9], t, trajectory[..., 3:9]
            ).prev_sample
            trajectory = torch.cat((pos, rot), -1)

        trajectory = torch.cat((trajectory, out[..., 9:]), -1)
        return trajectory

    # 作用: diffusion 產生 trajectory	物理意義: 從 noise denoise 成 end-effector trajectory
    def compute_trajectory(
        self,
        trajectory_mask,
        rgb_obs,
        pcd_obs,
        instruction,
        curr_gripper,
        condition=None
    ):
        # Normalize all pos
        pcd_obs = pcd_obs.clone()
        curr_gripper = curr_gripper.clone()
        pcd_obs = torch.permute(self.normalize_pos(
            torch.permute(pcd_obs, [0, 1, 3, 4, 2])
        ), [0, 1, 4, 2, 3])
        curr_gripper[..., :3] = self.normalize_pos(curr_gripper[..., :3])
        if self.diffusion_space == "euclidean":
            curr_gripper = self.convert_rot(curr_gripper)

        # Prepare inputs
        fixed_inputs = self.encode_inputs(
            rgb_obs, pcd_obs, instruction, curr_gripper
        )

        # Condition on start-end pose
        B, nhist, D = curr_gripper.shape
        cond_data = torch.zeros(
            (B, trajectory_mask.size(1), D),
            device=rgb_obs.device
        )
        cond_mask = torch.zeros_like(cond_data)
        cond_mask = cond_mask.bool()

        # Sample
        trajectory = self.conditional_sample(
            cond_data,
            cond_mask,
            fixed_inputs,
            GATcondition=condition,
        )


        # Normalize quaternion
        if self._rotation_parametrization != '6D':
            trajectory[:, :, 3:7] = normalise_quat(trajectory[:, :, 3:7])
        # Back to quaternion
        if self.diffusion_space == "euclidean":
            trajectory = self.unconvert_rot(trajectory)
        # unnormalize position
        trajectory[:, :, :3] = self.unnormalize_pos(trajectory[:, :, :3])
        # Convert gripper status to probaility
        if trajectory.shape[-1] > 7:
            trajectory[..., 7] = trajectory[..., 7].sigmoid()

        return trajectory

    def normalize_pos(self, pos):
        pos_min = self.gripper_loc_bounds[0].float().to(pos.device)
        pos_max = self.gripper_loc_bounds[1].float().to(pos.device)
        return (pos - pos_min) / (pos_max - pos_min) * 2.0 - 1.0

    def unnormalize_pos(self, pos):
        pos_min = self.gripper_loc_bounds[0].float().to(pos.device)
        pos_max = self.gripper_loc_bounds[1].float().to(pos.device)
        return (pos + 1.0) / 2.0 * (pos_max - pos_min) + pos_min
    
    def normalize_r0(self, r0):
        r0_min = torch.tensor(self.r0_min).float().to(r0.device)
        r0_max = torch.tensor(self.r0_max).float().to(r0.device)
        return (r0 - r0_min) / (r0_max - r0_min) * 2.0 - 1.0

    def unnormalize_r0(self, r0):
        r0_min = torch.tensor(self.r0_min).float().to(r0.device)
        r0_max = torch.tensor(self.r0_max).float().to(r0.device)
        return (r0 + 1.0) / 2.0 * (r0_max - r0_min) + r0_min

    def convert_rot(self, signal):
        signal[..., 3:7] = normalise_quat(signal[..., 3:7])
        if self._rotation_parametrization == '6D':
            # The following code expects wxyz quaternion format!
            if self._quaternion_format == 'xyzw':
                signal[..., 3:7] = signal[..., (6, 3, 4, 5)]
            rot = quaternion_to_matrix(signal[..., 3:7])
            res = signal[..., 7:] if signal.size(-1) > 7 else None
            if len(rot.shape) == 4:
                B, L, D1, D2 = rot.shape
                rot = rot.reshape(B * L, D1, D2)
                rot_6d = get_ortho6d_from_rotation_matrix(rot)
                rot_6d = rot_6d.reshape(B, L, 6)
            else:
                rot_6d = get_ortho6d_from_rotation_matrix(rot)
            signal = torch.cat([signal[..., :3], rot_6d], dim=-1)
            if res is not None:
                signal = torch.cat((signal, res), -1)
        return signal

    def unconvert_rot(self, signal):
        if self._rotation_parametrization == '6D':
            res = signal[..., 9:] if signal.size(-1) > 9 else None
            if len(signal.shape) == 3:
                B, L, _ = signal.shape
                rot = signal[..., 3:9].reshape(B * L, 6)
                mat = compute_rotation_matrix_from_ortho6d(rot)
                quat = matrix_to_quaternion(mat)
                quat = quat.reshape(B, L, 4)
            else:
                rot = signal[..., 3:9]
                mat = compute_rotation_matrix_from_ortho6d(rot)
                quat = matrix_to_quaternion(mat)
            signal = torch.cat([signal[..., :3], quat], dim=-1)
            if res is not None:
                signal = torch.cat((signal, res), -1)
            # The above code handled wxyz quaternion format!
            if self._quaternion_format == 'xyzw':
                signal[..., 3:7] = signal[..., (4, 5, 6, 3)]
        return signal

    def convert2rel(self, pcd, curr_gripper):
        """Convert coordinate system relaative to current gripper."""
        center = curr_gripper[:, -1, :3]  # (batch_size, 3)
        bs = center.shape[0]
        pcd = pcd - center.view(bs, 1, 3, 1, 1)
        curr_gripper = curr_gripper.clone()
        curr_gripper[..., :3] = curr_gripper[..., :3] - center.view(bs, 1, 3)
        return pcd, curr_gripper

    def forward(
        self,
        gt_trajectory,
        trajectory_mask,
        rgb_obs,
        pcd_obs,
        instruction,
        curr_gripper,
        run_inference=False,
        sample=None,
        return_features=False,
    ):
        """
        Arguments:
            gt_trajectory: (B, trajectory_length, 3+4+X)
            trajectory_mask: (B, trajectory_length)
            timestep: (B, 1)
            rgb_obs: (B, num_cameras, 3, H, W) in [0, 1]
            pcd_obs: (B, num_cameras, 3, H, W) in world coordinates
            instruction: (B, max_instruction_length, 512)
            curr_gripper: (B, nhist, 3+4+X)

        Note:
            Regardless of rotation parametrization, the input rotation
            is ALWAYS expressed as a quaternion form.
            The model converts it to 6D internally if needed.

        1. 如果 use_gat，先用 joints_coords 建 robot graph condition
        2. 如果 relative=True，把 point cloud 和 gripper pose 轉成相對座標
        3. 把 gt_trajectory 拆成：
        - SE(3) pose trajectory
        - gripper openness target
        4. inference 時呼叫 compute_trajectory()
        5. training 時 normalize position
        6. 根據 diffusion_space 選 Lie loss 或 Euclidean loss
        """


        if self.use_gat:
            ctx_feats, ctx, instr_feats, arm_feats, arm, fps_feats, fps = self.graph_planner.encode_inputs(
                sample['rgbs'], sample['pcds'],
                sample['instr'], sample['joints_coords']
            )
            data_list = self.graph_planner.build_local_graph(
                arm, arm_feats, fps, fps_feats,
            )
            batch_data = Batch.from_data_list(data_list)
            gat_embedding = self.gat_encoder(
                batch_data.x.to(gt_trajectory.device),
                batch_data.edge_index.to(gt_trajectory.device),
                batch_data.edge_attr.to(gt_trajectory.device),
                batch_data.batch.to(gt_trajectory.device)
            )
            condition = gat_embedding.to(gt_trajectory.device)
        else:
            condition = None

        if self._relative:
            pcd_obs, curr_gripper = self.convert2rel(pcd_obs, curr_gripper)
        if gt_trajectory is not None:
            gt_openess = gt_trajectory[..., 7:]
            gt_trajectory = gt_trajectory[..., :7]
        curr_gripper = curr_gripper[..., :7]

        # gt_trajectory is expected to be in the quaternion format
        if run_inference:
            traj = self.compute_trajectory(
                trajectory_mask,
                rgb_obs,
                pcd_obs,
                instruction,
                curr_gripper,
                condition
            )
            if return_features:
                return traj, condition
            return traj
        # Normalize all pos (shared by both diffusion-space variants).
        gt_trajectory = gt_trajectory.clone()
        pcd_obs = pcd_obs.clone()
        curr_gripper = curr_gripper.clone()
        gt_trajectory[:, :, :3] = self.normalize_pos(gt_trajectory[:, :, :3])
        pcd_obs = torch.permute(self.normalize_pos(
            torch.permute(pcd_obs, [0, 1, 3, 4, 2])
        ), [0, 1, 4, 2, 3])
        curr_gripper[..., :3] = self.normalize_pos(curr_gripper[..., :3])

        if self.diffusion_space == "euclidean":
            return self._compute_loss_euclidean(
                gt_trajectory, gt_openess, pcd_obs, curr_gripper,
                rgb_obs, instruction, condition,
            )
        return self._compute_loss_lie(
            gt_trajectory, gt_openess, pcd_obs, curr_gripper,
            rgb_obs, instruction, condition,
        )

    def _compute_loss_lie(self, gt_trajectory, gt_openess, pcd_obs,
                          curr_gripper, rgb_obs, instruction, condition):
        # Prepare inputs (Lie path keeps quaternion-format curr_gripper).
        fixed_inputs = self.encode_inputs(
            rgb_obs, pcd_obs, instruction, curr_gripper
        )

        # Condition on start-end pose
        cond_data = torch.zeros_like(gt_trajectory)
        cond_mask = torch.zeros_like(cond_data)
        cond_mask = cond_mask.bool()

        batch, seq_len, _ = gt_trajectory.shape
        device = gt_trajectory.device

        t = torch.randint(
            0,
            self.noise_scheduler.timesteps,
            (batch,)
        ).long()

        gt_trajectory = gt_trajectory.view(batch * seq_len, -1)
        r0_flat = lie_metrics.as_lie(gt_trajectory) # (B * seq_len, 3, 4)

        # Convert into tangent space for normalization
        r0_flat = lie_metrics.as_repr(r0_flat, "tan")

        # Normalize in tangent space
        r0_flat = self.normalize_r0(r0_flat)

        # Convert back to lie group
        r0_flat = lie_metrics.as_lie(r0_flat)

        zt_flat = NormalSE3._sample_unit(n=(batch * seq_len,)).to(device) # (B * seq_len, 3, 4)

        all_sqrt_alphas = torch.tensor(self.noise_scheduler.sqrt_alphas).to(device)
        sqrt_alphas_t = all_sqrt_alphas[t].unsqueeze(1)
        sqrt_alphas_t = sqrt_alphas_t.unsqueeze(1).repeat(1, seq_len, 1)
        sqrt_alphas_t = sqrt_alphas_t.view(batch * seq_len, -1)

        ta_flat = -zt_flat

        rt_flat = ops.add(r0_flat, SE3.exp_map(sqrt_alphas_t * zt_flat))

        r0_flat = lie_metrics.as_repr(r0_flat, "tan")
        zt_flat = lie_metrics.as_repr(zt_flat, "tan")
        rt_flat = lie_metrics.as_repr(rt_flat, "tan")
        ta_flat = lie_metrics.as_repr(ta_flat, "tan")

        rt = rt_flat.view(batch, seq_len, -1)
        ta = ta_flat.view(batch, seq_len, -1)

        noisy_trajectory = rt

        if self.loss_formulation == "new":
            # First-layer weighted-MSE loss (source branch: new_loss).
            pred = self.policy_forward_pass(
                noisy_trajectory, t.to(device), fixed_inputs, condition
            )[0]

            weights = torch.tensor(
                [20.0, 20.0, 20.0, 10.0, 10.0, 10.0], device=ta.device
            )

            ta = ta.view(batch * seq_len, -1)
            movement = pred[..., :6]
            movement = movement.view(batch * seq_len, -1)
            openness = pred[..., 6:]
            sq_error = (movement - ta) ** 2
            weighted_error = sq_error * weights
            loss_se3 = weighted_error.mean()
            loss_gripper = F.binary_cross_entropy_with_logits(openness, gt_openess)
            total_loss = loss_se3 + loss_gripper
            return total_loss, loss_se3.item(), loss_gripper.item()

        # "old" loss: per-layer MSE on (trans, rot) x [20.0, 10.0] +
        # per-layer BCE openness, summed across all prediction layers.
        # Source branch: gat-score-2enc.
        pred = self.policy_forward_pass(
            noisy_trajectory, t.to(device), fixed_inputs, condition
        )

        loss_se3 = 0.0
        loss_gripper = 0.0
        ta = ta.view(batch * seq_len, -1)
        for layer_pred in pred:
            movement = layer_pred[..., :6].reshape(batch * seq_len, -1)
            openness = layer_pred[..., 6:]
            loss_se3 = loss_se3 + (
                F.mse_loss(ta[..., :3], movement[..., :3]) * 20.0
                + F.mse_loss(ta[..., 3:], movement[..., 3:]) * 10.0
            )
            loss_gripper = loss_gripper + F.binary_cross_entropy_with_logits(
                openness, gt_openess
            )
        total_loss = loss_se3 + loss_gripper
        return (
            total_loss,
            loss_se3.item() if hasattr(loss_se3, "item") else float(loss_se3),
            loss_gripper.item() if hasattr(loss_gripper, "item") else float(loss_gripper),
        )

    def _compute_loss_euclidean(self, gt_trajectory, gt_openess, pcd_obs,
                                curr_gripper, rgb_obs, instruction, condition):
        # Convert quaternion -> 6D ortho (DDPM ablation predicts in this space).
        gt_trajectory = self.convert_rot(gt_trajectory)
        curr_gripper = self.convert_rot(curr_gripper)

        fixed_inputs = self.encode_inputs(
            rgb_obs, pcd_obs, instruction, curr_gripper
        )

        cond_data = torch.zeros_like(gt_trajectory)
        cond_mask = torch.zeros_like(cond_data).bool()

        # Sample noise + a random timestep, then add noise to clean trajectories.
        noise = torch.randn(gt_trajectory.shape, device=gt_trajectory.device)
        timesteps = torch.randint(
            0,
            self.position_noise_scheduler.config.num_train_timesteps,
            (len(noise),), device=noise.device,
        ).long()
        pos = self.position_noise_scheduler.add_noise(
            gt_trajectory[..., :3], noise[..., :3], timesteps
        )
        rot = self.rotation_noise_scheduler.add_noise(
            gt_trajectory[..., 3:9], noise[..., 3:9], timesteps
        )
        noisy_trajectory = torch.cat((pos, rot), -1)
        noisy_trajectory[cond_mask] = cond_data[cond_mask]
        assert not cond_mask.any()

        # Predict the noise residual (full per-layer list).
        pred = self.policy_forward_pass(
            noisy_trajectory, timesteps, fixed_inputs, condition,
        )

        # Per-layer L1 reconstruction + BCE openness, summed across layers.
        loss_se3 = 0.0
        loss_gripper = 0.0
        for layer_pred in pred:
            trans = layer_pred[..., :3]
            layer_rot = layer_pred[..., 3:9]
            loss_se3 = loss_se3 + (
                30 * F.l1_loss(trans, noise[..., :3], reduction='mean')
                + 10 * F.l1_loss(layer_rot, noise[..., 3:9], reduction='mean')
            )
            if torch.numel(gt_openess) > 0:
                openess = layer_pred[..., 9:]
                loss_gripper = loss_gripper + F.binary_cross_entropy_with_logits(
                    openess, gt_openess
                )
        total_loss = loss_se3 + loss_gripper
        return (
            total_loss,
            float(loss_se3) if isinstance(loss_se3, float) else loss_se3.item(),
            float(loss_gripper) if isinstance(loss_gripper, float) else loss_gripper.item(),
        )


class DiffusionHead(nn.Module):

    def __init__(self,
                 embedding_dim=60,
                 num_attn_heads=8,
                 use_instruction=False,
                 rotation_parametrization='quat',
                 nhist=3,
                 lang_enhanced=False,
                 use_gat=True,
                 diffusion_space="lie"):
        super().__init__()
        self.use_instruction = use_instruction
        self.lang_enhanced = lang_enhanced
        self.use_gat = use_gat
        self.diffusion_space = diffusion_space
        if '6D' in rotation_parametrization:
            rotation_dim = 6  # continuous 6D
        else:
            rotation_dim = 4  # quaternion

        # Encoders
        # In Lie/tangent space the trajectory is a 6D tangent vector; in
        # Euclidean DDPM space it's 3D position + 6D rotation = 9 features.
        traj_in_dim = 6 if diffusion_space == "lie" else 9
        self.traj_encoder = nn.Linear(traj_in_dim, embedding_dim)
        self.relative_pe_layer = RotaryPositionEncoding3D(embedding_dim)
        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(embedding_dim),
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        self.curr_gripper_emb = nn.Sequential(
            nn.Linear(embedding_dim * nhist, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        self.traj_time_emb = SinusoidalPosEmb(embedding_dim)

        # Attention from trajectory queries to language
        self.traj_lang_attention = nn.ModuleList([
            ParallelAttention(
                num_layers=1,
                d_model=embedding_dim, n_heads=num_attn_heads,
                self_attention1=False, self_attention2=False,
                cross_attention1=True, cross_attention2=False,
                rotary_pe=False, apply_ffn=False
            )
        ])

        # Estimate attends to context (no subsampling)
        self.cross_attn = FFWRelativeCrossAttentionModule(
            embedding_dim, num_attn_heads, num_layers=2, use_adaln=True
        )

        # GAT-conditioned cross-attention; only present in branches that
        # consume the GAT planner's output.
        if self.use_gat:
            self.cross_attn_condition = FFWRelativeCrossAttentionModule(
                embedding_dim, num_attn_heads, num_layers=1, use_adaln=True
            )

        # Shared attention layers
        if not self.lang_enhanced:
            self.self_attn = FFWRelativeSelfAttentionModule(
                embedding_dim, num_attn_heads, num_layers=4, use_adaln=True
            )
        else:  # interleave cross-attention to language
            self.self_attn = FFWRelativeSelfCrossAttentionModule(
                embedding_dim, num_attn_heads,
                num_self_attn_layers=4,
                num_cross_attn_layers=3,
                use_adaln=True
            )

        # Specific (non-shared) Output layers:
        # 1. Rotation
        self.rotation_proj = nn.Linear(embedding_dim, embedding_dim)
        if not self.lang_enhanced:
            self.rotation_self_attn = FFWRelativeSelfAttentionModule(
                embedding_dim, num_attn_heads, 2, use_adaln=True
            )
        else:  # interleave cross-attention to language
            self.rotation_self_attn = FFWRelativeSelfCrossAttentionModule(
                embedding_dim, num_attn_heads, 2, 1, use_adaln=True
            )
        # Lie path predicts the 3D tangent rotation directly; Euclidean DDPM
        # predicts the 6D ortho rotation (then converted via unconvert_rot).
        rot_pred_dim = 3 if diffusion_space == "lie" else 6
        self.rotation_predictor = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, rot_pred_dim)
        )

        # 2. Position
        self.position_proj = nn.Linear(embedding_dim, embedding_dim)
        if not self.lang_enhanced:
            self.position_self_attn = FFWRelativeSelfAttentionModule(
                embedding_dim, num_attn_heads, 2, use_adaln=True
            )
        else:  # interleave cross-attention to language
            self.position_self_attn = FFWRelativeSelfCrossAttentionModule(
                embedding_dim, num_attn_heads, 2, 1, use_adaln=True
            )
        self.position_predictor = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 3)
        )

        # 3. Openess
        self.openess_predictor = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1)
        )

        # Condition projection (paired with cross_attn_condition above).
        if self.use_gat:
            self.condition_proj = nn.Linear(512, embedding_dim)

    def forward(self, trajectory, timestep,
                context_feats, context, instr_feats, adaln_gripper_feats,
                fps_feats, fps_pos, condition=None):
        """
        Arguments:
            trajectory: (B, trajectory_length, 3+6+X)
            timestep: (B, 1)
            context_feats: (B, N, F)
            context: (B, N, F, 2)
            instr_feats: (B, max_instruction_length, F)
            adaln_gripper_feats: (B, nhist, F)
            fps_feats: (N, B, F), N < context_feats.size(1)
            fps_pos: (B, N, F, 2)
        """
        # Trajectory features
        traj_feats = self.traj_encoder(trajectory)  # (B, L, F)

        # Trajectory features cross-attend to context features
        traj_time_pos = self.traj_time_emb(
            torch.arange(0, traj_feats.size(1), device=traj_feats.device)
        )[None].repeat(len(traj_feats), 1, 1)
        if self.use_instruction:
            traj_feats, _ = self.traj_lang_attention[0](
                seq1=traj_feats, seq1_key_padding_mask=None,
                seq2=instr_feats, seq2_key_padding_mask=None,
                seq1_pos=None, seq2_pos=None,
                seq1_sem_pos=traj_time_pos, seq2_sem_pos=None
            )
        if condition is not None:
            condition = self.condition_proj(condition)
            # condition = condition.unsqueeze(1).expand(-1, traj_feats.size(1), -1)  # (B, N, F)
            # traj_feats_orig = traj_feats
            traj_feats = self.cross_attn_condition(
                query=traj_feats.transpose(0, 1),  # (N, B, F)
                value=condition.transpose(0, 1),  # (N, B, F)
                query_pos=None,
                value_pos=None,
                diff_ts=self.time_emb(timestep) # (B, F)
            )[-1].transpose(0, 1)  # (B, N, F)
            # traj_feats = traj_feats + traj_feats_orig
        traj_feats = traj_feats + traj_time_pos

        # Predict position, rotation, opening
        traj_feats = einops.rearrange(traj_feats, 'b l c -> l b c')
        context_feats = einops.rearrange(context_feats, 'b l c -> l b c')
        adaln_gripper_feats = einops.rearrange(
            adaln_gripper_feats, 'b l c -> l b c'
        )
        pos_pred, rot_pred, openess_pred = self.prediction_head(
            trajectory[..., :3], traj_feats,
            context[..., :3], context_feats,
            timestep, adaln_gripper_feats,
            fps_feats, fps_pos,
            instr_feats
        )
        return [torch.cat((pos_pred, rot_pred, openess_pred), -1)]
        # return [rot_pred]

    def prediction_head(self,
                        gripper_pcd, gripper_features,
                        context_pcd, context_features,
                        timesteps, curr_gripper_features,
                        sampled_context_features, sampled_rel_context_pos,
                        instr_feats):
        """
        Compute the predicted action (position, rotation, opening).

        Args:
            gripper_pcd: A tensor of shape (B, N, 3)
            gripper_features: A tensor of shape (N, B, F)
            context_pcd: A tensor of shape (B, N, 3)
            context_features: A tensor of shape (N, B, F)
            timesteps: A tensor of shape (B,) indicating the diffusion step
            curr_gripper_features: A tensor of shape (M, B, F)
            sampled_context_features: A tensor of shape (K, B, F)
            sampled_rel_context_pos: A tensor of shape (B, K, F, 2)
            instr_feats: (B, max_instruction_length, F)
        """
        # Diffusion timestep
        time_embs = self.encode_denoising_timestep(
            timesteps, curr_gripper_features
        )

        # Positional embeddings
        rel_gripper_pos = self.relative_pe_layer(gripper_pcd)
        rel_context_pos = self.relative_pe_layer(context_pcd)

        # Cross attention from gripper to full context
        gripper_features = self.cross_attn(
            query=gripper_features,
            value=context_features,
            query_pos=rel_gripper_pos,
            value_pos=rel_context_pos,
            diff_ts=time_embs
        )[-1]

        # Self attention among gripper and sampled context
        features = torch.cat([gripper_features, sampled_context_features], 0)
        rel_pos = torch.cat([rel_gripper_pos, sampled_rel_context_pos], 1)
        features = self.self_attn(
            query=features,
            query_pos=rel_pos,
            diff_ts=time_embs,
            context=instr_feats,
            context_pos=None
        )[-1]

        num_gripper = gripper_features.shape[0]

        # Rotation head
        rotation, rotation_features = self.predict_rot(
            features, rel_pos, time_embs, num_gripper, instr_feats
        )

        # Position head
        position, position_features = self.predict_pos(
            features, rel_pos, time_embs, num_gripper, instr_feats
        )

        # Openess head from position head
        # openess = self.openess_predictor(position_features)
        openess = self.openess_predictor(rotation_features)

        return position, rotation, openess

    def encode_denoising_timestep(self, timestep, curr_gripper_features):
        """
        Compute denoising timestep features and positional embeddings.

        Args:
            - timestep: (B,)

        Returns:
            - time_feats: (B, F)
        """
        time_feats = self.time_emb(timestep)

        curr_gripper_features = einops.rearrange(
            curr_gripper_features, "npts b c -> b npts c"
        )
        curr_gripper_features = curr_gripper_features.flatten(1)
        curr_gripper_feats = self.curr_gripper_emb(curr_gripper_features)
        return time_feats + curr_gripper_feats

    def predict_pos(self, features, rel_pos, time_embs, num_gripper,
                    instr_feats):
        position_features = self.position_self_attn(
            query=features,
            query_pos=rel_pos,
            diff_ts=time_embs,
            context=instr_feats,
            context_pos=None
        )[-1]
        position_features = einops.rearrange(
            position_features[:num_gripper], "npts b c -> b npts c"
        )
        position_features = self.position_proj(position_features)  # (B, N, C)
        position = self.position_predictor(position_features)
        return position, position_features

    def predict_rot(self, features, rel_pos, time_embs, num_gripper,
                    instr_feats):
        rotation_features = self.rotation_self_attn(
            query=features,
            query_pos=rel_pos,
            diff_ts=time_embs,
            context=instr_feats,
            context_pos=None
        )[-1]
        rotation_features = einops.rearrange(
            rotation_features[:num_gripper], "npts b c -> b npts c"
        )
        rotation_features = self.rotation_proj(rotation_features)  # (B, N, C)
        rotation = self.rotation_predictor(rotation_features)
        return rotation, rotation_features
