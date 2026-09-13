import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
import einops

from lda.gat.encoder import Encoder
from lda.gat.position_encodings import FourierEncode, BatchFourierEncode

class GraphPlanner(nn.Module):

    def __init__(
        self,
        backbone="clip",
        image_size=(256, 256),
        embedding_dim=60,
        num_vis_ins_attn_layers=2,
        use_instruction=False,
        fps_subsampling_factor=5,
        njoints=3,
        num_bands=10,
    ):
        super().__init__()
        self.encoder = Encoder(
            backbone=backbone,
            image_size=image_size,
            embedding_dim=embedding_dim,
            num_sampling_level=1,
            njoints=njoints,
            num_vis_ins_attn_layers=num_vis_ins_attn_layers,
            fps_subsampling_factor=fps_subsampling_factor,
        )
        self.band = num_bands
        self.use_instruction = use_instruction

    def encode_inputs(self, visible_rgb, visible_pcd, instruction, cur_arm):
        rgb_feats_pyramid, pcd_pyramid = self.encoder.encode_images(
            visible_rgb, visible_pcd
        )

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

        # (B, njoints, F)
        arm_feats, _ = self.encoder.encode_curr_arm(
            cur_arm, context_feats, context
        )
        arm_pos = cur_arm[..., :3]

        fps_feats, fps_pos = self.encoder.run_fps(
            context_feats.transpose(0, 1),
            context.transpose(0, 1),
            # self.encoder.relative_pe_layer(context)
        )

        return (
            context_feats, context,
            instr_feats,
            arm_feats, arm_pos,
            fps_feats, fps_pos
        )

    def build_local_graph(self, arm_pos, arm_feats, context, context_feats):
        """
        Build PyG Data objects representing local graphs with maximum parallelization
        - Nodes: arm_pos, context
        - Node Features: arm_feats, context_feats
        - Edges: arm_pos-arm_pos, gripper-context
        Args: 
            arm_pos: (B, njoints, 3)
            arm_feats: (B, njoints, F)
            context: (B, N, 3)
            context_feats: (B, N, F)
        Returns:
            List of PyG Data objects
        """
        B, N_a, _ = arm_pos.shape
        _, N_c, _ = context.shape
        device = arm_pos.device
        
        # Concatenate positions and features for all batches
        pos = torch.cat([arm_pos, context], dim=1)  # (B, N_a + N_c, 3)
        feats = torch.cat([arm_feats, context_feats], dim=1)  # (B, N_a + N_c, F)
        
        # Create node types for all batches
        node_type = torch.cat([
            torch.zeros(N_a, dtype=torch.long, device=device),
            torch.ones(N_c, dtype=torch.long, device=device),
        ]).unsqueeze(0).expand(B, -1)  # (B, N_a + N_c)
        
        # Create one-hot encoding
        type_onehot = F.one_hot(node_type, num_classes=2).to(torch.float)
        
        # Concatenate all node features
        x = torch.cat([feats, pos, type_onehot], dim=-1)  # (B, N_a + N_c, F + 5)
        
        # Pre-compute edge topology (same for all batches)
        edge_index = self._build_edge_topology(N_a, N_c, device)  # (2, E)
        
        # Compute all edge attributes in parallel
        edge_attr = self._compute_all_edge_attrs(arm_pos, context, N_a, N_c)  # (B, E, D)
        
        # Create Data objects
        data_list = []
        for b in range(B):
            data = Data(
                x=x[b],
                pos=pos[b],
                edge_index=edge_index,
                edge_attr=edge_attr[b],
                node_type=node_type[b],
            )
            data_list.append(data)
        
        return data_list

    def _build_edge_topology(self, N_a, N_c, device):
        """Build edge index tensor (same topology for all graphs)"""
        # Arm-arm edges
        arm_edges = torch.stack([
            torch.arange(N_a - 1, device=device),
            torch.arange(1, N_a, device=device)
        ], dim=0)
        
        # Gripper-context edges
        gripper_idx = N_a - 1
        gripper_context_edges = torch.stack([
            torch.full((N_c,), gripper_idx, device=device),
            torch.arange(N_a, N_a + N_c, device=device)
        ], dim=0)
        
        return torch.cat([arm_edges, gripper_context_edges], dim=1)

    def _compute_all_edge_attrs(self, arm_pos, context, N_a, N_c):
        """Compute edge attributes for all batches in parallel"""
        B = arm_pos.shape[0]
        gripper_idx = N_a - 1
        
        # Compute arm-arm relative positions for all batches
        arm_rels = arm_pos[:, 1:] - arm_pos[:, :-1]  # (B, N_a-1, 3)
        arm_rels_flat = arm_rels.reshape(-1, 3)  # (B*(N_a-1), 3)
        
        # Compute gripper-context relative positions for all batches
        gripper_pos = arm_pos[:, gripper_idx:gripper_idx+1]  # (B, 1, 3)
        gripper_context_rels = context - gripper_pos  # (B, N_c, 3)
        gripper_context_rels_flat = gripper_context_rels.reshape(-1, 3)  # (B*N_c, 3)
        
        # Batch encode all relative positions at once
        all_rels = torch.cat([arm_rels_flat, gripper_context_rels_flat], dim=0)
        all_edge_attrs = BatchFourierEncode(all_rels)  # (B*(N_a-1+N_c), D)
        
        # Reshape back to batch format
        D = all_edge_attrs.shape[1]
        arm_attrs = all_edge_attrs[:B*(N_a-1)].reshape(B, N_a-1, D)
        gripper_attrs = all_edge_attrs[B*(N_a-1):].reshape(B, N_c, D)
        
        # Concatenate arm and gripper-context edge attributes
        return torch.cat([arm_attrs, gripper_attrs], dim=1)  # (B, N_a-1+N_c, D)

    def policy_forward_pass(self, trajectory, timestep, fixed_inputs):
        """
        Predict the Noise for the given trajectory at a specific timestep.

        """
        pass

    def conditional_sample(self, condition_data, condition_mask, fixed_inputs):
        """
        
        """
        pass

    def compute_trajectory(
        self,
        trajectory_mask,
        rgb_obs,
        pcd_obs,
        instruction,
        curr_gripper,
    ):
        pass

    def normalize_pos(self, pos):
        pass

    def unnormalize_pos(self, pos):
        pass

    def convert_rot(self, signal):
        pass

    def unconvert_rot(self, signal):
        pass

    def convert2rel(self, pcd, curr_gripper):
        pass


    def forward(
        self,
        gt_trajectory,
        trajectory_mask,
        rgb_obs,
        pcd_obs,
        instruction,
        curr_gripper,
        run_inference=False
    ):
        pass