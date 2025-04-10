import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

from einops import rearrange
# from rotary_embedding_torch import RotaryEmbedding
from .CustomRotaryEmbedding import CustomRotaryEmbedding
from .modulated_convolustion import ModulatedConv1DLayer
from timm.models.layers import DropPath
from timm.models.vision_transformer import Mlp
from transformers import PretrainedConfig, PreTrainedModel

import torch.distributed as dist
import torch.nn.functional as F
from opensora.acceleration.checkpoint import auto_grad_checkpoint
from opensora.acceleration.communications import gather_forward_split_backward, split_forward_gather_backward
from opensora.acceleration.parallel_states import get_sequence_parallel_group
from opensora.models.layers.blocks import (
    Attention,
    CaptionEmbedder,
    MultiHeadCrossAttention,
    PatchEmbed3D,
    PositionEmbedding2D,
    CyclicPositionEmbedding2D,
    SeqParallelAttention,
    SeqParallelMultiHeadCrossAttention,
    SizeEmbedder,
    T2IFinalLayer,
    TimestepEmbedder,
    approx_gelu,
    get_2d_sincos_pos_embed,
    get_layernorm,
    t2i_modulate,
)
from opensora.registry import MODELS
from transformers import PretrainedConfig, PreTrainedModel
from opensora.utils.ckpt_utils import load_checkpoint








class STDiT2Block(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_ratio=4.0,
        drop_path=0.0,
        enable_flash_attn=False,
        enable_layernorm_kernel=False,
        enable_sequence_parallelism=False,
        rope=None,
        qk_norm=False,
        qk_norm_legacy=False,
        temporal=False,

    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.enable_flash_attn = enable_flash_attn
        self.temporal = temporal
        # TODO:
        self.enable_sequence_parallelism = enable_sequence_parallelism
        if self.enable_sequence_parallelism and not temporal:
            attn_cls = SeqParallelAttention
            self.attn = attn_cls(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            enable_flash_attn=enable_flash_attn,
            qk_norm=qk_norm,
            # qk_norm_legacy=qk_norm_legacy,
            )
            mha_cls = SeqParallelMultiHeadCrossAttention
        else:
            attn_cls = Attention
            self.attn = attn_cls(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            enable_flash_attn=enable_flash_attn,
            qk_norm=qk_norm,
            qk_norm_legacy=qk_norm_legacy,
            )
            mha_cls = MultiHeadCrossAttention

        # spatial branch
        self.norm1 = get_layernorm(hidden_size, eps=1e-6, affine=False, use_kernel=enable_layernorm_kernel)
        
        self.scale_shift_table = nn.Parameter(torch.randn(6, hidden_size) / hidden_size**0.5)

        # cross attn
        self.cross_attn = mha_cls(hidden_size, num_heads)

        # mlp branch
        self.norm2 = get_layernorm(hidden_size, eps=1e-6, affine=False, use_kernel=enable_layernorm_kernel)
        self.mlp = Mlp(
            in_features=hidden_size, hidden_features=int(hidden_size * mlp_ratio), act_layer=approx_gelu, drop=0
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        # temporal branch
        self.rope = rope
        self.norm_temp = get_layernorm(hidden_size, eps=1e-6, affine=False, use_kernel=enable_layernorm_kernel)  # new
        self.attn_temp = Attention(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            enable_flash_attn=self.enable_flash_attn,
            rope=rope,
            qk_norm=qk_norm,
            qk_norm_legacy=qk_norm_legacy,

        )
        self.scale_shift_table_temporal = nn.Parameter(torch.randn(3, hidden_size) / hidden_size**0.5)  # new

    def t_mask_select(self, x_mask, x, masked_x, T, S):
        # x: [B, (T, S), C]
        # mased_x: [B, (T, S), C]
        # x_mask: [B, T]
        x = rearrange(x, "B (T S) C -> B T S C", T=T, S=S)
        masked_x = rearrange(masked_x, "B (T S) C -> B T S C", T=T, S=S)
        x = torch.where(x_mask[:, :, None, None], x, masked_x)
        x = rearrange(x, "B T S C -> B (T S) C")
        return x

    def to_device(self, device):
        super().to(device)  # Move the module to the specified device
        if self.rope != None and hasattr(self.rope, 'freqs'):
            self.rope.freqs = self.rope.freqs.to(device)  # Move freqs to the correct device
            
    def forward(self, x, y, t, t_tmp, mask=None, x_mask=None, t0=None, t0_tmp=None, T=None, S=None, **kwargs):
        B, N, C = x.shape

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.scale_shift_table[None] + t.reshape(B, 6, -1)
        ).chunk(6, dim=1)
        shift_tmp, scale_tmp, gate_tmp = (self.scale_shift_table_temporal[None] + t_tmp.reshape(B, 3, -1)).chunk(
            3, dim=1
        )
        if x_mask is not None:
            shift_msa_zero, scale_msa_zero, gate_msa_zero, shift_mlp_zero, scale_mlp_zero, gate_mlp_zero = (
                self.scale_shift_table[None] + t0.reshape(B, 6, -1)
            ).chunk(6, dim=1)
            shift_tmp_zero, scale_tmp_zero, gate_tmp_zero = (
                self.scale_shift_table_temporal[None] + t0_tmp.reshape(B, 3, -1)
            ).chunk(3, dim=1)

        # modulate
        x_m = t2i_modulate(self.norm1(x), shift_msa, scale_msa)
        if x_mask is not None:
            x_m_zero = t2i_modulate(self.norm1(x), shift_msa_zero, scale_msa_zero)
            x_m = self.t_mask_select(x_mask, x_m, x_m_zero, T, S)

        # spatial branch
        x_s = rearrange(x_m, "B (T S) C -> (B T) S C", T=T, S=S)
        x_s = self.attn(x_s)
        x_s = rearrange(x_s, "(B T) S C -> B (T S) C", T=T, S=S)
        if x_mask is not None:
            x_s_zero = gate_msa_zero * x_s
            x_s = gate_msa * x_s
            x_s = self.t_mask_select(x_mask, x_s, x_s_zero, T, S)
        else:
            x_s = gate_msa * x_s
        x = x + self.drop_path(x_s)

        # modulate
        x_m = t2i_modulate(self.norm_temp(x), shift_tmp, scale_tmp)
        if x_mask is not None:
            x_m_zero = t2i_modulate(self.norm_temp(x), shift_tmp_zero, scale_tmp_zero)
            x_m = self.t_mask_select(x_mask, x_m, x_m_zero, T, S)

        # temporal branch
        x_t = rearrange(x_m, "B (T S) C -> (B S) T C", T=T, S=S)
        x_t = self.attn_temp(x_t)
        x_t = rearrange(x_t, "(B S) T C -> B (T S) C", T=T, S=S)
        if x_mask is not None:
            x_t_zero = gate_tmp_zero * x_t
            x_t = gate_tmp * x_t
            x_t = self.t_mask_select(x_mask, x_t, x_t_zero, T, S)
        else:
            x_t = gate_tmp * x_t
        x = x + self.drop_path(x_t)

        # cross attn
        x = x + self.cross_attn(x, y, mask)

        # modulate
        x_m = t2i_modulate(self.norm2(x), shift_mlp, scale_mlp)
        if x_mask is not None:
            x_m_zero = t2i_modulate(self.norm2(x), shift_mlp_zero, scale_mlp_zero)
            x_m = self.t_mask_select(x_mask, x_m, x_m_zero, T, S)

        # mlp
        x_mlp = self.mlp(x_m)
        if x_mask is not None:
            x_mlp_zero = gate_mlp_zero * x_mlp
            x_mlp = gate_mlp * x_mlp
            x_mlp = self.t_mask_select(x_mask, x_mlp, x_mlp_zero, T, S)
        else:
            x_mlp = gate_mlp * x_mlp
        x = x + self.drop_path(x_mlp)

        return x


class STDiT2Config(PretrainedConfig):
    
    model_type = "STDiT2"

    def __init__(
        self,
        input_size=(None, None, None),
        input_sq_size=32,
        in_channels=4,
        patch_size=(1, 2, 2),
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        pred_sigma=True,
        drop_path=0.0,
        no_temporal_pos_emb=False,
        caption_channels=4096,
        model_max_length=120,
        freeze=None,
        qk_norm=False,
        qk_norm_legacy=False,
        enable_sequence_parallelism=False,
        enable_flash_attn=False,
        enable_layernorm_kernel=False,
        enable_condition=False,
        condition_downsample=False,
        controlnet='dive',
        duplicated_depth=14,
        scale_factor=1.0,
        use_separate_scales=False,
        cyclic_pos_emb=False,
        task=2,
        **kwargs,
    ):
        self.input_size = input_size
        self.input_sq_size = input_sq_size
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.class_dropout_prob = class_dropout_prob
        self.pred_sigma = pred_sigma
        self.drop_path = drop_path
        self.no_temporal_pos_emb = no_temporal_pos_emb
        self.caption_channels = caption_channels
        self.model_max_length = model_max_length
        self.freeze = freeze
        self.qk_norm = qk_norm
        self.qk_norm_legacy = qk_norm_legacy
        self.enable_sequence_parallelism = enable_sequence_parallelism
        self.enable_flash_attn = enable_flash_attn
        self.enable_layernorm_kernel = enable_layernorm_kernel
        self.enable_condition = enable_condition
        self.scale_factor = scale_factor
        self.condition_downsample = condition_downsample
        self.controlnet = controlnet
        self.duplicated_depth = duplicated_depth
        self.use_separate_scales = use_separate_scales
        self.cyclic_pos_emb = cyclic_pos_emb
        self.task = task
        super().__init__(**kwargs)


@MODELS.register_module()
class STDiT2(PreTrainedModel):

    config_class = STDiT2Config

    def __init__(
        self,
        config
    ):
        super().__init__(config)
        self.pred_sigma = config.pred_sigma
        self.in_channels = config.in_channels
        self.out_channels = config.in_channels * 2 if config.pred_sigma else config.in_channels
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.no_temporal_pos_emb = config.no_temporal_pos_emb
        self.depth = config.depth
        self.mlp_ratio = config.mlp_ratio
        self.enable_flash_attn = config.enable_flash_attn
        self.enable_layernorm_kernel = config.enable_layernorm_kernel
        
        # computation related
        self.enable_sequence_parallelism = config.enable_sequence_parallelism
        
        # condition control
        self.enable_condition = config.enable_condition
        self.condition_downsample = config.condition_downsample
        self.scale_factor = config.scale_factor
        self.controlnet = config.controlnet
        self.duplicated_depth = config.duplicated_depth
        self.task = config.task
        
        # pos embedding scale 
        self.use_separate_scales = config.use_separate_scales
        self.cyclic_pos_emb = config.cyclic_pos_emb
        if self.cyclic_pos_emb:
            self.pos_embed = CyclicPositionEmbedding2D(config.hidden_size)
        else:
            self.pos_embed = PositionEmbedding2D(config.hidden_size)

        # support dynamic input
        self.patch_size = config.patch_size
        self.input_size = config.input_size
        self.input_sq_size = config.input_sq_size
        
        

        self.x_embedder = PatchEmbed3D(config.patch_size, config.in_channels, config.hidden_size)
        self.t_embedder = TimestepEmbedder(config.hidden_size)
        self.t_block = nn.Sequential(nn.SiLU(), nn.Linear(config.hidden_size, 6 * config.hidden_size, bias=True))
        self.t_block_temp = nn.Sequential(nn.SiLU(), nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=True))  # new
        self.y_embedder = CaptionEmbedder(
            in_channels=config.caption_channels,
            hidden_size=config.hidden_size,
            uncond_prob=config.class_dropout_prob,
            act_layer=approx_gelu,
            token_num=config.model_max_length,
        )

        drop_path = [x.item() for x in torch.linspace(0, config.drop_path, config.depth)]
        # print("drop_path: ", drop_path) # default: 0.0
        self.rope = CustomRotaryEmbedding(dim=self.hidden_size // self.num_heads)  # new
        self.blocks = nn.ModuleList(
            [
                STDiT2Block(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=self.mlp_ratio,
                    drop_path=drop_path[i],
                    enable_flash_attn=self.enable_flash_attn,
                    enable_layernorm_kernel=self.enable_layernorm_kernel,
                    rope=self.rope.rotate_queries_or_keys,
                    qk_norm=config.qk_norm,
                    qk_norm_legacy=config.qk_norm_legacy,
                    enable_sequence_parallelism=config.enable_sequence_parallelism,
                )
                for i in range(self.depth)
            ]
        )
        
        
     
        # controlnet: duplicated spatial blocks
        # Copy the first 13 blocks including their initial weights
        self.duplicated_blocks = nn.ModuleList(
            [copy.deepcopy(self.blocks[i]) for i in range(self.duplicated_depth)] # .to_device(next(self.blocks[i].parameters()).device)
        )

        # Add zero-initialized linear layers
        # TODO:
        self.zero_linear_layers = nn.ModuleList(
            [nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size) for _ in range(self.duplicated_depth + 1)]
        ) 
        
        self.zero_linear_layers_traj = nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size)
        
        

        self.zero_conv = nn.ModuleList(
            [ModulatedConv1DLayer(config) for _ in range(self.duplicated_depth + 1)]
        )
        
        self.style = nn.ParameterList(
            [nn.Parameter(torch.Tensor(config.hidden_size)) for _ in range(self.task)]
        )        
        for param in self.style:
            nn.init.normal_(param, mean=0.0, std=1.0)  # 使用正态分布初始化
        
        # self.gamma_pos = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        
        
        self.final_layer = T2IFinalLayer(config.hidden_size, np.prod(self.patch_size), self.out_channels)

        # multi_res
        assert self.hidden_size % 3 == 0, "hidden_size must be divisible by 3"
        self.csize_embedder = SizeEmbedder(self.hidden_size // 3)
        self.ar_embedder = SizeEmbedder(self.hidden_size // 3)
        self.fl_embedder = SizeEmbedder(self.hidden_size)  # new
        self.fps_embedder = SizeEmbedder(self.hidden_size)  # new

        # init model
        self.initialize_weights()
        self.initialize_temporal()
        
        #freeze
        if config.freeze is not None:
            assert config.freeze in ["not_temporal", "text", "pos_emb", "blocks"]
            if config.freeze == "not_temporal":
                self.freeze_not_temporal()
            if config.freeze == "text":
                self.freeze_text()
            if config.freeze == "pos_emb":
                self.finetune_pos_emb()
            if config.freeze == "blocks":
                self.freeze_block()
                
        

    def get_dynamic_size(self, x):
        _, _, T, H, W = x.size()
        if T % self.patch_size[0] != 0:
            T += self.patch_size[0] - T % self.patch_size[0]
        if H % self.patch_size[1] != 0:
            H += self.patch_size[1] - H % self.patch_size[1]
        if W % self.patch_size[2] != 0:
            W += self.patch_size[2] - W % self.patch_size[2]
        T = T // self.patch_size[0]
        H = H // self.patch_size[1]
        W = W // self.patch_size[2]
        return (T, H, W)

    def forward(
        self, x, timestep, y, 
            mask=None, x_mask=None, num_frames=None,
            height=None, width=None, ar=None, fps=None, 
            full_height=None, full_width=None,   
            layout_conditions: torch.Tensor = None, 
            traj_conditions: torch.Tensor = None, 
            **kwargs):
        """
        Forward pass of STDiT.
        Args:
            x (torch.Tensor): latent representation of video; of shape [B, C, T, H, W]
            timestep (torch.Tensor): diffusion time steps; of shape [B]
            y (torch.Tensor): representation of prompts; of shape [B, 1, N_token, C]
            mask (torch.Tensor): mask for selecting prompt tokens; of shape [B, N_token]

        Returns:
            x (torch.Tensor): output latent representation; of shape [B, C, T, H, W]
        """
        B = x.shape[0]
        dtype = self.x_embedder.proj.weight.dtype
        x = x.to(dtype)
        timestep = timestep.to(dtype)
        y = y.to(dtype)
        
   
        
        
   

        # === process data info ===
        # 1. get dynamic size
        
        hw = torch.cat([height[:, None], width[:, None]], dim=1)
        # hw = torch.cat([full_height[:, None], full_width[:, None]], dim=1)
        csize = self.csize_embedder(hw, B)

        # 2. get aspect ratio
        ar = ar.unsqueeze(1)
        ar = self.ar_embedder(ar, B)
        data_info = torch.cat([csize, ar], dim=1)

        # 3. get number of frames
        fl = num_frames.unsqueeze(1)
        fps = fps.unsqueeze(1)
        fl = self.fl_embedder(fl, B)
        fl = fl + self.fps_embedder(fps, B)

        # === get dynamic shape size ===
        _, _, Tx, Hx, Wx = x.size()
        T, H, W = self.get_dynamic_size(x)
        
        # adjust for sequence parallelism
        # we need to ensure H * W is divisible by sequence parallel size
        # for simplicity, we can adjust the height to make it divisible
        if self.enable_sequence_parallelism:
            sp_size = dist.get_world_size(get_sequence_parallel_group())
            if H % sp_size != 0:
                h_pad_size = sp_size - H % sp_size
            else:
                h_pad_size = 0

            if h_pad_size > 0:
                hx_pad_size = h_pad_size * self.patch_size[1]

                # pad x along the H dimension
                H += h_pad_size
                x = F.pad(x, (0, 0, 0, hx_pad_size))
                
        S = H * W
       
        # 2-dim scale pos_embedding
        if self.use_separate_scales:
            base_size = (H,W) 
            scale = (full_height[0].item() / self.input_sq_size, full_width[0].item() / self.input_sq_size) # self.input_sq_size: 512
        else:
            base_size = round(S**0.5)
            resolution_sq = (full_height[0].item() * full_width[0].item()) ** 0.5
            scale = resolution_sq / self.input_sq_size 
        
        pos_emb = self.pos_embed(x, H, W, scale=scale, base_size=base_size, use_separate_scales=self.use_separate_scales) # torch.Size([1, H*W, hidden_feature])

        if self.enable_condition:
            condition_pos_emb = self.pos_embed(layout_conditions.to(dtype), H, W, scale=scale, base_size=base_size, use_separate_scales=self.use_separate_scales) # torch.Size([1, H*W, hidden_feature])
            
            
        # embedding
        x = self.x_embedder(x)  # [B, THW, F]
        x = rearrange(x, "B (T S) C -> B T S C", T=T, S=S)
        x = x + pos_emb
        
        # if self.enable_condition:
        #     lay_out_condition = layout_conditions.to(x.device)
        #     T_con, H_con, W_con = self.get_dynamic_size(lay_out_condition) 
        #     lay_out_condition = auto_grad_checkpoint(self.x_embedder, lay_out_condition) # [B, C', T', H, W] -> [B, (T'S), F]
        
        # shard over the sequence dim if sp is enabled
        if self.enable_sequence_parallelism:
            x = split_forward_gather_backward(x, get_sequence_parallel_group(), dim=2, grad_scale="down")
            
            # if self.enable_condition:
            #     lay_out_condition = rearrange(lay_out_condition, "B (T S) C -> B T S C", T=T, S=S)
            #     lay_out_condition = split_forward_gather_backward(lay_out_condition, get_sequence_parallel_group(), dim=2, grad_scale="down")
            S = S // dist.get_world_size(get_sequence_parallel_group())
            
        x = rearrange(x, "B T S C -> B (T S) C")
        # if self.enable_sequence_parallelism:
        #     if self.enable_condition:
        #         lay_out_condition = rearrange(lay_out_condition, "B T S C -> B (T S) C")
        

        # prepare adaIN
        t = self.t_embedder(timestep, dtype=x.dtype)  # [B, C]
        t_spc = t + data_info  # [B, C]
        t_tmp = t + fl  # [B, C]
        t_spc_mlp = self.t_block(t_spc)  # [B, 6*C]
        t_tmp_mlp = self.t_block_temp(t_tmp)  # [B, 3*C]
        if x_mask is not None:
            t0_timestep = torch.zeros_like(timestep)
            t0 = self.t_embedder(t0_timestep, dtype=x.dtype)
            t0_spc = t0 + data_info
            t0_tmp = t0 + fl
            t0_spc_mlp = self.t_block(t0_spc)
            t0_tmp_mlp = self.t_block_temp(t0_tmp)
        else:
            t0_spc = None
            t0_tmp = None
            t0_spc_mlp = None
            t0_tmp_mlp = None

        # prepare y
        
        y = self.y_embedder(y, self.training)  # [B, 1, N_token, C]

        if mask is not None:
            if mask.shape[0] != y.shape[0]:
                mask = mask.repeat(y.shape[0] // mask.shape[0], 1)
            mask = mask.squeeze(1).squeeze(1)
            y = y.squeeze(1).masked_select(mask.unsqueeze(-1) != 0).view(1, -1, x.shape[-1])
            y_lens = mask.sum(dim=1).tolist()
        else:
            y_lens = [y.shape[2]] * y.shape[0]
            y = y.squeeze(1).view(1, -1, x.shape[-1])

        
        
        
        # blocks
        
        if self.controlnet == 'dive':
            for idx, block in enumerate(self.blocks):
                x = auto_grad_checkpoint(block, x, y, t_spc_mlp, t_tmp_mlp, y_lens, x_mask, t0_spc_mlp, t0_tmp_mlp, T, S) # x.shape: [B, N, C]
                if self.enable_condition:
                    if idx >= 0 and idx < len(self.duplicated_blocks):
                        if idx == 0:
                            lay_out_condition = layout_conditions.to(x.device)
                            lay_out_condition = auto_grad_checkpoint(self.x_embedder, lay_out_condition) # [B, C', T, H, W] -> [B, (TS), F]
                            lay_out_condition = auto_grad_checkpoint(self.zero_linear_layers[0], lay_out_condition)
                            x_controlnet = lay_out_condition
                        x_controlnet = auto_grad_checkpoint(self.duplicated_blocks[idx], x_controlnet + x, y, t_spc_mlp, t_tmp_mlp, y_lens, x_mask, t0_spc_mlp, t0_tmp_mlp, T, S)
                        # zero linear
                        x = auto_grad_checkpoint(self.zero_linear_layers[idx+1], x_controlnet) + x

        
        elif self.controlnet == 'dive_pos':
            for idx, block in enumerate(self.blocks):
                x = auto_grad_checkpoint(block, x, y, t_spc_mlp, t_tmp_mlp, y_lens, x_mask, t0_spc_mlp, t0_tmp_mlp, T, S) # x.shape: [B, N, C]
                if self.enable_condition:
                    if idx >= 0 and idx < len(self.duplicated_blocks):
                        if idx == 0:
                            lay_out_condition = layout_conditions.to(x.device)
                            lay_out_condition = auto_grad_checkpoint(self.x_embedder, lay_out_condition) # [B, C', T, H, W] -> [B, (TS), F]
                            lay_out_condition = auto_grad_checkpoint(self.zero_linear_layers[0], lay_out_condition)
                            # lay_out_condition = rearrange(lay_out_condition, "B (T S) C -> B T S C", T=T, S=S)
                            # lay_out_condition = lay_out_condition + condition_pos_emb * self.gamma_pos
                            # lay_out_condition = rearrange(lay_out_condition, "B T S C -> B (T S) C")
                            x_controlnet = lay_out_condition
                        x_controlnet = auto_grad_checkpoint(self.duplicated_blocks[idx], x_controlnet + x, y, t_spc_mlp, t_tmp_mlp, y_lens, x_mask, t0_spc_mlp, t0_tmp_mlp, T, S)
                        # zero linear
                        x = auto_grad_checkpoint(self.zero_linear_layers[idx+1], x_controlnet) + x

        
        elif self.controlnet == 'traj_dive_pos_one_pipeline':
            for idx, block in enumerate(self.blocks):
                
                x = auto_grad_checkpoint(block, x, y, t_spc_mlp, t_tmp_mlp, y_lens, x_mask, t0_spc_mlp, t0_tmp_mlp, T, S) # x.shape: [B, N, C]
                if self.enable_condition:
                    if idx >= 0 and idx < len(self.duplicated_blocks):
                        if idx == 0:
                            # 1. hdmap + bbox
                            layout_condition = layout_conditions.to(x.device)
                            layout_condition = auto_grad_checkpoint(self.x_embedder, layout_condition) # [B, C', T, H, W] -> [B, (TS), F]
                            layout_condition = auto_grad_checkpoint(self.zero_linear_layers[0], layout_condition)
                            # layout_condition = rearrange(layout_condition, "B (T S) C -> B T S C", T=T, S=S)
                            # 2. traj
                            traj_condition = traj_conditions.to(x.device)
                            traj_condition = auto_grad_checkpoint(self.x_embedder, traj_condition) # [B, C', T, H, W] -> [B, (TS), F]
                            traj_condition = auto_grad_checkpoint(self.zero_linear_layers_traj, traj_condition)
                            # traj_condition = rearrange(traj_condition, "B (T S) C -> B T S C", T=T, S=S)
                            # 3. condition_pos: 
                            # condition_pos_emb * self.gamma_pos
                            # 4. === add ===
                            all_condition = layout_condition + traj_condition # + condition_pos_emb * self.gamma_pos
                            # all_condition = rearrange(all_condition, "B T S C -> B (T S) C")
                            x_controlnet = all_condition
                        x_controlnet = auto_grad_checkpoint(self.duplicated_blocks[idx], x_controlnet + x, y, t_spc_mlp, t_tmp_mlp, y_lens, x_mask, t0_spc_mlp, t0_tmp_mlp, T, S)
                        # zero linear
                        x = auto_grad_checkpoint(self.zero_linear_layers[idx+1], x_controlnet) + x
        
        
        elif self.controlnet == 'traj_dive':
            for idx, block in enumerate(self.blocks):
                x = auto_grad_checkpoint(block, x, y, t_spc_mlp, t_tmp_mlp, y_lens, x_mask, t0_spc_mlp, t0_tmp_mlp, T, S) # x.shape: [B, N, C]
                if self.enable_condition:
                    if idx >= 0 and idx < len(self.duplicated_blocks):
                        if idx == 0:
                            lay_out_condition = layout_conditions.to(x.device)
                            lay_out_condition = auto_grad_checkpoint(self.x_embedder, lay_out_condition) # [B, C', T, H, W] -> [B, (TS), F]
                            lay_out_condition = auto_grad_checkpoint(self.zero_conv[0], lay_out_condition, self.style[0])
                            
                            traj_condition = traj_conditions.to(x.device)
                            traj_condition = auto_grad_checkpoint(self.x_embedder, traj_condition) # [B, C', T, H, W] -> [B, (TS), F]
                            traj_condition = auto_grad_checkpoint(self.zero_conv[0], traj_condition, self.style[1])
                            # x_controlnet = lay_out_condition
                        lay_out_condition = auto_grad_checkpoint(self.duplicated_blocks[idx], lay_out_condition + x, y, t_spc_mlp, t_tmp_mlp, y_lens, x_mask, t0_spc_mlp, t0_tmp_mlp, T, S)
                        traj_condition = auto_grad_checkpoint(self.duplicated_blocks[idx], traj_condition + x, y, t_spc_mlp, t_tmp_mlp, y_lens, x_mask, t0_spc_mlp, t0_tmp_mlp, T, S)
                        
                        # zero linear
                        x = auto_grad_checkpoint(self.zero_conv[idx+1], lay_out_condition, self.style[0])\
                            + auto_grad_checkpoint(self.zero_conv[idx+1], traj_condition, self.style[1])\
                                + x
                                
        
        
        elif self.controlnet == 'traj_controlnet': # controlnet-based
            for idx, block in enumerate(self.blocks):
                x_origin = auto_grad_checkpoint(block, x, y, t_spc_mlp, t_tmp_mlp, y_lens, x_mask, t0_spc_mlp, t0_tmp_mlp, T, S) # [B, (TS), F]
                if self.enable_condition and (idx >= 0 and idx < len(self.duplicated_blocks)):
        
                    if idx == 0:
                        T_con, H_con, W_con = self.get_dynamic_size(layout_conditions) 
                        lay_out_condition = layout_conditions.to(x.device)
                        lay_out_condition = auto_grad_checkpoint(self.x_embedder, lay_out_condition) # [B, C', T, H, W] -> [B, (TS), F]
                        lay_out_condition = auto_grad_checkpoint(self.zero_conv[0], lay_out_condition, self.style[0]) # [B, (TS), F]
                        lay_out_condition = auto_grad_checkpoint(self.duplicated_blocks[idx], lay_out_condition + x, y, t_spc_mlp, t_tmp_mlp, y_lens, x_mask, t0_spc_mlp, t0_tmp_mlp, T_con, H_con*W_con)
                        
                        traj_condition = traj_conditions.to(x.device)
                        traj_condition = auto_grad_checkpoint(self.x_embedder, traj_condition) # [B, C', T, H, W] -> [B, (TS), F]
                        traj_condition = auto_grad_checkpoint(self.zero_conv[0], traj_condition, self.style[1]) # [B, (TS), F]
                        traj_condition = auto_grad_checkpoint(self.duplicated_blocks[idx], traj_condition + x, y, t_spc_mlp, t_tmp_mlp, y_lens, x_mask, t0_spc_mlp, t0_tmp_mlp, T_con, H_con*W_con)
                    else:
                        lay_out_condition = auto_grad_checkpoint(self.duplicated_blocks[idx], lay_out_condition, y, t_spc_mlp, t_tmp_mlp, y_lens, x_mask, t0_spc_mlp, t0_tmp_mlp, T_con, H_con*W_con)
                        traj_condition = auto_grad_checkpoint(self.duplicated_blocks[idx], traj_condition, y, t_spc_mlp, t_tmp_mlp, y_lens, x_mask, t0_spc_mlp, t0_tmp_mlp, T_con, H_con*W_con)
                    
                    x = auto_grad_checkpoint(self.zero_conv[idx+1], lay_out_condition, self.style[0])\
                            + auto_grad_checkpoint(self.zero_conv[idx+1], traj_condition, self.style[1])\
                            + x_origin
                else:
                    x = x_origin
        
        elif self.controlnet == 'controlnet':
            # blocks
            for idx, block in enumerate(self.blocks):
                x_origin = auto_grad_checkpoint(block, x, y, t_spc_mlp, t_tmp_mlp, y_lens, x_mask, t0_spc_mlp, t0_tmp_mlp, T, S) # [B, (TS), F]
                if self.enable_condition and (idx >= 0 and idx < len(self.duplicated_blocks)):
        
                    if idx == 0:
                        lay_out_condition = layout_conditions.to(x.device)
                        T_con, H_con, W_con = self.get_dynamic_size(lay_out_condition) 
                        lay_out_condition = auto_grad_checkpoint(self.x_embedder, lay_out_condition) # [B, C', T, H, W] -> [B, (TS), F]
                        lay_out_condition = auto_grad_checkpoint(self.zero_linear_layers[0], lay_out_condition) # [B, (TS), F]
                        x_controlnet = lay_out_condition
                        
                        if self.condition_downsample:
                            # x downsample
                            x = x.reshape(B, self.hidden_size, -1)
                            x = x.reshape(B, self.hidden_size, T, H, W)
                            x = x.reshape(B * T, self.hidden_size, H, W)
                            x = F.interpolate(x, scale_factor=self.scale_factor, mode='nearest') 
                            x = x.reshape(B, -1, self.hidden_size) # [B, (TS), F]

                        x_controlnet = auto_grad_checkpoint(self.duplicated_blocks[idx], x_controlnet + x, y, t_spc_mlp, t_tmp_mlp, y_lens, x_mask, t0_spc_mlp, t0_tmp_mlp, T_con, H_con*W_con)
                    else:
                        x_controlnet = auto_grad_checkpoint(self.duplicated_blocks[idx], x_controlnet, y, t_spc_mlp, t_tmp_mlp, y_lens, x_mask, t0_spc_mlp, t0_tmp_mlp, T_con, H_con*W_con)
                    
                    # zero linear
                    # upsample
                    if self.condition_downsample:
                        x_controlnet_upsample = x_controlnet.clone()
                        x_controlnet_upsample = x_controlnet_upsample.reshape(B, self.hidden_size, -1)
                        x_controlnet_upsample = x_controlnet_upsample.reshape(B, self.hidden_size, T_con, H_con, W_con)
                        x_controlnet_upsample = x_controlnet_upsample.reshape(B * T_con, self.hidden_size, H_con, W_con)
                        x_controlnet_upsample = F.interpolate(x_controlnet_upsample, scale_factor=1/self.scale_factor, mode='nearest') 
                        x_controlnet_upsample = x_controlnet_upsample.reshape(B, -1, self.hidden_size) # [B, (TS), F]
                        x = auto_grad_checkpoint(self.zero_linear_layers[idx+1], x_controlnet_upsample) + x_origin
                    else:
                        x = auto_grad_checkpoint(self.zero_linear_layers[idx+1], x_controlnet) + x_origin
                else:
                    x = x_origin

        elif self.controlnet == 'sd':
            if self.enable_condition:
                duplicated_conditions = []
                for idx, duplicated_block in enumerate(self.duplicated_blocks):
                    if idx == 0:
                        lay_out_condition = layout_conditions.to(x.device) 
                        T_con, H_con, W_con = self.get_dynamic_size(lay_out_condition) 
                        lay_out_condition = auto_grad_checkpoint(self.x_embedder, lay_out_condition) # [B, C', T', H, W] -> [B, (T'S), F]
                        lay_out_condition = auto_grad_checkpoint(self.zero_linear_layers[0], lay_out_condition)
                        x_controlnet = lay_out_condition
                        if self.condition_downsample:
                            x_downsample = x.clone()
                            x_downsample = x_downsample.reshape(B * T, self.hidden_size, H, W)
                            x_downsample = F.interpolate(x_downsample, scale_factor=self.scale_factor, mode='nearest') 
                            x_downsample = x_downsample.reshape(B, -1, self.hidden_size) # [B, (TS), F]
                            x_controlnet = auto_grad_checkpoint(self.duplicated_blocks[idx], x_controlnet + x_downsample, y, t_spc_mlp, t_tmp_mlp, y_lens, x_mask, t0_spc_mlp, t0_tmp_mlp, T_con, H_con*W_con)
                        else:
                            x_controlnet = auto_grad_checkpoint(self.duplicated_blocks[idx], x_controlnet + x, y, t_spc_mlp, t_tmp_mlp, y_lens, x_mask, t0_spc_mlp, t0_tmp_mlp, T_con, H_con*W_con)
                    else:
                        x_controlnet = auto_grad_checkpoint(self.duplicated_blocks[idx], x_controlnet, y, t_spc_mlp, t_tmp_mlp, y_lens, x_mask, t0_spc_mlp, t0_tmp_mlp, T_con, H_con*W_con)
                    # zero linear
                    # save in duplicated_conditions
                    zero_linear_output = auto_grad_checkpoint(self.zero_linear_layers[idx+1], x_controlnet)
                    if self.condition_downsample:
                        # upsample
                        zero_linear_output = zero_linear_output.reshape(B * T_con, self.hidden_size, H_con, W_con)
                        zero_linear_output = F.interpolate(zero_linear_output, scale_factor=1/self.scale_factor, mode='nearest') 
                        zero_linear_output = zero_linear_output.reshape(B, -1, self.hidden_size) # [B, (TS), F]
                    duplicated_conditions.append(zero_linear_output.detach().clone())
                
            for idx, block in enumerate(self.blocks):
                if self.enable_condition:
                    if idx >= 0 and idx < len(self.duplicated_blocks):
                        x = auto_grad_checkpoint(block, x, y, t_spc_mlp, t_tmp_mlp, y_lens, x_mask, t0_spc_mlp, t0_tmp_mlp, T, S) # x.shape: [B, N, C]    
                    
                    if idx >= len(self.duplicated_blocks) and idx < len(self.blocks):
                        x = auto_grad_checkpoint(block, x + duplicated_conditions[len(self.blocks)-idx-1], y, t_spc_mlp, t_tmp_mlp, y_lens, x_mask, t0_spc_mlp, t0_tmp_mlp, T, S) # x.shape: [B, N, C]    
                else:
                    x = auto_grad_checkpoint(block, x, y, t_spc_mlp, t_tmp_mlp, y_lens, x_mask, t0_spc_mlp, t0_tmp_mlp, T, S) # x.shape: [B, N, C]    
        
        else:
            raise ValueError("model.controlnet has wrong value: " + self.controlnet)
        
        '''
        elif self.controlnet == 'sd':
            if self.enable_condition:
                duplicated_conditions = []
                for idx, duplicated_block in enumerate(self.duplicated_blocks):
                    if idx == 0:
                        lay_out_condition = layout_conditions.to(x.device)
                        lay_out_condition = auto_grad_checkpoint(self.x_embedder, lay_out_condition) # [B, C', T', H, W] -> [B, (T'S), F]
                        lay_out_condition = auto_grad_checkpoint(self.zero_linear_layers[0], lay_out_condition)
                        x_controlnet = lay_out_condition
                        x_controlnet = auto_grad_checkpoint(self.duplicated_blocks[idx], x_controlnet + x, y, t_spc_mlp, t_tmp_mlp, y_lens, x_mask, t0_spc_mlp, t0_tmp_mlp, T, S)
                    else:
                        x_controlnet = auto_grad_checkpoint(self.duplicated_blocks[idx], x_controlnet, y, t_spc_mlp, t_tmp_mlp, y_lens, x_mask, t0_spc_mlp, t0_tmp_mlp, T, S)
                    # zero linear
                    # save in duplicated_conditions
                    zero_linear_output = auto_grad_checkpoint(self.zero_linear_layers[idx+1], x_controlnet)
                    duplicated_conditions.append(zero_linear_output.detach().clone())
                
            for idx, block in enumerate(self.blocks):
                if self.enable_condition:
                    if idx >= 0 and idx < len(self.duplicated_blocks):
                        x = auto_grad_checkpoint(block, x, y, t_spc_mlp, t_tmp_mlp, y_lens, x_mask, t0_spc_mlp, t0_tmp_mlp, T, S) # x.shape: [B, N, C]    
                    
                    if idx >= len(self.duplicated_blocks) and idx < len(self.blocks):
                        x = auto_grad_checkpoint(block, x + duplicated_conditions[len(self.blocks)-idx-1], y, t_spc_mlp, t_tmp_mlp, y_lens, x_mask, t0_spc_mlp, t0_tmp_mlp, T, S) # x.shape: [B, N, C]    
                else:
                    x = auto_grad_checkpoint(block, x, y, t_spc_mlp, t_tmp_mlp, y_lens, x_mask, t0_spc_mlp, t0_tmp_mlp, T, S) # x.shape: [B, N, C]    
        '''           
                    
                        
        
        if self.enable_sequence_parallelism:
            x = rearrange(x, "B (T S) C -> B T S C", T=T, S=S)
            x = gather_forward_split_backward(x, get_sequence_parallel_group(), dim=2, grad_scale="up")
            S = S * dist.get_world_size(get_sequence_parallel_group())
            x = rearrange(x, "B T S C -> B (T S) C", T=T, S=S)
            
        # final process
        x = self.final_layer(x, t, x_mask, t0_spc, T, S)  # [B, N, C=T_p * H_p * W_p * C_out]
        x = self.unpatchify(x, T, H, W, Tx, Hx, Wx)  # [B, C_out, T, H, W]

        # cast to float32 for better accuracy
        x = x.to(torch.float32)
        return x

    def unpatchify(self, x, N_t, N_h, N_w, R_t, R_h, R_w):
        """
        Args:
            x (torch.Tensor): of shape [B, N, C]

        Return:
            x (torch.Tensor): of shape [B, C_out, T, H, W]
        """

        # N_t, N_h, N_w = [self.input_size[i] // self.patch_size[i] for i in range(3)]
        T_p, H_p, W_p = self.patch_size
        x = rearrange(
            x,
            "B (N_t N_h N_w) (T_p H_p W_p C_out) -> B C_out (N_t T_p) (N_h H_p) (N_w W_p)",
            N_t=N_t,
            N_h=N_h,
            N_w=N_w,
            T_p=T_p,
            H_p=H_p,
            W_p=W_p,
            C_out=self.out_channels,
        )
        # unpad
        x = x[:, :, :R_t, :R_h, :R_w]
        return x

    def unpatchify_old(self, x):
        c = self.out_channels
        t, h, w = [self.input_size[i] // self.patch_size[i] for i in range(3)]
        pt, ph, pw = self.patch_size

        x = x.reshape(shape=(x.shape[0], t, h, w, pt, ph, pw, c))
        x = rearrange(x, "n t h w r p q c -> n c t r h p w q")
        imgs = x.reshape(shape=(x.shape[0], c, t * pt, h * ph, w * pw))
        return imgs

    def get_spatial_pos_embed(self, H, W, scale=1.0, base_size=None):
        pos_embed = get_2d_sincos_pos_embed(
            self.hidden_size,
            (H, W),
            scale=scale,
            base_size=base_size,
        )
        pos_embed = torch.from_numpy(pos_embed).float().unsqueeze(0).requires_grad_(False)
        return pos_embed

    def freeze_block(self):
        for idx, block in enumerate(self.blocks):
            for param in block.parameters():
                param.requires_grad = False
        
    def freeze_not_temporal(self):
        for n, p in self.named_parameters():
            if "attn_temp" not in n:
                p.requires_grad = False

    def freeze_text(self):
        for n, p in self.named_parameters():
            if "cross_attn" in n:
                p.requires_grad = False
    
    def finetune_pos_emb(self):
        for param in self.parameters():
            param.requires_grad = False
        for param in self.x_embedder.parameters():
            param.requires_grad = True
        for idx, block in enumerate(self.blocks):
            if idx == 0:
                for param in block.parameters():
                    param.requires_grad = True
                        

    def initialize_temporal(self):
        for block in self.blocks:
            nn.init.constant_(block.attn_temp.proj.weight, 0)
            nn.init.constant_(block.attn_temp.proj.bias, 0)

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)
        
        # Initialize zero_linear_layers
        for zero_linear in self.zero_linear_layers:
            nn.init.constant_(zero_linear.weight, 0.0)
            nn.init.constant_(zero_linear.bias, 0.0)
        
        # Initialize self.zero_linear_layers_traj
        nn.init.constant_(self.zero_linear_layers_traj.weight, 0.0)
        nn.init.constant_(self.zero_linear_layers_traj.bias, 0.0)
        
        for zero_conv in self.zero_conv:
            zero_conv.initialize_conv()
        

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        nn.init.normal_(self.t_block[1].weight, std=0.02)
        nn.init.normal_(self.t_block_temp[1].weight, std=0.02)

        # Initialize caption embedding MLP:
        nn.init.normal_(self.y_embedder.y_proj.fc1.weight, std=0.02)
        nn.init.normal_(self.y_embedder.y_proj.fc2.weight, std=0.02)

        # Zero-out adaLN modulation layers in PixArt blocks:
        for block in self.blocks:
            nn.init.constant_(block.cross_attn.proj.weight, 0)
            nn.init.constant_(block.cross_attn.proj.bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def initialize_duplicated_blocks(self):
        for i in range(len(self.duplicated_blocks)):
            self.duplicated_blocks[i].load_state_dict(self.blocks[i].state_dict())

        




from opensora.models.stdit.condition_modules import BiMoT,ConditionImageAdapter,AdapterFusion,CameraEmbProj,TimeCompressionCNN
@MODELS.register_module()
class STDiT2_BiMoT(PreTrainedModel):

    config_class = STDiT2Config

    def __init__(
        self,
        config
    ):
        super().__init__(config)
        self.pred_sigma = config.pred_sigma
        self.in_channels = config.in_channels
        self.out_channels = config.in_channels * 2 if config.pred_sigma else config.in_channels
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.no_temporal_pos_emb = config.no_temporal_pos_emb
        self.depth = config.depth
        self.mlp_ratio = config.mlp_ratio
        self.enable_flash_attn = config.enable_flash_attn
        self.enable_layernorm_kernel = config.enable_layernorm_kernel
        
        # computation related
        self.enable_sequence_parallelism = config.enable_sequence_parallelism
        
        # condition control
        self.enable_condition = config.enable_condition
        self.condition_downsample = config.condition_downsample
        self.scale_factor = config.scale_factor
        
        # pos embedding scale 
        self.use_separate_scales = config.use_separate_scales

        # support dynamic input
        self.patch_size = config.patch_size
        self.input_size = config.input_size
        self.input_sq_size = config.input_sq_size
        self.pos_embed = PositionEmbedding2D(config.hidden_size)

        self.x_embedder = PatchEmbed3D(config.patch_size, config.in_channels, config.hidden_size)
        self.t_embedder = TimestepEmbedder(config.hidden_size)
        self.t_block = nn.Sequential(nn.SiLU(), nn.Linear(config.hidden_size, 6 * config.hidden_size, bias=True))
        self.t_block_temp = nn.Sequential(nn.SiLU(), nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=True))  # new
        self.y_embedder = CaptionEmbedder(
            in_channels=config.caption_channels,
            hidden_size=config.hidden_size,
            uncond_prob=config.class_dropout_prob,
            act_layer=approx_gelu,
            token_num=config.model_max_length,
        )

        drop_path = [x.item() for x in torch.linspace(0, config.drop_path, config.depth)]
        self.rope = CustomRotaryEmbedding(dim=self.hidden_size // self.num_heads)  # new
        self.blocks = nn.ModuleList(
            [
                STDiT2Block(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=self.mlp_ratio,
                    drop_path=drop_path[i],
                    enable_flash_attn=self.enable_flash_attn,
                    enable_layernorm_kernel=self.enable_layernorm_kernel,
                    rope=self.rope.rotate_queries_or_keys,
                    qk_norm=config.qk_norm,
                    qk_norm_legacy=config.qk_norm_legacy,
                    enable_sequence_parallelism=config.enable_sequence_parallelism,
                )
                for i in range(self.depth)
            ]
        )

        # === BiMoT ===
        # TODO: hyper-params
        self.camera_emb_in = CameraEmbProj(
            camera_num=1, # consider as single-view
            inner_channels=128, 
            out_channels=self.hidden_size,
        )
        self.map_in = ConditionImageAdapter(
            in_channels=3,
            out_channels=[int(i / 2) for i in  [4, 4,4,4]], 
            is_down_blocks=[True, True, True, False],
            num_res_blocks=2,
            inner_channel_factor=64 / 2,
            camera_emb_dim=self.hidden_size, # TODO: 
            downscale_factor=8,
            is_convolutional_down=False,
            is_inverse_color=True
        )
        self.box_in = ConditionImageAdapter(
            in_channels=3,
            out_channels=[int(i / 2) for i in  [4, 4,4,4]],
            is_down_blocks=[True, True, True, False],
            num_res_blocks=2,
            inner_channel_factor=64 / 2,
            camera_emb_dim=self.hidden_size, # TODO: 
            downscale_factor=8,
            is_convolutional_down=False,
            is_inverse_color=True
        )
        self.map_box_mix = AdapterFusion(
            out_channels=[4,4, 4, 4],
            split_num=2
        )
        
        self.map_box_attn_blocks = nn.ModuleList([])
        transformer_layers_per_block = 1
        if isinstance(transformer_layers_per_block, int):
            transformer_layers_per_block = [transformer_layers_per_block] * 4
        # Define the encoding cross attention layer
        for i in range(len((4, 4, 4, 4))):
            self.map_box_attn_blocks.append(
                BiMoT(
                    num_attention_heads=(8, 16, 32, 32)[max(i - 1, 0)],
                    attention_head_dim=(1152, 1152, 1152, 1152)[max(i - 1, 0)] //  (8, 16, 32, 32)[max(i - 1, 0)],
                    temporal_attention_head_dim=32,
                    in_channels=(1152,1152,1152,1152)[max(i - 1, 0)],
                    num_layers=transformer_layers_per_block[max(i - 1, 0)],
                )
            )        
        # Define the decoding cross attention layer
        for i in range(len((4, 4, 4, 4))-1):
            self.map_box_attn_blocks.append(
                BiMoT(
                    num_attention_heads= (8, 16, 32, 32)[min(3, 4 - i)],
                    attention_head_dim=(1152, 1152, 1152, 1152)[min(3, 4 - i)] //   (8, 16, 32, 32)[min(3, 4 - i)],
                    temporal_attention_head_dim=4,
                    in_channels=(1152,1152,1152,1152)[min(3, 4 - i)],
                    num_layers=transformer_layers_per_block[min(3, 4 - i)],
                )
            )     
        
        # === BiMoT ===
        
        
        
        self.final_layer = T2IFinalLayer(config.hidden_size, np.prod(self.patch_size), self.out_channels)

        # multi_res
        assert self.hidden_size % 3 == 0, "hidden_size must be divisible by 3"
        self.csize_embedder = SizeEmbedder(self.hidden_size // 3)
        self.ar_embedder = SizeEmbedder(self.hidden_size // 3)
        self.fl_embedder = SizeEmbedder(self.hidden_size)  # new
        self.fps_embedder = SizeEmbedder(self.hidden_size)  # new

        # init model
        self.initialize_weights()
        self.initialize_temporal()
        
        # freeze
        if config.freeze is not None:
            assert config.freeze in ["not_temporal", "text", "pos_emb"]
            if config.freeze == "not_temporal":
                self.freeze_not_temporal()
            elif config.freeze == "text":
                self.freeze_text()
            elif config.freeze == "pos_emb":
                self.finetune_pos_emb()

    def get_dynamic_size(self, x):
        _, _, T, H, W = x.size()
        if T % self.patch_size[0] != 0:
            T += self.patch_size[0] - T % self.patch_size[0]
        if H % self.patch_size[1] != 0:
            H += self.patch_size[1] - H % self.patch_size[1]
        if W % self.patch_size[2] != 0:
            W += self.patch_size[2] - W % self.patch_size[2]
        T = T // self.patch_size[0]
        H = H // self.patch_size[1]
        W = W // self.patch_size[2]
        return (T, H, W)

    def forward(
        self, x, timestep, y, mask=None, x_mask=None, num_frames=None,
            height=None, width=None, ar=None, fps=None, 
            full_height=None, full_width=None,   
            layout_conditions: torch.Tensor = None, 
            hdmap: torch.Tensor = None, 
            bbox: torch.Tensor = None, 
            **kwargs):
        """
        Forward pass of STDiT.
        Args:
            x (torch.Tensor): latent representation of video; of shape [B, C, T, H, W]
            timestep (torch.Tensor): diffusion time steps; of shape [B]
            y (torch.Tensor): representation of prompts; of shape [B, 1, N_token, C]
            mask (torch.Tensor): mask for selecting prompt tokens; of shape [B, N_token]

        Returns:
            x (torch.Tensor): output latent representation; of shape [B, C, T, H, W]
        """
        B = x.shape[0]
        dtype = self.x_embedder.proj.weight.dtype
        x = x.to(dtype)
        timestep = timestep.to(dtype)
        y = y.to(dtype)
        
   
        
        
   

        # === process data info ===
        # 1. get dynamic size
        hw = torch.cat([height[:, None], width[:, None]], dim=1)
        # hw = torch.cat([full_height[:, None], full_width[:, None]], dim=1)
        csize = self.csize_embedder(hw, B)

        # 2. get aspect ratio
        ar = ar.unsqueeze(1)
        ar = self.ar_embedder(ar, B)
        data_info = torch.cat([csize, ar], dim=1)

        # 3. get number of frames
        fl = num_frames.unsqueeze(1)
        fps = fps.unsqueeze(1)
        fl = self.fl_embedder(fl, B)
        fl = fl + self.fps_embedder(fps, B)

        # === get dynamic shape size ===
        _, _, Tx, Hx, Wx = x.size()
        T, H, W = self.get_dynamic_size(x)
        
        # adjust for sequence parallelism
        # we need to ensure H * W is divisible by sequence parallel size
        # for simplicity, we can adjust the height to make it divisible
        if self.enable_sequence_parallelism:
            sp_size = dist.get_world_size(get_sequence_parallel_group())
            if H % sp_size != 0:
                h_pad_size = sp_size - H % sp_size
            else:
                h_pad_size = 0

            if h_pad_size > 0:
                hx_pad_size = h_pad_size * self.patch_size[1]

                # pad x along the H dimension
                H += h_pad_size
                x = F.pad(x, (0, 0, 0, hx_pad_size))
                
        S = H * W
   
        # 2-dim scale pos_embedding  
        if self.use_separate_scales:
            base_size = (H,W) 
            scale = (full_height[0].item() / self.input_sq_size, full_width[0].item() / self.input_sq_size) # self.input_sq_size: 512
        else:
            base_size = round(S**0.5)
            resolution_sq = (full_height[0].item() * full_width[0].item()) ** 0.5
            scale = resolution_sq / self.input_sq_size 
        
        pos_emb = self.pos_embed(x, H, W, scale=scale, base_size=base_size, use_separate_scales=self.use_separate_scales) # torch.Size([1, H*W, hidden_feature])

        # === encode map && 3dbox condition ===
        if self.enable_condition:        
            #condition pre-processing
            num_frames = x.shape[2] # self.num_frames # T = x.shape[2]
            camera_emb = 0*self.camera_emb_in(0, B).unsqueeze(1) # [B, 1, C] # TODO:  consider as single-view 
            camera_emb = camera_emb.repeat(1,num_frames,1) # B, num_frames, C
            
            camera_emb = camera_emb.flatten(0, 1) # [(BT'),C]
            hdmap = hdmap.contiguous().flatten(0, 1) # BT'CHW -> (BT')CHW 
            bbox = bbox.contiguous().flatten(0, 1) 
            # save memory
            if self.training:
                map_condition_list = torch.utils.checkpoint.checkpoint(
                    self.map_in,
                    use_reentrant=False,
                    x=hdmap,
                    cam_emb=camera_emb
                )
                box_condition_list = torch.utils.checkpoint.checkpoint(
                    self.box_in,
                    use_reentrant=False,
                    x=bbox,
                    cam_emb=camera_emb
                )
               
            else:
                map_condition_list = self.map_in(hdmap, camera_emb)
                box_condition_list = self.box_in(bbox, camera_emb)
            
            layout_condition_list = self.map_box_mix([map_condition_list, box_condition_list])\
                if map_condition_list is not None and box_condition_list is not None else None
        # === encode map && 3dbox condition ===











        # embedding
        x = self.x_embedder(x)  # [B, N, C]
        x = rearrange(x, "B (T S) C -> B T S C", T=T, S=S)
        x = x + pos_emb
        
        # shard over the sequence dim if sp is enabled
        if self.enable_sequence_parallelism:
            x = split_forward_gather_backward(x, get_sequence_parallel_group(), dim=2, grad_scale="down")
            S = S // dist.get_world_size(get_sequence_parallel_group())
            
        x = rearrange(x, "B T S C -> B (T S) C")

        # prepare adaIN
        t = self.t_embedder(timestep, dtype=x.dtype)  # [B, C]
        t_spc = t + data_info  # [B, C]
        t_tmp = t + fl  # [B, C]
        t_spc_mlp = self.t_block(t_spc)  # [B, 6*C]
        t_tmp_mlp = self.t_block_temp(t_tmp)  # [B, 3*C]
        if x_mask is not None:
            t0_timestep = torch.zeros_like(timestep)
            t0 = self.t_embedder(t0_timestep, dtype=x.dtype)
            t0_spc = t0 + data_info
            t0_tmp = t0 + fl
            t0_spc_mlp = self.t_block(t0_spc)
            t0_tmp_mlp = self.t_block_temp(t0_tmp)
        else:
            t0_spc = None
            t0_tmp = None
            t0_spc_mlp = None
            t0_tmp_mlp = None

        # prepare y
        y = self.y_embedder(y, self.training)  # [B, 1, N_token, C]

        if mask is not None:
            if mask.shape[0] != y.shape[0]:
                mask = mask.repeat(y.shape[0] // mask.shape[0], 1)
            mask = mask.squeeze(1).squeeze(1)
            y = y.squeeze(1).masked_select(mask.unsqueeze(-1) != 0).view(1, -1, x.shape[-1])
            y_lens = mask.sum(dim=1).tolist()
        else:
            y_lens = [y.shape[2]] * y.shape[0]
            y = y.squeeze(1).view(1, -1, x.shape[-1])

        # blocks
        for idx, block in enumerate(self.blocks):
            # torch.cuda.empty_cache()
            if self.enable_condition and idx % 7 == 0:
                x = rearrange(x,"B (T H W) C -> (B T) C H W",H=H,W=W,B=B,T=T) # [BV, T'*S, C]->[BV*T', F, H, W] 
                # layout_condition in to patch
                lay_out_condition = layout_condition_list[idx%7].view(B, num_frames, -1, H*2, W*2)# [B*T, C', H*2, W*2]->[B, T, C', H*2, W*2],T=16
                lay_out_condition = lay_out_condition.permute(0, 2, 1, 3, 4) # [B, T', C', H, W] -> [B, C', T', H, W]
                lay_out_condition = auto_grad_checkpoint(self.x_embedder, lay_out_condition) # [B, C', T', H, W] -> [B, (T'S), F]
                #rearrange
                lay_out_condition = rearrange(lay_out_condition, "B (T H W) C -> (B T) C H W",H=H,W=W,B=B,T=T)
                image_only_indicator = torch.zeros(B, T, dtype=x.dtype, device=x.device)
                # save memory
                if self.training:
                    x = torch.utils.checkpoint.checkpoint(
                        self.map_box_attn_blocks[idx%7],
                        use_reentrant=False,
                        hidden_states=x,
                        additional_conditions=lay_out_condition,
                        image_only_indicator=image_only_indicator
                    )
                else:
                    x = self.map_box_attn_blocks[idx%7](x,lay_out_condition,image_only_indicator)# [B*T', F, H, W], [B*T', F, H, W], [B, T']
                x = rearrange(x, "(B T) C H W -> B (T H W) C",H=H,W=W,B=B,T=T)
                
            x = auto_grad_checkpoint(block, x, y, t_spc_mlp, t_tmp_mlp, y_lens, x_mask, t0_spc_mlp, t0_tmp_mlp, T, S) # x.shape: [B, N, C]
            
            


        if self.enable_sequence_parallelism:
            x = rearrange(x, "B (T S) C -> B T S C", T=T, S=S)
            x = gather_forward_split_backward(x, get_sequence_parallel_group(), dim=2, grad_scale="up")
            S = S * dist.get_world_size(get_sequence_parallel_group())
            x = rearrange(x, "B T S C -> B (T S) C", T=T, S=S)
            
        # final process
        x = self.final_layer(x, t, x_mask, t0_spc, T, S)  # [B, N, C=T_p * H_p * W_p * C_out]
        x = self.unpatchify(x, T, H, W, Tx, Hx, Wx)  # [B, C_out, T, H, W]

        # cast to float32 for better accuracy
        x = x.to(torch.float32)
        return x

    def unpatchify(self, x, N_t, N_h, N_w, R_t, R_h, R_w):
        """
        Args:
            x (torch.Tensor): of shape [B, N, C]

        Return:
            x (torch.Tensor): of shape [B, C_out, T, H, W]
        """

        # N_t, N_h, N_w = [self.input_size[i] // self.patch_size[i] for i in range(3)]
        T_p, H_p, W_p = self.patch_size
        x = rearrange(
            x,
            "B (N_t N_h N_w) (T_p H_p W_p C_out) -> B C_out (N_t T_p) (N_h H_p) (N_w W_p)",
            N_t=N_t,
            N_h=N_h,
            N_w=N_w,
            T_p=T_p,
            H_p=H_p,
            W_p=W_p,
            C_out=self.out_channels,
        )
        # unpad
        x = x[:, :, :R_t, :R_h, :R_w]
        return x

    def unpatchify_old(self, x):
        c = self.out_channels
        t, h, w = [self.input_size[i] // self.patch_size[i] for i in range(3)]
        pt, ph, pw = self.patch_size

        x = x.reshape(shape=(x.shape[0], t, h, w, pt, ph, pw, c))
        x = rearrange(x, "n t h w r p q c -> n c t r h p w q")
        imgs = x.reshape(shape=(x.shape[0], c, t * pt, h * ph, w * pw))
        return imgs

    def get_spatial_pos_embed(self, H, W, scale=1.0, base_size=None):
        pos_embed = get_2d_sincos_pos_embed(
            self.hidden_size,
            (H, W),
            scale=scale,
            base_size=base_size,
        )
        pos_embed = torch.from_numpy(pos_embed).float().unsqueeze(0).requires_grad_(False)
        return pos_embed

    def freeze_not_temporal(self):
        for n, p in self.named_parameters():
            if "attn_temp" not in n:
                p.requires_grad = False

    def freeze_text(self):
        for n, p in self.named_parameters():
            if "cross_attn" in n:
                p.requires_grad = False

    def finetune_pos_emb(self):
        for param in self.parameters():
            param.requires_grad = False
        for param in self.x_embedder.parameters():
            param.requires_grad = True
        for idx, block in enumerate(self.blocks):
            if idx == 0:
                for param in block.parameters():
                    param.requires_grad = True
                    
    def initialize_temporal(self):
        for block in self.blocks:
            nn.init.constant_(block.attn_temp.proj.weight, 0)
            nn.init.constant_(block.attn_temp.proj.bias, 0)

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)
        
        # Initialize fps_embedder
        nn.init.normal_(self.fps_embedder.mlp[0].weight, std=0.02)
        nn.init.constant_(self.fps_embedder.mlp[0].bias, 0)
        nn.init.constant_(self.fps_embedder.mlp[2].weight, 0)
        nn.init.constant_(self.fps_embedder.mlp[2].bias, 0)
        

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        nn.init.normal_(self.t_block[1].weight, std=0.02)
        nn.init.normal_(self.t_block_temp[1].weight, std=0.02)

        # Initialize caption embedding MLP:
        nn.init.normal_(self.y_embedder.y_proj.fc1.weight, std=0.02)
        nn.init.normal_(self.y_embedder.y_proj.fc2.weight, std=0.02)

        # Zero-out adaLN modulation layers in PixArt blocks:
        for block in self.blocks:
            nn.init.constant_(block.cross_attn.proj.weight, 0)
            nn.init.constant_(block.cross_attn.proj.bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)
        
        # Initialize map & box blocks
        for block in self.map_box_attn_blocks:
            nn.init.constant_(block.proj_out_1_a.weight, 0)
            nn.init.constant_(block.proj_out_1_b.weight, 0)
            nn.init.constant_(block.proj_out_2_a.weight, 0)
            nn.init.constant_(block.proj_out_2_b.weight, 0)





@MODELS.register_module()
class STDiT2_BiMoT_VAE(PreTrainedModel):

    config_class = STDiT2Config

    def __init__(
        self,
        config
    ):
        super().__init__(config)
        self.pred_sigma = config.pred_sigma
        self.in_channels = config.in_channels
        self.out_channels = config.in_channels * 2 if config.pred_sigma else config.in_channels
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.no_temporal_pos_emb = config.no_temporal_pos_emb
        self.depth = config.depth
        self.mlp_ratio = config.mlp_ratio
        self.enable_flash_attn = config.enable_flash_attn
        self.enable_layernorm_kernel = config.enable_layernorm_kernel
        
        # computation related
        self.enable_sequence_parallelism = config.enable_sequence_parallelism
        
        # condition control
        self.enable_condition = config.enable_condition
        self.condition_downsample = config.condition_downsample
        self.scale_factor = config.scale_factor
        
        # pos embedding scale 
        self.use_separate_scales = config.use_separate_scales

        # support dynamic input
        self.patch_size = config.patch_size
        self.input_size = config.input_size
        self.input_sq_size = config.input_sq_size
        self.pos_embed = PositionEmbedding2D(config.hidden_size)

        self.x_embedder = PatchEmbed3D(config.patch_size, config.in_channels, config.hidden_size)
        self.t_embedder = TimestepEmbedder(config.hidden_size)
        self.t_block = nn.Sequential(nn.SiLU(), nn.Linear(config.hidden_size, 6 * config.hidden_size, bias=True))
        self.t_block_temp = nn.Sequential(nn.SiLU(), nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=True))  # new
        self.y_embedder = CaptionEmbedder(
            in_channels=config.caption_channels,
            hidden_size=config.hidden_size,
            uncond_prob=config.class_dropout_prob,
            act_layer=approx_gelu,
            token_num=config.model_max_length,
        )

        drop_path = [x.item() for x in torch.linspace(0, config.drop_path, config.depth)]
        self.rope = CustomRotaryEmbedding(dim=self.hidden_size // self.num_heads)  # new
        self.blocks = nn.ModuleList(
            [
                STDiT2Block(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=self.mlp_ratio,
                    drop_path=drop_path[i],
                    enable_flash_attn=self.enable_flash_attn,
                    enable_layernorm_kernel=self.enable_layernorm_kernel,
                    rope=self.rope.rotate_queries_or_keys,
                    qk_norm=config.qk_norm,
                    qk_norm_legacy=config.qk_norm_legacy,
                    enable_sequence_parallelism=config.enable_sequence_parallelism,
                )
                for i in range(self.depth)
            ]
        )

        # === BiMoT ===
        # TODO: hyper-params

        self.map_box_attn_blocks = nn.ModuleList([])
        transformer_layers_per_block = 1
        if isinstance(transformer_layers_per_block, int):
            transformer_layers_per_block = [transformer_layers_per_block] * 4
        # Define the encoding cross attention layer
        for i in range(len((4, 4, 4, 4))):
            self.map_box_attn_blocks.append(
                BiMoT(
                    num_attention_heads=(8, 16, 32, 32)[max(i - 1, 0)],
                    attention_head_dim=(1152, 1152, 1152, 1152)[max(i - 1, 0)] //  (8, 16, 32, 32)[max(i - 1, 0)],
                    temporal_attention_head_dim=32,
                    in_channels=(1152,1152,1152,1152)[max(i - 1, 0)],
                    num_layers=transformer_layers_per_block[max(i - 1, 0)],
                )
            )        
        # Define the decoding cross attention layer
        for i in range(len((4, 4, 4, 4))-1):
            self.map_box_attn_blocks.append(
                BiMoT(
                    num_attention_heads= (8, 16, 32, 32)[min(3, 4 - i)],
                    attention_head_dim=(1152, 1152, 1152, 1152)[min(3, 4 - i)] //   (8, 16, 32, 32)[min(3, 4 - i)],
                    temporal_attention_head_dim=4,
                    in_channels=(1152,1152,1152,1152)[min(3, 4 - i)],
                    num_layers=transformer_layers_per_block[min(3, 4 - i)],
                )
            )     
        
        # === BiMoT ===
        
        
        
        self.final_layer = T2IFinalLayer(config.hidden_size, np.prod(self.patch_size), self.out_channels)

        # multi_res
        assert self.hidden_size % 3 == 0, "hidden_size must be divisible by 3"
        self.csize_embedder = SizeEmbedder(self.hidden_size // 3)
        self.ar_embedder = SizeEmbedder(self.hidden_size // 3)
        self.fl_embedder = SizeEmbedder(self.hidden_size)  # new
        self.fps_embedder = SizeEmbedder(self.hidden_size)  # new

        # init model
        self.initialize_weights()
        self.initialize_temporal()
        
        # freeze
        if config.freeze is not None:
            assert config.freeze in ["not_temporal", "text", "pos_emb"]
            if config.freeze == "not_temporal":
                self.freeze_not_temporal()
            elif config.freeze == "text":
                self.freeze_text()
            elif config.freeze == "pos_emb":
                self.finetune_pos_emb()

    def get_dynamic_size(self, x):
        _, _, T, H, W = x.size()
        if T % self.patch_size[0] != 0:
            T += self.patch_size[0] - T % self.patch_size[0]
        if H % self.patch_size[1] != 0:
            H += self.patch_size[1] - H % self.patch_size[1]
        if W % self.patch_size[2] != 0:
            W += self.patch_size[2] - W % self.patch_size[2]
        T = T // self.patch_size[0]
        H = H // self.patch_size[1]
        W = W // self.patch_size[2]
        return (T, H, W)

    def forward(
        self, x, timestep, y, mask=None, x_mask=None, num_frames=None,
            height=None, width=None, ar=None, fps=None, 
            full_height=None, full_width=None,   
            layout_conditions: torch.Tensor = None, 
            hdmap: torch.Tensor = None, 
            bbox: torch.Tensor = None, 
            **kwargs):
        """
        Forward pass of STDiT.
        Args:
            x (torch.Tensor): latent representation of video; of shape [B, C, T, H, W]
            timestep (torch.Tensor): diffusion time steps; of shape [B]
            y (torch.Tensor): representation of prompts; of shape [B, 1, N_token, C]
            mask (torch.Tensor): mask for selecting prompt tokens; of shape [B, N_token]

        Returns:
            x (torch.Tensor): output latent representation; of shape [B, C, T, H, W]
        """
        B = x.shape[0]
        dtype = self.x_embedder.proj.weight.dtype
        x = x.to(dtype)
        timestep = timestep.to(dtype)
        y = y.to(dtype)
        

        # === process data info ===
        # 1. get dynamic size
        hw = torch.cat([height[:, None], width[:, None]], dim=1)
        # hw = torch.cat([full_height[:, None], full_width[:, None]], dim=1)
        csize = self.csize_embedder(hw, B)

        # 2. get aspect ratio
        ar = ar.unsqueeze(1)
        ar = self.ar_embedder(ar, B)
        data_info = torch.cat([csize, ar], dim=1)

        # 3. get number of frames
        fl = num_frames.unsqueeze(1)
        fps = fps.unsqueeze(1)
        fl = self.fl_embedder(fl, B)
        fl = fl + self.fps_embedder(fps, B)

        # === get dynamic shape size ===
        _, _, Tx, Hx, Wx = x.size()
        T, H, W = self.get_dynamic_size(x)
        
        # adjust for sequence parallelism
        # we need to ensure H * W is divisible by sequence parallel size
        # for simplicity, we can adjust the height to make it divisible
        if self.enable_sequence_parallelism:
            sp_size = dist.get_world_size(get_sequence_parallel_group())
            if H % sp_size != 0:
                h_pad_size = sp_size - H % sp_size
            else:
                h_pad_size = 0

            if h_pad_size > 0:
                hx_pad_size = h_pad_size * self.patch_size[1]

                # pad x along the H dimension
                H += h_pad_size
                x = F.pad(x, (0, 0, 0, hx_pad_size))
                
        S = H * W
   
        # 2-dim scale pos_embedding
        if self.use_separate_scales:
            base_size = (H,W) 
            scale = (full_height[0].item() / self.input_sq_size, full_width[0].item() / self.input_sq_size) # self.input_sq_size: 512
        else:
            base_size = round(S**0.5)
            resolution_sq = (full_height[0].item() * full_width[0].item()) ** 0.5
            scale = resolution_sq / self.input_sq_size 
        
        pos_emb = self.pos_embed(x, H, W, scale=scale, base_size=base_size, use_separate_scales=self.use_separate_scales) # torch.Size([1, H*W, hidden_feature])

        

        # embedding
        x = self.x_embedder(x)  # [B, N, C]
        x = rearrange(x, "B (T S) C -> B T S C", T=T, S=S)
        x = x + pos_emb
        
        # shard over the sequence dim if sp is enabled
        if self.enable_sequence_parallelism:
            x = split_forward_gather_backward(x, get_sequence_parallel_group(), dim=2, grad_scale="down")
            S = S // dist.get_world_size(get_sequence_parallel_group())
            
        x = rearrange(x, "B T S C -> B (T S) C")

        # prepare adaIN
        t = self.t_embedder(timestep, dtype=x.dtype)  # [B, C]
        t_spc = t + data_info  # [B, C]
        t_tmp = t + fl  # [B, C]
        t_spc_mlp = self.t_block(t_spc)  # [B, 6*C]
        t_tmp_mlp = self.t_block_temp(t_tmp)  # [B, 3*C]
        if x_mask is not None:
            t0_timestep = torch.zeros_like(timestep)
            t0 = self.t_embedder(t0_timestep, dtype=x.dtype)
            t0_spc = t0 + data_info
            t0_tmp = t0 + fl
            t0_spc_mlp = self.t_block(t0_spc)
            t0_tmp_mlp = self.t_block_temp(t0_tmp)
        else:
            t0_spc = None
            t0_tmp = None
            t0_spc_mlp = None
            t0_tmp_mlp = None

        # prepare y
        y = self.y_embedder(y, self.training)  # [B, 1, N_token, C]

        if mask is not None:
            if mask.shape[0] != y.shape[0]:
                mask = mask.repeat(y.shape[0] // mask.shape[0], 1)
            mask = mask.squeeze(1).squeeze(1)
            y = y.squeeze(1).masked_select(mask.unsqueeze(-1) != 0).view(1, -1, x.shape[-1])
            y_lens = mask.sum(dim=1).tolist()
        else:
            y_lens = [y.shape[2]] * y.shape[0]
            y = y.squeeze(1).view(1, -1, x.shape[-1])

        # blocks
        for idx, block in enumerate(self.blocks):
            # torch.cuda.empty_cache()
            if self.enable_condition and idx % 7 == 0:
                x = rearrange(x,"B (T H W) C -> (B T) C H W",H=H,W=W,B=B,T=T) # [BV, T'*S, C]->[BV*T', F, H, W] 
                # layout_condition in to patch
                lay_out_condition = layout_conditions
                lay_out_condition = auto_grad_checkpoint(self.x_embedder, lay_out_condition) # [B, C', T', H, W] -> [B, (T'S), F]
                #rearrange
                lay_out_condition = rearrange(lay_out_condition, "B (T H W) C -> (B T) C H W",H=H,W=W,B=B,T=T)
                image_only_indicator = torch.zeros(B, T, dtype=x.dtype, device=x.device)
                # save memory
                if self.training:
                    x = torch.utils.checkpoint.checkpoint(
                        self.map_box_attn_blocks[idx%7],
                        use_reentrant=False,
                        hidden_states=x,
                        additional_conditions=lay_out_condition,
                        image_only_indicator=image_only_indicator
                    )
                else:
                    x = self.map_box_attn_blocks[idx%7](x,lay_out_condition,image_only_indicator)# [B*T', F, H, W], [B*T', F, H, W], [B, T']
                x = rearrange(x, "(B T) C H W -> B (T H W) C",H=H,W=W,B=B,T=T)
                
            x = auto_grad_checkpoint(block, x, y, t_spc_mlp, t_tmp_mlp, y_lens, x_mask, t0_spc_mlp, t0_tmp_mlp, T, S) # x.shape: [B, N, C]
            
            


        if self.enable_sequence_parallelism:
            x = rearrange(x, "B (T S) C -> B T S C", T=T, S=S)
            x = gather_forward_split_backward(x, get_sequence_parallel_group(), dim=2, grad_scale="up")
            S = S * dist.get_world_size(get_sequence_parallel_group())
            x = rearrange(x, "B T S C -> B (T S) C", T=T, S=S)
            
        # final process
        x = self.final_layer(x, t, x_mask, t0_spc, T, S)  # [B, N, C=T_p * H_p * W_p * C_out]
        x = self.unpatchify(x, T, H, W, Tx, Hx, Wx)  # [B, C_out, T, H, W]

        # cast to float32 for better accuracy
        x = x.to(torch.float32)
        return x

    def unpatchify(self, x, N_t, N_h, N_w, R_t, R_h, R_w):
        """
        Args:
            x (torch.Tensor): of shape [B, N, C]

        Return:
            x (torch.Tensor): of shape [B, C_out, T, H, W]
        """

        # N_t, N_h, N_w = [self.input_size[i] // self.patch_size[i] for i in range(3)]
        T_p, H_p, W_p = self.patch_size
        x = rearrange(
            x,
            "B (N_t N_h N_w) (T_p H_p W_p C_out) -> B C_out (N_t T_p) (N_h H_p) (N_w W_p)",
            N_t=N_t,
            N_h=N_h,
            N_w=N_w,
            T_p=T_p,
            H_p=H_p,
            W_p=W_p,
            C_out=self.out_channels,
        )
        # unpad
        x = x[:, :, :R_t, :R_h, :R_w]
        return x

    def unpatchify_old(self, x):
        c = self.out_channels
        t, h, w = [self.input_size[i] // self.patch_size[i] for i in range(3)]
        pt, ph, pw = self.patch_size

        x = x.reshape(shape=(x.shape[0], t, h, w, pt, ph, pw, c))
        x = rearrange(x, "n t h w r p q c -> n c t r h p w q")
        imgs = x.reshape(shape=(x.shape[0], c, t * pt, h * ph, w * pw))
        return imgs

    def get_spatial_pos_embed(self, H, W, scale=1.0, base_size=None):
        pos_embed = get_2d_sincos_pos_embed(
            self.hidden_size,
            (H, W),
            scale=scale,
            base_size=base_size,
        )
        pos_embed = torch.from_numpy(pos_embed).float().unsqueeze(0).requires_grad_(False)
        return pos_embed

    def freeze_not_temporal(self):
        for n, p in self.named_parameters():
            if "attn_temp" not in n:
                p.requires_grad = False

    def freeze_text(self):
        for n, p in self.named_parameters():
            if "cross_attn" in n:
                p.requires_grad = False

    def finetune_pos_emb(self):
        for param in self.parameters():
            param.requires_grad = False
        for param in self.x_embedder.parameters():
            param.requires_grad = True
        for idx, block in enumerate(self.blocks):
            if idx == 0:
                for param in block.parameters():
                    param.requires_grad = True
                    
    def initialize_temporal(self):
        for block in self.blocks:
            nn.init.constant_(block.attn_temp.proj.weight, 0)
            nn.init.constant_(block.attn_temp.proj.bias, 0)

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)
        
        # Initialize fps_embedder
        nn.init.normal_(self.fps_embedder.mlp[0].weight, std=0.02)
        nn.init.constant_(self.fps_embedder.mlp[0].bias, 0)
        nn.init.constant_(self.fps_embedder.mlp[2].weight, 0)
        nn.init.constant_(self.fps_embedder.mlp[2].bias, 0)
        

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        nn.init.normal_(self.t_block[1].weight, std=0.02)
        nn.init.normal_(self.t_block_temp[1].weight, std=0.02)

        # Initialize caption embedding MLP:
        nn.init.normal_(self.y_embedder.y_proj.fc1.weight, std=0.02)
        nn.init.normal_(self.y_embedder.y_proj.fc2.weight, std=0.02)

        # Zero-out adaLN modulation layers in PixArt blocks:
        for block in self.blocks:
            nn.init.constant_(block.cross_attn.proj.weight, 0)
            nn.init.constant_(block.cross_attn.proj.bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)
        
        # Initialize map & box blocks
        for block in self.map_box_attn_blocks:
            nn.init.constant_(block.proj_out_1_a.weight, 0)
            nn.init.constant_(block.proj_out_1_b.weight, 0)
            nn.init.constant_(block.proj_out_2_a.weight, 0)
            nn.init.constant_(block.proj_out_2_b.weight, 0)

@MODELS.register_module("STDiT2-XL/2")
def STDiT2_XL_2(from_pretrained=None, **kwargs):
    architecture = kwargs.pop('architecture', 'Non-Control') # 'Non-Control' 'ControlDiT' 'BiMoT' 'BiMoT-VAE'
    
    
    if from_pretrained is not None:
        if os.path.isdir(from_pretrained) or os.path.isfile(from_pretrained):
            # if it is a directory or a file, we load the checkpoint manually
            config = STDiT2Config(
                depth=28,
                hidden_size=1152,
                patch_size=(1, 2, 2),
                num_heads=16, **kwargs
            )
            if architecture == 'ControlDiT' or architecture == 'Non-Control':
                model = STDiT2(config)
            elif architecture == 'BiMoT':
                model = STDiT2_BiMoT(config)
            elif architecture == 'BiMoT-VAE':
                model = STDiT2_BiMoT_VAE(config)
            load_checkpoint(model, from_pretrained)
            if hasattr(model, 'initialize_duplicated_blocks') and config.duplicated_same:
                print("Initializing duplicated blocks ....................................")
                model.initialize_duplicated_blocks()
            return model
        else:
            # otherwise, we load the model from hugging face hub
            return STDiT2.from_pretrained(from_pretrained)
    else:
        # create a new model
        config = STDiT2Config(
            depth=28,
            hidden_size=1152,
            patch_size=(1, 2, 2),
            num_heads=16, **kwargs
        )
        if architecture == 'ControlDiT' or architecture == 'Non-Control':
            model = STDiT2(config)
        elif architecture == 'BiMoT':
            model = STDiT2_BiMoT(config)
        elif architecture == 'BiMoT-VAE':
            model = STDiT2_BiMoT_VAE(config)
    return model
