from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import math
import random
from diffusers.models.attention import BasicTransformerBlock, TemporalBasicTransformerBlock
from diffusers.models.resnet import SpatioTemporalResBlock, ResnetBlock2D


def downsample_block(is_convolutional_down, in_channels, out_channels):
    if is_convolutional_down:
        block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )
    else:
        block = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )
    return block


class GeneralConditionEncoder(nn.Module):
    """
        A simple convolutional bases encoder model for diverse condition
        Parameters:
            in_channels (`int`, *optional*, defaults to 3): The number of channels of the condition images(videos)
            out_channels (`int`, *optional*, defaults to 320): The number of channels the outputs
            inner_channels_factor (`int`, *optional*, defaults to 64): The number of inner channels factor
            downscale_factor: (`int`, *optional*, defaults to 8): The number of downscale factor sames to VAE of the SVD
            is_convolutional_down: (`bool`, *optional*, defaults to False): Use the convolution operation rather than maxpooling
            is_inverse_color: (`bool`, *optional*, defaults to False): Inverse the background color from white to black
    """

    def __init__(
            self,
            in_channels: int = 3,
            out_channels: int = 320,
            inner_channels_factor: int = 64,
            downscale_factor: int = 8,
            is_convolutional_down: bool = False,
            is_inverse_color: bool = False
    ):
        super().__init__()
        self.channels = [in_channels] + \
                        [int(inner_channels_factor * pow(2, i)) for i in range(int(math.log(downscale_factor, 2)) - 1)] + \
                        [out_channels]
        self.downsample_blocks = torch.nn.ModuleList([
            downsample_block(
                is_convolutional_down=is_convolutional_down,
                in_channels=self.channels[i],
                out_channels=self.channels[i + 1]
            ) for i in range(len(self.channels) - 1)
        ])
        self.is_inverse_color = is_inverse_color

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # if self.is_inverse_color:
        #     mask = 1. - (x >= 1).float()
        #     x = x * mask.to(x.device)

        for block in self.downsample_blocks:
            x = block(x)

        return x


class AdapterBlock(nn.Module):
    """
        A typical residual convolution block
        Parameters:
            in_channels (`int`, *optional*, defaults to 320): The number of channels of the condition images(videos)
            out_channels (`int`, *optional*, defaults to 320): The number of channels the outputs
            num_res_blocks: (`int`, *optional*, defaults to 2): The number of residual layers
            is_down: (`bool`, *optional*, defaults to False): Is to downsampling the features
            is_convolutional_down: (`bool`, *optional*, defaults to False): Use the convolution operation rather than maxpooling
    """

    def __init__(
            self,
            in_channels: int = 320,
            out_channels: int = 320,
            num_res_blocks: int = 2,
            camera_emb_dim: Optional[int] = None,
            is_down: bool = False,
            is_convolutional_down: bool = False,
    ):
        super().__init__()
        self.in_conv = None
        if is_down:
            self.downsample = downsample_block(
                is_convolutional_down=is_convolutional_down,
                in_channels=in_channels,
                out_channels=out_channels,
            )
        else:
            self.downsample = None
            if in_channels != out_channels:
                self.in_conv = nn.Sequential(
                    nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1),
                    nn.BatchNorm2d(out_channels),
                    nn.SiLU(inplace=True),
                )

        self.resnets = nn.ModuleList([
            ResnetBlock2D(in_channels=out_channels, temb_channels=camera_emb_dim, groups=2) for _ in range(num_res_blocks)
        ])

    def forward(self, x: torch.Tensor, cam_emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.downsample is not None:
            x = self.downsample(x)

        if self.in_conv is not None:
            x = self.in_conv(x)

        for block in self.resnets:
            x = block(x, cam_emb)
        return x


class ConditionImageAdapter(torch.nn.Module):
    """
        A residual convolutional bases encoder model for diverse condition
        Parameters:
            in_channels (`int`, *optional*, defaults to 3): The number of channels of the condition images(videos)
            out_channels (`list`, *optional*, defaults to [320, 640, 1280, 1280]): The number of channels of every layer
            is_down_blocks (`list`, *optional*, defaults to [True, True, True, False]): Do downsampling in adapter layer
            num_res_blocks: (`int`, *optional*, defaults to 2): The number of residual layers
            inner_channels_factor (`int`, *optional*, defaults to 64): The number of inner channels factor
            downscale_factor: (`int`, *optional*, defaults to 8): The number of downscale factor sames to VAE of the SVD
            is_convolutional_down: (`bool`, *optional*, defaults to False): Use the convolution operation rather than maxpooling
            is_inverse_color: (`bool`, *optional*, defaults to False): Inverse the background color from white to black
    """

    def __init__(
            self,
            in_channels: int = 3,
            out_channels: list = [320, 640, 1280, 1280],
            is_down_blocks: list = [True, True, True, False],
            num_res_blocks: int = 2,
            inner_channel_factor: int = 64,
            downscale_factor: int = 8,
            camera_emb_dim: Optional[int] = None,
            is_convolutional_down: bool = False,
            is_inverse_color: bool = False
    ):
        super().__init__()
        self.conv_in = GeneralConditionEncoder(
            in_channels=in_channels,
            out_channels=out_channels[0],
            inner_channels_factor=inner_channel_factor,
            downscale_factor=downscale_factor,
            is_convolutional_down=is_convolutional_down,
            is_inverse_color=is_inverse_color,
        )
        self.body = torch.nn.ModuleList(
            [
                AdapterBlock(
                    in_channels=out_channels[0],
                    out_channels=out_channels[0],
                    num_res_blocks=num_res_blocks,
                    camera_emb_dim=camera_emb_dim,
                    is_down=False,
                    is_convolutional_down=is_convolutional_down,
                )
            ] + [
                AdapterBlock(
                    in_channels=out_channels[max(i - 1, 0)],
                    out_channels=out_channels[i],
                    num_res_blocks=num_res_blocks,
                    camera_emb_dim=camera_emb_dim,
                    is_down=is_down_blocks[i],
                    is_convolutional_down=is_convolutional_down,
                ) for i in range(len(out_channels))
            ]
        )

    def forward(self, x: torch.Tensor, cam_emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        features = []
        x = self.conv_in(x)
        for block in self.body:
            x = block(x, cam_emb)
            features.append(x)
        return features


class AdapterFusion(nn.Module):
    """
        A simple fusion model for conditional features
        Parameters:
            out_channels (`list`, *optional*, defaults to [320, 640, 1280, 1280]): The number of channels of every layer
            split_num ('int', *optional*, defaults to 2): The number of conditions
    """

    def __init__(
            self,
            out_channels: list = [320, 640, 1280, 1280],
            split_num: int = 2,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.split_num = split_num
        self.fuse_conv = torch.nn.ModuleList(
            [
                torch.nn.Conv2d(out_channels[0], out_channels[0], kernel_size=3, stride=1, padding=1),
            ] + [
                torch.nn.Conv2d(out_channels[i], out_channels[i], kernel_size=3, stride=1, padding=1)
                for i in range(len(out_channels))
            ]
        )

    def forward(self, feature_list: list):
        out_features = []
        for i in range(len(self.out_channels) + 1):
            out_features.append(
                self.fuse_conv[i](
                    torch.cat(
                        [feature_list[j][i] for j in range(self.split_num)], dim=1
                    )
                )
            )
        return out_features


class CameraEmbProj(nn.Module):
    """
    A camara embedding generate model for multi-view circumstance.

    Parameters:
        camera_num (`int`, *optional*, defaults to 6): The number of cameras.
        inner_channels (`int`, *optional*, defaults to 128): The number of channels of the initial embedding.
        out_channels (`int`, *optional*, defaults to 1024): The number of channels of the camara embedding.
    """

    def __init__(
            self,
            camera_num: int = 6,
            inner_channels: int = 128,
            out_channels: int = 1024,
    ):
        super().__init__()
        self.time_params = nn.Parameter(torch.FloatTensor(camera_num, inner_channels), requires_grad=True)
        self.time_params.data.fill_(0.25)
        self.time_proj = nn.Sequential(
            nn.Linear(in_features=inner_channels, out_features=inner_channels),
            nn.SiLU(inplace=True),
            nn.Linear(in_features=inner_channels, out_features=inner_channels),
            nn.SiLU(inplace=True),
            nn.Linear(in_features=inner_channels, out_features=out_channels)
        )

    def forward(self, camera_ids: int = 6, batch_size: int = 1) -> torch.Tensor:
        if isinstance(camera_ids, int):
            return self.forward_single_camera(camera_ids, batch_size)
        else:
            return self.forward_all_camera(batch_size)

    def forward_single_camera(self, camera_num: int = 6, batch_size: int = 1) -> torch.Tensor:
        x = self.time_params[camera_num].unsqueeze(0)
        x = torch.repeat_interleave(x, batch_size, dim=0)
        x = self.time_proj(x)
        return x

    def forward_all_camera(self, batch_size: int = 1) -> torch.Tensor:
        x = self.time_params.unsqueeze(0)
        x = torch.repeat_interleave(x, batch_size, dim=0)
        return self.time_proj(x)

class CameraAttrEmb(nn.Module):
    """
    Embed CameraConditions(Rotation,Translation) shape = (bs*frame, 4, 27) to embedding space(bs*frame, 1, h, w)16,3,256,512
    yyn 20240524

    Parameters:
        pass
    """
    def __init__(
            self,
            hin: int = 4,
            win: int = 27,
            hout: int = 256,
            wout: int = 512,
    ):
        super().__init__()
        self.AttrEmb = nn.Sequential(
            nn.Linear(in_features=hin*win, out_features=wout),
            nn.SiLU(inplace=True),
            nn.Linear(in_features=wout, out_features=hout*wout),
            nn.SiLU(inplace=True)
        )
        self.hin = hin
        self.hout = hout
        self.win = win
        self.wout = wout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.AttrEmb(x.flatten(-2, -1))
        x = x.reshape(-1, self.hout, self.wout).unsqueeze(1).repeat(1,3,1,1)
        #print(x.shape)
        return x



class BiMoT(nn.Module):
    """
    A modulate Transformer model for video-like data.

    Parameters:
        num_attention_heads (`int`, *optional*, defaults to 16): The number of heads to use for multi-head attention.
        attention_head_dim (`int`, *optional*, defaults to 88): The number of channels in each head.
        temporal_attention_head_dim (`int`, *optional*, defaults to 88): The number of channels in each temporal head.
        in_channels (`int`, *optional*): The number of channels in the input and output.
        num_layers (`int`, *optional*, defaults to 1): The number of layers of Transformer blocks to use.
    """

    def __init__(
            self,
            num_attention_heads: int = 16,
            attention_head_dim: int = 88,
            temporal_attention_head_dim: int = 40,
            in_channels: int = 320,
            num_layers: int = 1,
    ):
        super().__init__()
        # Define the factors
        inner_dim = num_attention_heads * attention_head_dim
        temporal_num_attention_heads = in_channels // temporal_attention_head_dim

        # 2. Define input layers
        self.norm_hidden = torch.nn.GroupNorm(num_groups=4, num_channels=in_channels, eps=1e-6)
        self.norm_condition = torch.nn.GroupNorm(num_groups=4, num_channels=in_channels, eps=1e-6)

        self.proj_in_hidden_1 = nn.Linear(in_channels * 2, inner_dim)
        self.proj_in_hidden_2 = nn.Linear(in_channels * 2, inner_dim)
        self.proj_in_condition_1 = nn.Linear(in_channels, inner_dim)
        self.proj_in_condition_2 = nn.Linear(in_channels, inner_dim)

        # 3. Define transformers blocks
        self.spatial_transformer_blocks_1 = nn.ModuleList(
            [
                BasicTransformerBlock(
                    inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    cross_attention_dim=inner_dim,
                )
                for d in range(num_layers)
            ]
        )

        self.temporal_transformer_blocks = TemporalBasicTransformerBlock(
            in_channels,
            in_channels,
            temporal_num_attention_heads,
            temporal_attention_head_dim,
        )

        self.spatial_transformer_blocks_2 = nn.ModuleList(
            [
                BasicTransformerBlock(
                    inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    cross_attention_dim=inner_dim,
                )
                for d in range(num_layers)
            ]
        )

        # 4. Define output layers
        self.proj_out_1_a = zero_module(nn.Linear(inner_dim, in_channels))
        self.proj_out_1_b = zero_module(nn.Linear(inner_dim, in_channels))
        self.proj_out_2_a = zero_module(nn.Linear(inner_dim, in_channels))
        self.proj_out_2_b = zero_module(nn.Linear(inner_dim, in_channels))
        
        self.batchnorm_1 = nn.BatchNorm2d(in_channels)
        self.batchnorm_2 = nn.BatchNorm2d(in_channels)

    def forward(
            self,
            hidden_states: torch.FloatTensor,
            additional_conditions: torch.FloatTensor,
            image_only_indicator: torch.Tensor,
    ):
        # 1. Input
        
        if not additional_conditions.shape[0] == hidden_states.shape[0]:
            ratio = hidden_states.shape[0] // additional_conditions.shape[0]
            additional_conditions = torch.repeat_interleave(additional_conditions, ratio, dim=0)
        if not additional_conditions.shape[1] == hidden_states.shape[1]:
            ratio = hidden_states.shape[1] // additional_conditions.shape[1]
            additional_conditions = torch.repeat_interleave(additional_conditions, ratio, dim=1)

        assert additional_conditions.shape == hidden_states.shape
        batch_frames, channel, height, width = hidden_states.shape
        residual_hidden = hidden_states
        residual_cond = additional_conditions
        hidden_states_input = self.norm_hidden(hidden_states)
        condition_states = self.norm_condition(additional_conditions)

        inner_dim = hidden_states.shape[1]
        hidden_states = hidden_states_input.permute(0, 2, 3, 1).reshape(batch_frames, height * width, inner_dim)
        condition_states = condition_states.permute(0, 2, 3, 1).reshape(batch_frames, height * width, inner_dim)

        hidden_states = self.proj_in_hidden_1(torch.cat([hidden_states, condition_states], dim=2))

        condition_states = self.proj_in_condition_1(condition_states)

        # 2. Blocks
        for block in self.spatial_transformer_blocks_1:
            condition_states = block(
                condition_states,
                encoder_hidden_states=hidden_states,
            )

        # 3. Output
        signal_a = self.proj_out_1_a(condition_states)
        signal_a = signal_a.reshape(batch_frames, height, width, inner_dim).permute(0, 3, 1, 2).contiguous()
        signal_b = self.proj_out_1_a(condition_states)
        signal_b = signal_b.reshape(batch_frames, height, width, inner_dim).permute(0, 3, 1, 2).contiguous()

        residual_cond_norm = self.batchnorm_1(residual_cond)
        condition_states = signal_a * residual_cond_norm + signal_b + residual_cond

        # 4. Temporal Augment
        num_frames = image_only_indicator.shape[-1]
        batch_frames, inner_dim, height, width = condition_states.shape
        condition_states = condition_states.permute(0, 2, 3, 1).reshape(batch_frames, height * width, inner_dim)

        
        condition_states = self.temporal_transformer_blocks(condition_states, num_frames).reshape(batch_frames, height, width, inner_dim).permute(0, 3, 1, 2).contiguous()

        # 5. Input
        condition_states = self.norm_condition(condition_states)

        hidden_states = hidden_states_input.permute(0, 2, 3, 1).reshape(batch_frames, height * width, inner_dim)
        condition_states = condition_states.permute(0, 2, 3, 1).reshape(batch_frames, height * width, inner_dim)

        hidden_states = self.proj_in_hidden_2(
            torch.cat([hidden_states, condition_states], dim=2))

        condition_states = self.proj_in_condition_2(condition_states)

        # 6. Blocks
        for block in self.spatial_transformer_blocks_2:
            hidden_states = block(
                hidden_states,
                encoder_hidden_states=condition_states,
            )

        # 7. Output
        signal_a = self.proj_out_2_a(hidden_states)
        signal_a = signal_a.reshape(batch_frames, height, width, inner_dim).permute(0, 3, 1, 2).contiguous()
        signal_b = self.proj_out_2_b(hidden_states)
        signal_b = signal_b.reshape(batch_frames, height, width, inner_dim).permute(0, 3, 1, 2).contiguous()

        residual_hidden_norm = self.batchnorm_2(residual_hidden)
        output = signal_a * residual_hidden_norm + signal_b + residual_hidden

        return output


def zero_module(module):
    #weight init to zero
    for p in module.parameters():
        torch.nn.init.zeros_(p)
    return module



class MaskGenerator:
    def __init__(self, mask_ratios):
        valid_mask_names = [
            "mask_no",
            "mask_quarter_random",
            "mask_quarter_head",
            "mask_quarter_tail",
            "mask_quarter_head_tail",
            "mask_image_random",
            "mask_image_head",
            "mask_image_tail",
            "mask_image_head_tail",
        ]
        assert all(
            mask_name in valid_mask_names for mask_name in mask_ratios.keys()
        ), f"mask_name should be one of {valid_mask_names}, got {mask_ratios.keys()}"
        assert all(
            mask_ratio >= 0 for mask_ratio in mask_ratios.values()
        ), f"mask_ratio should be greater than or equal to 0, got {mask_ratios.values()}"
        assert all(
            mask_ratio <= 1 for mask_ratio in mask_ratios.values()
        ), f"mask_ratio should be less than or equal to 1, got {mask_ratios.values()}"
        # sum of mask_ratios should be 1
        assert math.isclose(
            sum(mask_ratios.values()), 1.0, abs_tol=1e-6
        ), f"sum of mask_ratios should be 1, got {sum(mask_ratios.values())}"
        print(f"mask ratios: {mask_ratios}")
        self.mask_ratios = mask_ratios

    def get_mask(self, x):
        mask_type = random.random()
        mask_name = None
        prob_acc = 0.0
        for mask, mask_ratio in self.mask_ratios.items():
            prob_acc += mask_ratio
            if mask_type < prob_acc:
                mask_name = mask
                break

        num_frames = x.shape[2]
        # Hardcoded condition_frames
        condition_frames_max = num_frames // 4

        mask = torch.ones(num_frames, dtype=torch.bool, device=x.device)
        if num_frames <= 1:
            return mask

        if mask_name == "mask_quarter_random":
            random_size = random.randint(1, condition_frames_max)
            random_pos = random.randint(0, x.shape[2] - random_size)
            mask[random_pos : random_pos + random_size] = 0
        elif mask_name == "mask_image_random":
            random_size = 1
            random_pos = random.randint(0, x.shape[2] - random_size)
            mask[random_pos : random_pos + random_size] = 0
        elif mask_name == "mask_quarter_head":
            random_size = random.randint(1, condition_frames_max)
            mask[:random_size] = 0
        elif mask_name == "mask_image_head":
            random_size = 1
            mask[:random_size] = 0
        elif mask_name == "mask_quarter_tail":
            random_size = random.randint(1, condition_frames_max)
            mask[-random_size:] = 0
        elif mask_name == "mask_image_tail":
            random_size = 1
            mask[-random_size:] = 0
        elif mask_name == "mask_quarter_head_tail":
            random_size = random.randint(1, condition_frames_max)
            mask[:random_size] = 0
            mask[-random_size:] = 0
        elif mask_name == "mask_image_head_tail":
            random_size = 1
            mask[:random_size] = 0
            mask[-random_size:] = 0

        return mask

    def get_masks(self, x):
        masks = []
        for _ in range(len(x)):
            mask = self.get_mask(x)
            masks.append(mask)
        masks = torch.stack(masks, dim=0)
        return masks
    
    
    
# 创建一个一维卷积层来压缩时间维度
class TimeCompressionCNN(nn.Module):
    def __init__(self, input_channels,num_frames, target_t_dim):
        super(TimeCompressionCNN, self).__init__()
        self.conv1d = nn.Conv1d(input_channels, input_channels, kernel_size=num_frames//target_t_dim, stride=num_frames//target_t_dim)

    def forward(self, x):
        B,T,C,H,W = x.shape
        x = x.permute(0, 3, 4, 1, 2) # (b, h, w, c, t)
        
        x = x.reshape(B*H*W, C, T)  # (b*h*w, c, t)
        x = self.conv1d(x) # (b*h*w, c, target_t_dim)
        
        
        x = x.reshape(B, H, W, -1, x.shape[-1])# (b, h, w, c, target_t_dim)
        x = x.permute(0, 4, 3, 1, 2)# (b,  target_t_dim,c, h, w)
        return x
