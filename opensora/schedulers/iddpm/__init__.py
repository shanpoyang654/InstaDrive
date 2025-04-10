from functools import partial

import torch

from opensora.registry import SCHEDULERS

from . import gaussian_diffusion as gd
from .respace import SpacedDiffusion, space_timesteps
from .speed import SpeeDiffusion


@SCHEDULERS.register_module("iddpm")
class IDDPM(SpacedDiffusion):
    def __init__(
        self,
        num_sampling_steps=None,
        timestep_respacing=None,
        noise_schedule="linear",
        use_kl=False,
        sigma_small=False,
        predict_xstart=False,
        learn_sigma=True,
        rescale_learned_sigmas=False,
        diffusion_steps=1000,
        cfg_scale=4.0,
        cfg_channel=None,
    ):
        betas = gd.get_named_beta_schedule(noise_schedule, diffusion_steps)
        if use_kl:
            loss_type = gd.LossType.RESCALED_KL
        elif rescale_learned_sigmas:
            loss_type = gd.LossType.RESCALED_MSE
        else:
            loss_type = gd.LossType.MSE
        if num_sampling_steps is not None:
            assert timestep_respacing is None
            timestep_respacing = str(num_sampling_steps)
        if timestep_respacing is None or timestep_respacing == "":
            timestep_respacing = [diffusion_steps]
        super().__init__(
            use_timesteps=space_timesteps(diffusion_steps, timestep_respacing),
            betas=betas,
            model_mean_type=(gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X),
            model_var_type=(
                (gd.ModelVarType.FIXED_LARGE if not sigma_small else gd.ModelVarType.FIXED_SMALL)
                if not learn_sigma
                else gd.ModelVarType.LEARNED_RANGE
            ),
            loss_type=loss_type,
            # rescale_timesteps=rescale_timesteps,
        )

        self.cfg_scale = cfg_scale
        self.cfg_channel = cfg_channel

    def sample(
        self,
        model,
        text_encoder,
        prompts,
        device,
        z=None,
        z_0=None,
        z_size=None,
        additional_args=None,
        mask=None,
        enable_condition=False,
        **kwargs
    ):
        n = len(prompts)        
        if z is None:
            z = torch.randn(n, *z_size, device=device)
        if z_0 is None:
            z = torch.cat([z, z], 0)
        else:
            z = torch.cat([z, z_0], 0)
                              
        model_args = text_encoder.encode(prompts)
        y_null = text_encoder.null(n)
        model_args["y"] = torch.cat([model_args["y"], y_null], 0)
        if additional_args is not None:
            model_args.update(additional_args)
        y_null = text_encoder.null(n)
        model_args["y"] = torch.cat([model_args["y"][:1], y_null], 0)
        model_args["height"] = model_args["height"][:1]
        model_args["width"] = model_args["width"][:1]
        model_args["full_height"] = model_args["full_height"][:1]
        model_args["full_width"] = model_args["full_width"][:1]
        model_args["ar"] = model_args["ar"][:1]
        model_args["fps"] = model_args["fps"][:1]
        model_args["num_frames"] = model_args["num_frames"][:1]
        model_args["mask"] = model_args["mask"][:1]
        
        if enable_condition:
            model_args["hdmap"] = torch.cat([model_args["hdmap"][:1], torch.zeros_like(model_args["hdmap"][:1])], 0)
            model_args["traj"] = torch.cat([model_args["traj"][:1], torch.zeros_like(model_args["traj"][:1])], 0)
            model_args["bbox"] = torch.cat([model_args["bbox"][:1], torch.zeros_like(model_args["bbox"][:1])], 0)
            model_args["layout_conditions"] = torch.cat([model_args["layout_conditions"][:1], torch.zeros_like(model_args["layout_conditions"][:1])], 0)
            model_args["traj_conditions"] = torch.cat([model_args["traj_conditions"][:1], torch.zeros_like(model_args["traj_conditions"][:1])], 0)
        
       
        
        forward = partial(forward_with_cfg, model, cfg_scale=self.cfg_scale, cfg_channel=self.cfg_channel)
        samples = self.p_sample_loop(
            forward,
            z.shape,
            z,
            clip_denoised=False,
            model_kwargs=model_args,
            progress=True,
            device=device,
            mask=mask,
        )
        samples, _ = samples.chunk(2, dim=0)
        return samples

def forward_with_cfg(model, x, timestep, y, cfg_scale, cfg_channel=None, **kwargs):
    # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
    half = x[: len(x) // 2]
    combined = torch.cat([half, half], dim=0)
    if "x_mask" in kwargs and kwargs["x_mask"] is not None:
        if len(kwargs["x_mask"]) != len(x):
            kwargs["x_mask"] = torch.cat([kwargs["x_mask"], kwargs["x_mask"]], dim=0)
    model_out = model.forward(combined, timestep, y, **kwargs)
    model_out = model_out["x"] if isinstance(model_out, dict) else model_out
    if cfg_channel is None:
        cfg_channel = model_out.shape[1] // 2
    eps, rest = model_out[:, :cfg_channel], model_out[:, cfg_channel:]
    cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
    half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
    eps = torch.cat([half_eps, half_eps], dim=0)
    return torch.cat([eps, rest], dim=1)
