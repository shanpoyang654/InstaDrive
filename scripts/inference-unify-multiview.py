import sys


import json
import os
import re
from tqdm import tqdm

import colossalai
import torch
import torch.nn.functional as F
import torch.distributed as dist
from colossalai.cluster import DistCoordinator
from mmengine.runner import set_random_seed

from opensora.datasets import IMG_FPS, save_sample
from opensora.datasets.utils import read_from_path
from opensora.models.text_encoder.t5 import text_preprocessing
from opensora.registry import MODELS, SCHEDULERS, build_module
from opensora.utils.config_utils import parse_configs, load_prompts
from opensora.utils.misc import to_torch_dtype
from opensora.common import *
from opensora.datasets.dataloader import prepare_dataloader
from opensora.acceleration.parallel_states import (
    get_data_parallel_group,
    set_data_parallel_group,
    set_sequence_parallel_group,
)


def pad_tensors_to_max_t(tensor_list):
    """
    Pads tensors in the list along the temporal dimension (dim=1) with zeros
    to match the maximum temporal length among them.

    Args:
        tensor_list (list of torch.Tensor): List of tensors with shape [3, t_i, h, w].

    Returns:
        list of torch.Tensor: List of tensors padded to the maximum temporal length.
    """
    # Verify that all tensors have 4 dimensions and the same number of channels, height, and width
    if not all(t.dim() == 4 for t in tensor_list):
        raise ValueError("All tensors must have 4 dimensions [3, t, h, w].")

    channels, _, height, width = tensor_list[0].shape
    for idx, tensor in enumerate(tensor_list):
        if tensor.shape[0] != channels or tensor.shape[2] != height or tensor.shape[3] != width:
            raise ValueError(f"All tensors must have the same channels, height, and width. "
                             f"Tensor {idx} has shape {tensor.shape}, expected [3, t, {height}, {width}].")

    # Determine the maximum temporal length
    max_t = max(tensor.shape[1] for tensor in tensor_list)

    padded_tensors = []
    for tensor in tensor_list:
        current_t = tensor.shape[1]
        if current_t < max_t:
            pad_t = max_t - current_t
            # Create a padding tensor of zeros with shape [3, pad_t, h, w]
            padding = torch.zeros(
                tensor.size(0), pad_t, tensor.size(2), tensor.size(3),
                device=tensor.device, dtype=tensor.dtype
            )
            # Concatenate the padding tensor along the temporal dimension
            tensor = torch.cat([tensor, padding], dim=1)
        padded_tensors.append(tensor)

    return padded_tensors


def concatenate_tensors(tensor_list, dim=3):
    """
    Concatenates a list of tensors along the specified dimension after ensuring
    their temporal dimensions are equal by padding the smaller tensors.

    Args:
        tensor_list (list of torch.Tensor): List of tensors with shape [3, t_i, h, w].
        dim (int): Dimension along which to concatenate (default: 3 for width).

    Returns:
        torch.Tensor: Concatenated tensor.
    """
    # Pad tensors to have the same temporal dimension
    padded_tensors = pad_tensors_to_max_t(tensor_list)

    # Concatenate along the specified dimension
    concatenated_tensor = torch.cat(padded_tensors, dim=dim)

    return concatenated_tensor

def collect_references_batch(reference_paths, vae, image_size):
    refs_x = []
    for reference_path in reference_paths:
        if reference_path is None:
            refs_x.append([])
            continue
        ref_path = reference_path.split(";")
        ref = []
        for r_path in ref_path:
            r = read_from_path(r_path, image_size, transform_name="resize_crop")
            r_x = vae.encode(r.unsqueeze(0).to(vae.device, vae.dtype))
            r_x = r_x.squeeze(0)
            ref.append(r_x)
        refs_x.append(ref)
    # refs_x: [batch, ref_num, C, T, H, W]
    return refs_x


def process_mask_strategy(mask_strategy):
    mask_batch = []
    mask_strategy = mask_strategy.split(";")
    for mask in mask_strategy:
        mask_group = mask.split(",")
        assert len(mask_group) >= 1 and len(mask_group) <= 6, f"Invalid mask strategy: {mask}"
        if len(mask_group) == 1:
            mask_group.extend(["0", "0", "0", "1", "0"])
        elif len(mask_group) == 2:
            mask_group.extend(["0", "0", "1", "0"])
        elif len(mask_group) == 3:
            mask_group.extend(["0", "1", "0"])
        elif len(mask_group) == 4:
            mask_group.extend(["1", "0"])
        elif len(mask_group) == 5:
            mask_group.append("0")
        mask_batch.append(mask_group)
    return mask_batch


def apply_mask_strategy(z, refs_x, mask_strategys, loop_i):
    masks = []
    for i, mask_strategy in enumerate(mask_strategys):
        mask = torch.ones(z.shape[2], dtype=torch.float, device=z.device)
        if mask_strategy is None:
            masks.append(mask)
            continue
        mask_strategy = process_mask_strategy(mask_strategy)
        for mst in mask_strategy:
            loop_id, m_id, m_ref_start, m_target_start, m_length, edit_ratio = mst
            loop_id = int(loop_id)
            if loop_id != loop_i:
                continue
            m_id = int(m_id)
            m_ref_start = int(m_ref_start)
            m_length = int(m_length)
            m_target_start = int(m_target_start)
            edit_ratio = float(edit_ratio)
            ref = refs_x[i][m_id]  # [C, T, H, W]
            if m_ref_start < 0:
                m_ref_start = ref.shape[1] + m_ref_start
            if m_target_start < 0:
                # z: [B, C, T, H, W]
                m_target_start = z.shape[2] + m_target_start
            z[i, :, m_target_start : m_target_start + m_length] = ref[:, m_ref_start : m_ref_start + m_length]
            mask[m_target_start : m_target_start + m_length] = edit_ratio
        masks.append(mask)
    masks = torch.stack(masks)
    return masks


def process_prompts(prompts, num_loop):
    ret_prompts = []
    for prompt in prompts:
        if prompt.startswith("|0|"):
            prompt_list = prompt.split("|")[1:]
            text_list = []
            for i in range(0, len(prompt_list), 2):
                start_loop = int(prompt_list[i])
                text = prompt_list[i + 1]
                text = text_preprocessing(text)
                end_loop = int(prompt_list[i + 2]) if i + 2 < len(prompt_list) else num_loop
                text_list.extend([text] * (end_loop - start_loop))
            assert len(text_list) == num_loop, f"Prompt loop mismatch: {len(text_list)} != {num_loop}"
            ret_prompts.append(text_list)
        else:
            prompt = text_preprocessing(prompt)
            ret_prompts.append([prompt] * num_loop)
    return ret_prompts


def extract_json_from_prompts(prompts):
    additional_infos = []
    ret_prompts = []
    for prompt in prompts:
        parts = re.split(r"(?=[{\[])", prompt)
        assert len(parts) <= 2, f"Invalid prompt: {prompt}"
        ret_prompts.append(parts[0])
        if len(parts) == 1:
            additional_infos.append({})
        else:
            additional_infos.append(json.loads(parts[1]))
    return ret_prompts, additional_infos


def main():
    # ======================================================
    # 1. cfg and init distributed env
    # ======================================================
    cfg = parse_configs(training=False)
    print(cfg)

    # init distributed
    if os.environ.get("WORLD_SIZE", None):
        use_dist = True
        colossalai.launch_from_torch({})
        coordinator = DistCoordinator()

        if coordinator.world_size > 1:
            set_sequence_parallel_group(dist.group.WORLD)
            enable_sequence_parallelism = True
        else:
            enable_sequence_parallelism = False
    else:
        use_dist = False
        enable_sequence_parallelism = False
    enable_sequence_parallelism = False # [todo] some bug exist if seq parallel enabled
    # ======================================================
    # 2. runtime variables
    # ======================================================
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = to_torch_dtype(cfg.dtype)
    set_random_seed(seed=cfg.seed)

    val_dataset = create_instance_from_config(cfg.val_dataset)
    val_dataloader, sampler_to_io = prepare_dataloader(
        val_dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        seed=cfg.get("seed", 42),
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        process_group=get_data_parallel_group(),
    )
    target_texts = []
    if cfg.prompt_path is not None:
        target_texts = load_prompts(cfg.prompt_path)

    # ======================================================
    # 3. build model & load weights
    # ======================================================
    # 3.1. build model
    input_size = (cfg.num_frames, *cfg.image_size)
    vae = build_module(cfg.vae, MODELS)
    latent_size = vae.get_latent_size(input_size)
    text_encoder = build_module(cfg.text_encoder, MODELS, device=device)  # T5 must be fp32
    model = build_module(
        cfg.model,
        MODELS,
        input_size=latent_size,
        in_channels=vae.out_channels,
        caption_channels=text_encoder.output_dim,
        model_max_length=text_encoder.model_max_length,
        enable_sequence_parallelism=enable_sequence_parallelism,
    )
    text_encoder.y_embedder = model.y_embedder  # hack for classifier-free guidance

    # 3.2. move to device & eval
    vae = vae.to(device, dtype).eval()
    model = model.to(device, dtype).eval()

    # 3.3. build scheduler
    scheduler = build_module(cfg.scheduler, SCHEDULERS)

    # 3.4. support for multi-resolution
    model_args = dict()
    if cfg.multi_resolution == "PixArtMS":
        image_size = cfg.image_size
        hw = torch.tensor([image_size], device=device, dtype=dtype).repeat(cfg.batch_size, 1)
        ar = torch.tensor([[image_size[0] / image_size[1]]], device=device, dtype=dtype).repeat(cfg.batch_size, 1)
        model_args["data_info"] = dict(ar=ar, hw=hw)
    elif cfg.multi_resolution == "STDiT2":
        image_size = cfg.image_size
        height = torch.tensor([image_size[0]], device=device, dtype=dtype).repeat(cfg.batch_size)
        width = torch.tensor([image_size[1]], device=device, dtype=dtype).repeat(cfg.batch_size)
        num_frames = torch.tensor([cfg.num_frames], device=device, dtype=dtype).repeat(cfg.batch_size)
        ar = torch.tensor([image_size[0] / image_size[1]], device=device, dtype=dtype).repeat(cfg.batch_size)
        if cfg.num_frames == 1:
            cfg.fps = IMG_FPS
        fps = torch.tensor([cfg.fps], device=device, dtype=dtype).repeat(cfg.batch_size)
        model_args["height"] = height
        model_args["width"] = width
        model_args["num_frames"] = num_frames
        model_args["ar"] = ar
        model_args["fps"] = fps

    # 3.5 reference
    '''
    if cfg.reference_path is not None:
        assert len(cfg.reference_path) == len(
            prompts
        ), f"Reference path mismatch: {len(cfg.reference_path)} != {len(prompts)}"
        assert len(cfg.reference_path) == len(
            cfg.mask_strategy
        ), f"Mask strategy mismatch: {len(cfg.mask_strategy)} != {len(prompts)}"
    else:
    '''


    # ======================================================
    # 4. inference
    # ======================================================
    sample_idx = 0
    save_dir = cfg.save_dir
    os.makedirs(save_dir, exist_ok=True)

    # 4.1. batch generation
    dataloader_iter = iter(val_dataloader)
    with tqdm(
            enumerate(dataloader_iter, 0),
            disable=not coordinator.is_master(),
            total=len(val_dataloader)) as pbar:
        for step, batch in pbar:
            x = batch.pop("video").to(device, dtype)  # [B, C, T, H, W]
            if len(x.shape) == 6: # 不支持多视，取第一个视角
                x = x[:, 0, ...]

            x0 = x.clone()
            x = x0[:, :, : cfg.num_frames]
            y = batch.pop("text")
            with torch.no_grad():
                # Prepare visual inputs
                x = vae.encode(x)  # [B, C, T, H/P, W/P]
                # Prepare text inputs
                # model_args = text_encoder.encode(y)
                model_args = dict()
            for k, v in batch.items():
                if k not in ["3dbox", "hdmap"]:
                    model_args[k] = v.to(device, dtype)

            refs_x = [t for t in torch.unbind(x[:cfg.condition_frame_length], dim=0)]
            refs_x = [refs_x]
            mask_strategy = [f"0,0,0,0,{cfg.condition_frame_length},0"]

            # batch_prompts_raw = prompts[i : i + cfg.batch_size]
            # batch_prompts_raw, additional_infos = extract_json_from_prompts(batch_prompts_raw)
            # batch_prompts_loops = process_prompts(batch_prompts_raw, cfg.loop)
            # handle the last batch
            # if len(batch_prompts_raw) < cfg.batch_size and cfg.multi_resolution == "STDiT2":
            #     model_args["height"] = model_args["height"][: len(batch_prompts_raw)]
            #     model_args["width"] = model_args["width"][: len(batch_prompts_raw)]
            #     model_args["num_frames"] = model_args["num_frames"][: len(batch_prompts_raw)]
            #     model_args["ar"] = model_args["ar"][: len(batch_prompts_raw)]
            #     model_args["fps"] = model_args["fps"][: len(batch_prompts_raw)]

            # 4.2. load reference videos & images
            # for j, info in enumerate(additional_infos):
            #     if "reference_path" in info:
            #         cfg.reference_path[i + j] = info["reference_path"]
            #     if "mask_strategy" in info:
            #         cfg.mask_strategy[i + j] = info["mask_strategy"]
            # refs_x = collect_references_batch(cfg.reference_path[i : i + cfg.batch_size], vae, cfg.image_size)
            # mask_strategy = cfg.mask_strategy[i : i + cfg.batch_size]

            # 4.3. diffusion sampling
            old_sample_idx = sample_idx
            # generate multiple samples for each prompt
            for k in range(max(len(target_texts), 1)):
                sample_idx = old_sample_idx
                video_clips = []

                # 4.4. long video generation
                for loop_i in range(cfg.loop):
                    # 4.4 sample in hidden space
                    # batch_prompts = [prompt[loop_i] for prompt in batch_prompts_loops]

                    # 4.5. apply mask strategy
                    masks = None
                    # if cfg.reference_path is not None:
                    if loop_i > 0:
                        ref_x = vae.encode(video_clips[-1])
                        for j, refs in enumerate(refs_x):
                            if refs is None:
                                refs_x[j] = [ref_x[j]]
                            else:
                                refs.append(ref_x[j])
                            if mask_strategy[j] is None:
                                mask_strategy[j] = ""
                            else:
                                mask_strategy[j] += ";"
                            mask_strategy[
                                j
                            ] += f"{loop_i},{len(refs)-1},-{cfg.condition_frame_length},0,{cfg.condition_frame_length}"

                    # sampling
                    z = torch.randn(1, vae.out_channels, *latent_size, device=device, dtype=dtype)
                    z_0 = z.clone()
                    if cfg.infer_mode == 't2v' and loop_i==0: # t2v模式下第一轮推理不加条件帧
                        # mask_strategy = [f"0,0,0,0,0,{cfg.edit_ratio}"]
                        # refs_x = []
                        pass
                    else:
                        if loop_i==0: # 编辑模式，第0次迭代对条件帧进行编辑，若要对视频进行整体编辑可以将条件帧帧数设为视频总帧数
                            mask_strategy = [f"0,0,0,0,{cfg.condition_frame_length},{cfg.edit_ratio}"]
                        masks = apply_mask_strategy(z, refs_x, mask_strategy, loop_i)

                    if cfg.model.get("enable_condition", False):

                        model_args["layout_conditions"] = batch["hdmap"][:1, loop_i * cfg.num_frames:(loop_i + 1) * cfg.num_frames].to(device, dtype) + \
                                                          batch["3dbox"][:1, loop_i * cfg.num_frames:(loop_i + 1) * cfg.num_frames].to(device, dtype)

                        if loop_i > 0:
                            model_args["layout_conditions"] = batch["hdmap"][:1,loop_i * (cfg.num_frames - 4):loop_i * (cfg.num_frames - 4) + cfg.num_frames].to(device, dtype) + \
                                                              batch["3dbox"][:1, loop_i * (cfg.num_frames - 4):loop_i * (cfg.num_frames - 4) + cfg.num_frames].to(device, dtype)

                        model_args["layout_conditions"] = vae.encode(
                            model_args["layout_conditions"].permute(0, 2, 1, 3, 4))
                    samples = scheduler.sample(
                        model,
                        text_encoder,
                        z=z,
                        z_0=z_0,
                        prompts=y[:1] if len(target_texts)==0 else [target_texts[k]],
                        device=device,
                        additional_args=model_args,
                        progress=True,
                        mask=masks,
                        enable_condition=cfg.model.get("enable_condition", False)
                    )
                    samples = vae.decode(samples.to(dtype))
                    video_clips.append(samples)

                    # 4.7. save video
                    if loop_i == cfg.loop - 1:
                        for idx in range(len(video_clips[0])):
                            video_clips_i = [video_clips[0][idx]] + [
                                video_clips[i][idx][:, cfg.condition_frame_length :] for i in range(1, cfg.loop)
                            ]
                            video = torch.cat(video_clips_i, dim=1)
                            # video = torch.cat([video, torch.zeros(3, x0[0].shape[1] - video.shape[1], video.shape[2], video.shape[3]).to(device, dtype)], dim=1)
                            save_path = os.path.join(save_dir, f"step_{step:08d}_rank_{dist.get_rank():02d}_text_{k:02d}_video")
                            # if cfg.num_sample != 1:
                            #     save_path = f"{save_path}-{k}"
                            # save_sample(video, fps=cfg.fps // cfg.frame_interval, save_path=save_path)
                            if cfg.model.get("enable_condition", False):
                                condition_images = batch["hdmap"].to(device, dtype)[0].detach() + \
                                                   batch["3dbox"].to(device, dtype)[0].detach()
                                condition_images = condition_images.permute(1, 0, 2, 3)
                                cat_img = concatenate_tensors([x0[0], condition_images, video], dim=3)
                                save_sample(cat_img, fps=cfg.save_fps, save_path=save_path)
                            else:
                                # cat_img = torch.cat([x0[0], video], dim=3)
                                cat_img = concatenate_tensors([x0[0], video], dim=3)
                                save_sample(cat_img, fps=cfg.save_fps, save_path=save_path)
                            with open(os.path.join(save_dir, f"step_{step:08d}_rank_{dist.get_rank():02d}_text_{k:02d}_prompt.txt"), "w") as fp:
                                fp.write(y[0] if len(target_texts)==0 else target_texts[k])
                            sample_idx += 1

if __name__ == "__main__":
    main()
