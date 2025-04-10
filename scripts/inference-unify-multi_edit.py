import sys


import json
import os
import re
from tqdm import tqdm
import math

import colossalai
import torch
import torch.nn.functional as F
import torch.distributed as dist
from colossalai.cluster import DistCoordinator
from mmengine.runner import set_random_seed

from opensora.datasets import IMG_FPS
from opensora.datasets.utils import read_from_path, save_sample
from opensora.models.text_encoder.t5 import text_preprocessing
from opensora.registry import MODELS, SCHEDULERS, build_module
from opensora.utils.misc import to_torch_dtype
from opensora.utils.train_utils import MaskGenerator
from opensora.common import *
from opensora.datasets.dataloader import prepare_dataloader
from opensora.acceleration.parallel_states import (
    get_data_parallel_group,
    set_data_parallel_group,
    set_sequence_parallel_group,
)

from opensora.utils.config_utils import (
    define_experiment_workspace,
    parse_configs
)

# === mmdet3d ===
from opensora.datasets import NuScenesTDataset, CarlaDataset
from mmdet3d.datasets.pipelines import LoadMultiViewImageFromFiles, Collect3D
# === mmdet3d ===
from einops import rearrange
from glob import glob
from torchvision.utils import save_image
import ipdb


'''
condition_frame_length 1
vae view-encoded
'''

from PIL import Image
def save_imgs(x, label_str, dir, camera_list, img_path, normalize=True, value_range=(-1, 1)):
    label_dir = os.path.join(dir, label_str)
    os.makedirs(label_dir, exist_ok=True)
    for camera in camera_list:
        os.makedirs(os.path.join(label_dir, camera), exist_ok=True)
    
    # x has shape [C, T, H, W*V]
    C, T, H, WV = x.shape
    V = len(camera_list)  # Number of camera views
    W = WV // V  # Compute the width for each view
    for t in range(T):
        images_per_view = torch.split(x[:, t], W, dim=-1)  # Split along width dimension
        for v in range(V):
            img = images_per_view[v]  # Shape [C, H, W]
            img_file_path = img_path[t][v]
            camera_name = camera_list[v]
            save_path = os.path.join(label_dir, camera_name, os.path.basename(img_file_path))
            
            # if img.min() < 0 or img.max() > 1:
            #     img = (img - img.min()) / (img.max() - img.min())
            if normalize:
                if value_range == "cond":
                    value_range = (0,1)
                low, high = value_range
                img = img.clamp(min=low, max=high)
                img = (img - low) / (max(high - low, 1e-5))
                

            img = img.permute(1, 2, 0)
            img = (img * 255).byte().to("cpu").numpy()
            im = Image.fromarray(img)
            im.save(save_path)
            
            # save_image(img, save_path, normalize=False, value_range=(0,1))
            print(f"Saved {save_path}")
    


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



def pad_tensors_to_min_t(tensor_list):
    """
    Pads tensors in the list along the temporal dimension (dim=1) with zeros
    to match the minimum temporal length among them.

    Args:
        tensor_list (list of torch.Tensor): List of tensors with shape [3, t_i, h, w].

    Returns:
        list of torch.Tensor: List of tensors padded to the minimum temporal length.
    """
    # Verify that all tensors have 4 dimensions and the same number of channels, height, and width
    if not all(t.dim() == 4 for t in tensor_list):
        raise ValueError("All tensors must have 4 dimensions [3, t, h, w].")

    channels, _, height, width = tensor_list[0].shape
    for idx, tensor in enumerate(tensor_list):
        if tensor.shape[0] != channels or tensor.shape[2] != height or tensor.shape[3] != width:
            raise ValueError(f"All tensors must have the same channels, height, and width. "
                             f"Tensor {idx} has shape {tensor.shape}, expected [3, t, {height}, {width}].")

    # Find the minimum temporal length among all tensors
    min_t = min(tensor.shape[1] for tensor in tensor_list)

    # Trim tensors to match the minimum temporal length
    trimmed_tensors = [tensor[:, :min_t, :, :] for tensor in tensor_list]

    return trimmed_tensors

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
    padded_tensors = pad_tensors_to_min_t(tensor_list)

    # Concatenate along the specified dimension
    concatenated_tensor = torch.cat(padded_tensors, dim=dim)

    return concatenated_tensor


def concatenate_tensors_add(video_with_bbox, video, dim=1):
    # Pad tensors to have the same temporal dimension
    tensor_list = [video_with_bbox, video]
    padded_tensors = pad_tensors_to_min_t(tensor_list)

    # Concatenate along the specified dimension
    # concatenated_tensor = torch.cat(padded_tensors, dim=dim)
    concatenated_tensor = padded_tensors[0] + padded_tensors[1]

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
            # TODO:
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
    
    # TODO:
    enable_sequence_parallelism = False

    # ======================================================
    # 2. runtime variables
    # ======================================================
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = to_torch_dtype(cfg.dtype)
    set_random_seed(seed=cfg.seed)

    
    # ======================================================
    # build dataset and dataloader
    # ======================================================
    if cfg.dataset_version == "NuScenes" and cfg.get("val_dataset", None):
        val_dataset = create_instance_from_config(cfg.val_dataset)
    elif cfg.dataset_version == "NuScenes_interpolated_12Hz" and cfg.get("val_dataset", None): 
        pipeline = [
            LoadMultiViewImageFromFiles(camera_list=cfg.camera_list),
            Collect3D(
                keys=["img", 'description'],
                meta_lis_keys=["filename"]
            )
        ]
        val_dataset_args = cfg.val_dataset.copy()
        val_dataset_args.pop('_class_name')
        val_dataset_args.pop('transform_name')
        val_dataset = NuScenesTDataset(**val_dataset_args,
                                        pipeline=pipeline)
    elif cfg.dataset_version == "Carla" and cfg.get("val_dataset", None): 
        pipeline = [
            LoadMultiViewImageFromFiles(camera_list=cfg.camera_list),
            Collect3D(
                keys=["img", 'description'],
                meta_lis_keys=["filename"]
            )
        ]
        
        val_dataset_args = cfg.val_dataset.copy()
        val_dataset_args.pop('_class_name')
        val_dataset_args.pop('transform_name')
        val_dataset = CarlaDataset(**val_dataset_args,
                                        pipeline=pipeline)
    
    
    val_dataloader, sampler_to_io = prepare_dataloader(
        val_dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        seed=cfg.get("seed", 42),
        shuffle=True, # TODO:
        drop_last=True,
        pin_memory=True,
        process_group=get_data_parallel_group(),
    )

    # ======================================================
    # 3. build model & load weights
    # ======================================================
    # 3.1. build model
    input_size = (cfg.num_frames, *cfg.full_size)
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
        full_size = cfg.full_size
        height = torch.tensor([image_size[0]], device=device, dtype=dtype).repeat(cfg.batch_size)
        width = torch.tensor([image_size[1]], device=device, dtype=dtype).repeat(cfg.batch_size)
        full_height = torch.tensor([full_size[0]], device=device, dtype=dtype).repeat(cfg.batch_size)
        full_width = torch.tensor([full_size[1]], device=device, dtype=dtype).repeat(cfg.batch_size)
        num_frames = torch.tensor([cfg.num_frames], device=device, dtype=dtype).repeat(cfg.batch_size)
        ar = torch.tensor([image_size[0] / image_size[1]], device=device, dtype=dtype).repeat(cfg.batch_size)
        if cfg.num_frames == 1:
            cfg.fps = IMG_FPS
        fps = torch.tensor([cfg.fps], device=device, dtype=dtype).repeat(cfg.batch_size)
        model_args["height"] = height
        model_args["width"] = width
        model_args["height"] = full_height
        model_args["width"] = full_width
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
    '''
    if cfg.sample_name is not None:
        sample_name = cfg.sample_name
    elif cfg.prompt_as_path:
        sample_name = ""
    else:
    '''
    
    # Initialize save_dir as an empty string
    save_dir = ""
    # === Create an experiment folder === 
    if not use_dist or coordinator.is_master():
        os.makedirs(cfg.save_dir, exist_ok=True)
        
        output_dirs = glob(f"{cfg.save_dir}/*")
        indices = []
        for dir_name in output_dirs:
            match = re.search(r'/(\d+)-', dir_name)
            if match:
                indices.append(int(match.group(1)))
        if indices:
            experiment_index = max(indices)
        else:
            experiment_index = 0
        experiment_index += 1
            
        model_name = cfg.model["type"].replace("/", "-")
        exp_name = f"{experiment_index:03d}-{model_name}"
        save_dir = f"{cfg.save_dir}/{exp_name}"
        # save_dir = '/mnt/iag/user/yangzhuoran/local/stable/metavdt/outputs/infer/nuscenes_osv1_1_ti2v_baseline_2view_2dataloader_2model/066-STDiT2-XL-2/'
        os.makedirs(save_dir, exist_ok=True)
        
        # TODO: make dir
        os.makedirs(os.path.join(save_dir, f"img"), exist_ok=True)
        os.makedirs(os.path.join(save_dir, f"video"), exist_ok=True)
    
    # 所有进程等待主进程创建目录
    if use_dist:
        # torch.distributed.barrier()
        save_dir_list = [save_dir]  # 必须使用可变对象（如列表）
        torch.distributed.broadcast_object_list(save_dir_list, src=0)
        save_dir = save_dir_list[0]  # 非主进程获取到 save_dir
    print("save_dir: rank_", dist.get_rank(), " _: ", save_dir) 
    
    # save_dir = '/mnt/iag/user/yangzhuoran/local/stable/metavdt/outputs/infer/nuscenes_osv1_1_ti2v_baseline_2view_2dataloader_2model/066-STDiT2-XL-2/'
    
    #     # Broadcast the save_dir to all ranks
    #     save_dir = torch.tensor(list(save_dir.encode('utf-8')), dtype=torch.uint8)
    #     save_dir = torch.nn.functional.pad(save_dir, (0, 1024 - len(save_dir)))  # Pad to fixed size if necessary
    #     dist.broadcast(save_dir, 0)  # Broadcast from rank 0 to all other ranks
    # else:
    #     # Other ranks receive the save_dir from rank 0
    #     save_dir = torch.zeros(1024, dtype=torch.uint8)  # Assuming the path won't exceed 256 characters
    #     dist.broadcast(save_dir, 0)  # Receive from rank 0
    #     save_dir = save_dir.cpu().numpy().tobytes().decode('utf-8').strip('\x00')  # Convert back to string
        

    # 4.1. batch generation
    dataloader_iter = iter(val_dataloader)
    with tqdm(
            enumerate(dataloader_iter, 0),
            # disable=not coordinator.is_master(),
            total=len(val_dataloader)) as pbar:
        for step, batch in pbar:
            # TODO:
            
            
            # if step not in [1178]:
            #     continue   
            
            
            x = batch.pop("video").to(device, dtype)  # [B, V, C, T, H, W]
            x0 = x.clone()
            x = x0[:, :, : cfg.num_frames]
            V = x0.shape[1]
            x0 = torch.cat([x0[:,i] for i in range(V)], dim=4) # [B, C, T, H, V*W]
            y = batch.pop("text")
            # ipdb.set_trace()
            
            #y[0] = 'Cloudy, No rain, ' + y[0]
            #y[0] = 'Rainy Heavily, ' + y[0]
            #y[0] = 'Sunny, ' + y[0]
            y[0] = 'Night, ' + y[0]
            print(y)
            
            with torch.no_grad():
                # Prepare visual inputs
                encoded_views = []
                V = x.shape[1]
                for i in range(V):
                    encoded = vae.encode(x[:, i])  # Encode each view
                    encoded_views.append(encoded)  # Store the result
                x = torch.cat(encoded_views, dim=4)  # [B, C, T, H/P, V * W/P]
                # Prepare text inputs
                model_args = text_encoder.encode(y)
            for k, value in batch.items():
                if k not in ["bbox", "hdmap", "traj", "img_path"]:
                    model_args[k] = value.to(device, dtype)
                elif k == "bbox":
                    box_conditions = value.to(device, dtype)
                    model_args["bbox"] = box_conditions
                elif k == "hdmap":
                    map_conditions = value.to(device, dtype)
                    model_args["hdmap"] = map_conditions
                elif k == "traj":
                    map_conditions = value.to(device, dtype)
                    model_args["traj"] = map_conditions
                elif k == 'img_path':
                    T = len(value)               # Number of time steps
                    V = len(value[0])            # Number of views
                    B = len(value[0][0])         # Number of batch items

                    # Reorganize the list from [T[V[B]]] to [B, T, V]
                    model_args["img_path"] = [[[
                        value[t][v][b] for v in range(V)
                    ] for t in range(T)] for b in range(B)]
                    

            # TODO: mask_strategy
            '''
            if cfg.mask_ratios is not None and cfg.get("mask_training", True):
                mask_generator = MaskGenerator(cfg.mask_ratios)
                mask = mask_generator.get_masks(x)
                model_args["x_mask"] = mask
            else:
                mask = None
            refs_x = [t for t in torch.unbind(x[:1], dim=0)]
            refs_x = [refs_x]
            mask_strategy=["0"]
            '''
            
            
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
            for k in range(1):
                sample_idx = old_sample_idx
                video_clips = []

                # 4.4. long video generation
                if cfg.whole_scene:
                    cfg.loop = math.floor(( T - 1 ) / 15)
                for loop_i in range(cfg.loop):
                    # 4.4 sample in hidden space
                    # batch_prompts = [prompt[loop_i] for prompt in batch_prompts_loops]

                    # 4.5. apply mask strategy
                    masks = None
                    # if cfg.reference_path is not None:
                    if loop_i > 0:
                        # video_clips[-1] - [B,C,T,H,W*V]
                        views = torch.split(video_clips[-1], video_clips[-1].shape[4] // V, dim=4)  
                        encoded_views = [vae.encode(view) for view in views]
                        ref_x = torch.cat(encoded_views, dim=4) # [B,C,T,H,6W]
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
                    
                    '''
                    if cfg.mask_ratios is not None and cfg.get("mask_training", True):
                        masks = apply_mask_strategy(z, refs_x, mask_strategy, loop_i)
                        model_args["x_mask"] = torch.ones_like(model_args["x_mask"][:1], dtype=torch.bool, device=x.device)
                        model_args["x_mask"][0][0] = 0
                    else:
                        masks = None
                    '''
                    if cfg.infer_mode == 't2v' and loop_i==0: # t2v模式下第一轮推理不加条件帧
                        # mask_strategy = [f"0,0,0,0,0,{cfg.edit_ratio}"]
                        # refs_x = []
                        pass
                    else:
                        if loop_i==0: # 编辑模式，第0次迭代对条件帧进行编辑，若要对视频进行整体编辑可以将条件帧帧数设为视频总帧数
                            mask_strategy = [f"0,0,0,0,{cfg.condition_frame_length},{cfg.edit_ratio}"]
                        masks = apply_mask_strategy(z, refs_x, mask_strategy, loop_i)
                                
                    
                    if cfg.model.get("enable_condition", False):

                        model_args["layout_conditions"] = batch["hdmap"][:1, :, loop_i * cfg.num_frames:(loop_i + 1) * cfg.num_frames].to(device, dtype) + \
                                                          batch["bbox"][:1, :, loop_i * cfg.num_frames:(loop_i + 1) * cfg.num_frames].to(device, dtype) 
                                                          
                        batch["traj"][:1, :, 0] = torch.zeros(batch["traj"][:1, :, 0].shape, dtype=batch["traj"][:1, :, 0].dtype, device=batch["traj"][:1, :, 0].device)     
                        model_args["traj_conditions"] = batch["traj"][:1, :, loop_i * cfg.num_frames:(loop_i + 1) * cfg.num_frames].to(device, dtype)
                        
                        
                        if loop_i > 0:
                            model_args["layout_conditions"] = batch["hdmap"][:1, :, loop_i * (cfg.num_frames - cfg.condition_frame_length):loop_i * (cfg.num_frames - cfg.condition_frame_length) + cfg.num_frames].to(device, dtype) + \
                                                              batch["bbox"][:1, :, loop_i * (cfg.num_frames - cfg.condition_frame_length):loop_i * (cfg.num_frames - cfg.condition_frame_length) + cfg.num_frames].to(device, dtype) # + \
                            
                            batch["traj"][:1, :, loop_i*(cfg.num_frames-cfg.condition_frame_length)] = torch.zeros(batch["traj"][:1, :, 0].shape, dtype=batch["traj"][:1, :, 0].dtype, device=batch["traj"][:1, :, 0].device) 
                            model_args["traj_conditions"] = batch["traj"][:1, :, loop_i * (cfg.num_frames - cfg.condition_frame_length):loop_i * (cfg.num_frames - cfg.condition_frame_length) + cfg.num_frames].to(device, dtype)
                        
                        if cfg.model.get("condition_downsample", False):
                            B, V, T, _, _, _ = model_args["layout_conditions"].shape
                            downsampled_layout = []
                            for i in range(V):
                                downsampled = rearrange(model_args["layout_conditions"][:,i], 'B T C H W -> (B T) C H W', B=B, T=T)
                                downsampled = F.interpolate(downsampled, scale_factor=cfg.model.scale_factor, mode='nearest')
                                downsampled = rearrange(downsampled, '(B T) C H W -> B T C H W', B=B, T=T)
                                downsampled_layout.append(downsampled)
                            model_args["layout_conditions"] = torch.stack(downsampled_layout, dim=1) # [B,V,T,C,H',W']
                            
                            
                            downsampled_traj = []
                            for i in range(V):
                                downsampled = rearrange(model_args["traj_conditions"][:,i], 'B T C H W -> (B T) C H W', B=B, T=T)
                                downsampled = F.interpolate(downsampled, scale_factor=cfg.model.scale_factor, mode='nearest')
                                downsampled = rearrange(downsampled, '(B T) C H W -> B T C H W', B=B, T=T)
                                downsampled_traj.append(downsampled)
                            model_args["traj_conditions"] = torch.stack(downsampled_traj, dim=1) # [B,V,T,C,H',W']
                                    
                        B, V, T, _, _, _ = model_args["layout_conditions"].shape
                        encoded_layout = []
                        for i in range(V):
                            encoded = vae.encode(model_args["layout_conditions"][:, i].permute(0,2,1,3,4))  # Encode each view
                            encoded_layout.append(encoded)  # Store the result
                        model_args["layout_conditions"] = torch.cat(encoded_layout, dim=4)
                        
                        encoded_traj = []
                        for i in range(V):
                            encoded = vae.encode(model_args["traj_conditions"][:, i].permute(0,2,1,3,4))  # Encode each view
                            encoded_traj.append(encoded)  # Store the result
                        model_args["traj_conditions"] = torch.cat(encoded_traj, dim=4)
                        

                    samples = scheduler.sample(
                        model,
                        text_encoder,
                        z=z,
                        z_0=z_0,
                        prompts=y[:1],
                        device=device,
                        additional_args=model_args,
                        progress=True,
                        mask=masks,
                        enable_condition=cfg.model.get("enable_condition", False)
                    ) # [B,C,T,H,W*V]
                    views = torch.split(samples, samples.shape[4] // V, dim=4)  # Split along width axis (dim=4)
                    decoded_views = [vae.decode(view.to(dtype)) for view in views]
                    samples = torch.cat(decoded_views, dim=4) # [B,C,T,H,W*V]
                    video_clips.append(samples)

                    # 4.7. save video
                    
                    if loop_i == cfg.loop - 1:
                        # if not use_dist or coordinator.is_master():
                            for idx in range(len(video_clips[0])): #  B
                                video_clips_i = [video_clips[0][idx]] + [
                                    video_clips[i][idx][:, cfg.condition_frame_length :] for i in range(1, cfg.loop)
                                ]
                                video = torch.cat(video_clips_i, dim=1)
                                # video = torch.cat([video, torch.zeros(3, x0[0].shape[1] - video.shape[1], video.shape[2], video.shape[3]).to(device, dtype)], dim=1)
                                save_path = os.path.join(save_dir, f"video", f"step_{step:08d}_rank_{dist.get_rank():02d}_video")
                                
                                if cfg.model.get("enable_condition", False):
                                    condition_images = batch["hdmap"].to(device, dtype)[0].detach() + \
                                                       batch["bbox"].to(device, dtype)[0].detach() + \
                                                       batch["traj"].to(device, dtype)[0].detach()     # [V,T,C,H,W]
                                    condition_images = torch.cat([condition_images[i] for i in range(V)], dim=3) # [T,C,H,W*V]
                                    condition_images = condition_images.permute(1, 0, 2, 3) # [C,T,H,W*V]
                                    cat_img = concatenate_tensors([x0[0], condition_images, video], dim=2)
                                    # cat_img = torch.cat([x0[0], condition_images, video], dim=2)
                                    save_sample(cat_img, fps=cfg.save_fps, save_path=save_path, cfg=cfg, divide=True, tag=(condition_images.shape[2], 2*condition_images.shape[2])) # cat
                                    save_sample(x0[0], fps=cfg.save_fps, save_path=save_path+"_gt",cfg=cfg)
                                    save_sample(video, fps=cfg.save_fps, save_path=save_path+"_pred", cfg=cfg)
                                    video_with_bbox = batch["bbox"].to(device, dtype)[0].detach()
                                    video_with_bbox = torch.cat([video_with_bbox[i] for i in range(V)], dim=3)
                                    video_with_bbox = video_with_bbox.permute(1, 0, 2, 3)
                                    video_with_bbox = concatenate_tensors_add(video_with_bbox, video, dim=1) # [C,T,H,W*V]
                                    # video_with_bbox = video_with_bbox + video
                                    save_sample(video_with_bbox, fps=cfg.save_fps, save_path=save_path+"_pred_with_bbox", cfg=cfg)
                                    
                                    
                                    cat_img_with_box = concatenate_tensors([x0[0], condition_images, video_with_bbox], dim=2)
                                    # cat_img = torch.cat([x0[0], condition_images, video], dim=2)
                                    save_sample(cat_img_with_box, fps=cfg.save_fps, save_path=save_path+"_cat_pred_with_bbox", cfg=cfg, divide=True, tag=(condition_images.shape[2], 2*condition_images.shape[2])) # cat
                                    
                                    
                                    # save imgs
                                    if cfg.save_imgs:
                                        directory = os.path.join(save_dir, f"img")
                                        save_imgs(x0[0], 'gt', directory, cfg.camera_list, model_args['img_path'][0])
                                        save_imgs(condition_images, 'cond', directory, cfg.camera_list, model_args['img_path'][0])
                                        save_imgs(video, 'pred', directory, cfg.camera_list, model_args['img_path'][0])
                                        save_imgs(video, 'pred_with_idx', directory, cfg.camera_list, model_args['img_path'][0])
                                    
                                else:
                                    cat_img = concatenate_tensors([x0[0], video], dim=2)
                                    # cat_img = torch.cat([x0[0], video], dim=2)
                                    save_sample(cat_img, fps=cfg.save_fps, save_path=save_path, cfg=cfg) # cat
                                    save_sample(x0[0], fps=cfg.save_fps, save_path=save_path+"_gt", cfg=cfg)
                                    save_sample(video, fps=cfg.save_fps, save_path=save_path+"_pred", cfg=cfg)
                                    if cfg.save_imgs:
                                        directory = os.path.join(save_dir, f"img")
                                        save_imgs(x0[0], 'gt', directory, cfg.camera_list, model_args['img_path'][0])
                                        save_imgs(video, 'pred', directory, cfg.camera_list, model_args['img_path'][0])
                                    
                                print("frame: ", cat_img.shape[1])
                                with open(os.path.join(save_dir, f"video", f"step_{step:08d}_rank_{dist.get_rank():02d}_prompt.txt"), "w") as fp:
                                    fp.write(y[0])
                                sample_idx += 1

if __name__ == "__main__":
    main()
