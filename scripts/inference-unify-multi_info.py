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
from opensora.datasets import NuScenesTDataset
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
def save_imgs(x, label_str, dir, camera_list, img_path, step, rank, normalize=True, value_range=(-1, 1)):
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
            save_path = os.path.join(label_dir, camera_name, str(step) + "_" + str(rank) + "_" + os.path.basename(img_file_path))
            
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
    
   

    # 4.1. batch generation
    dataloader_iter = iter(val_dataloader)
    with tqdm(
            enumerate(dataloader_iter, 0),
            # disable=not coordinator.is_master(),
            total=len(val_dataloader)) as pbar:
        for step, batch in pbar:
            x = batch.pop("video").to(device, dtype)  # [B, V, C, T, H, W]
            x0 = x.clone()
            x = x0[:, :, : cfg.num_frames]
            V = x0.shape[1]
            x0 = torch.cat([x0[:,i] for i in range(V)], dim=4) # [B, C, T, H, V*W]
            y = batch.pop("text")
            
            
            
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
                    
      

            old_sample_idx = sample_idx
            # generate multiple samples for each prompt
            for k in range(1):
                sample_idx = old_sample_idx
                video_clips = []

                # 4.4. long video generation
                for loop_i in range(cfg.loop):
   
                    # 4.7. save video
                    
                    if loop_i == cfg.loop - 1:
                        # if not use_dist or coordinator.is_master():
                            for idx in range(1): #  B
                                
                                # video = torch.cat([video, torch.zeros(3, x0[0].shape[1] - video.shape[1], video.shape[2], video.shape[3]).to(device, dtype)], dim=1)
                                save_path = os.path.join(save_dir, f"video", f"step_{step:08d}_rank_{dist.get_rank():02d}_video")
                                
                                if cfg.model.get("enable_condition", False):
                                    save_sample(x0[0], fps=cfg.save_fps, save_path=save_path+"_gt",cfg=cfg)
                                    
                               
                                    # 保存到 txt 文件中
                                    output_file = save_path + "img_paths.txt"
                                    img_paths = model_args['img_path'][0]
                                    with open(output_file, "w") as file:
                                        # 将所有时间步的数据展平成一行，并用双引号包裹每个路径
                                        flattened_paths = [f'"{item.split("/")[-1]}"' for sublist in img_paths for item in sublist]
                                        file.write(",".join(flattened_paths))
                                
                                    
                                # print("frame: ", cat_img.shape[1])
                                with open(os.path.join(save_dir, f"video", f"step_{step:08d}_rank_{dist.get_rank():02d}_prompt.txt"), "w") as fp:
                                    fp.write(y[0])
                                sample_idx += 1

if __name__ == "__main__":
    main()
