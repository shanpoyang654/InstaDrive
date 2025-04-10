import setproctitle
setproctitle.setproctitle("sora")

from copy import deepcopy
from datetime import timedelta
from datetime import timedelta
from pprint import pprint

import torch
import torch.distributed as dist
import torch.nn.functional as F
from einops import rearrange
import wandb
from colossalai.booster import Booster
from colossalai.booster.plugin import LowLevelZeroPlugin
from colossalai.cluster import DistCoordinator
from colossalai.nn.optimizer import HybridAdam
from colossalai.utils import get_current_device, set_seed
from tqdm import tqdm

from opensora.acceleration.checkpoint import set_grad_checkpoint
from opensora.acceleration.parallel_states import (
    get_data_parallel_group,
    set_data_parallel_group,
    set_sequence_parallel_group,
)
from opensora.acceleration.plugin import ZeroSeqParallelPlugin
from opensora.datasets.dataloader import prepare_dataloader
from opensora.datasets.utils import  save_sample
from opensora.registry import DATASETS, MODELS, SCHEDULERS, build_module
from opensora.utils.ckpt_utils import load, model_gathering, model_sharding, record_model_param_shape, save
from opensora.utils.config_utils import (
    define_experiment_workspace,
    parse_configs,
    save_training_config,
)
from opensora.utils.misc import (
    Timer,
    all_reduce_mean,
    create_logger,
    create_tensorboard_writer,
    format_numel_str,
    get_model_numel,
    requires_grad,
    to_torch_dtype,
)
from opensora.utils.train_utils import MaskGenerator, create_colossalai_plugin, update_ema
from opensora.common import *

import time
import torch.nn.functional as F


# === mmdet3d ===
from opensora.datasets import NuScenesTDataset
from mmdet3d.datasets.pipelines import LoadMultiViewImageFromFiles, Collect3D
# === mmdet3d ===

'''
condition_frame_length 1
vae view-encoded
'''


from torchvision.utils import save_image


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
    

from torchvision.io import write_video
def save_sample_test(box_conditions, fps, save_path, flag=0):
    """
    Save the video from box_conditions tensor after normalizing and merging the view dimension into the width.
    
    Args:
        box_conditions (torch.Tensor): Tensor of shape [V, T, C, H, W], where V is the number of views.
        fps (int): Frames per second for the output video.
        save_path (str): Path to save the video.
    """
    save_path += ".mp4"
    # Input tensor shape: [V, C, T, H, W]
    if flag == 0:
        V = box_conditions.shape[0]
        box_conditions = torch.cat([box_conditions[i] for i in range(V)], dim=3) # [C,T,H,W*V]
    
    # Step 2: Normalize the tensor to [0, 1] if it's not already
    if box_conditions.min() < 0 or box_conditions.max() > 1:
        box_conditions = (box_conditions - box_conditions.min()) / (box_conditions.max() - box_conditions.min())
    
    
    box_conditions = box_conditions.permute(1, 2, 3, 0) # [T,H,W*V,C]
    
    # Step 4: Rescale from [0, 1] to [0, 255]
    box_conditions = (box_conditions * 255).byte().to("cpu") # [T,H,W,C]
    
    # Step 5: Save the video using write_video
    write_video(save_path, box_conditions, fps=fps, video_codec="h264", options={'crf': '0'}) # , video_codec="libx264", options={'crf': '0'})
    
    


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

def main():
    # ======================================================
    # 1. args & cfg
    # ======================================================
    cfg = parse_configs(training=True)
    dist.init_process_group(backend="nccl", timeout=timedelta(hours=24))
    torch.cuda.set_device(dist.get_rank() % torch.cuda.device_count())

    
    set_seed(cfg.get("seed", 1024))
    coordinator = DistCoordinator()
    print(cfg)
    if cfg.debug:
        cfg.outputs = cfg.outputs + "_debug"
        cfg.resume_inplace = False
    exp_name, exp_dir = define_experiment_workspace(cfg, coordinator.is_master())
    coordinator.block_all()

    if coordinator.is_master():
        os.makedirs(exp_dir, exist_ok=True)
        os.makedirs(os.path.join(exp_dir, "previews"), exist_ok=True)
        save_training_config(cfg.to_dict(), exp_dir)

    # ======================================================
    # 2. runtime variables & colossalai launch
    # ======================================================
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    assert cfg.dtype in ["fp16", "bf16"], f"Unknown mixed precision {cfg.dtype}"

    # 2.1. colossalai init distributed training
    # we set a very large timeout to avoid some processes exit early

    device = get_current_device()
    # == device and dtype ==
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    cfg_dtype = cfg.get("dtype", "bf16")
    assert cfg_dtype in ["fp16", "bf16"], f"Unknown mixed precision {cfg_dtype}"
    dtype = to_torch_dtype(cfg.get("dtype", "bf16"))

    # 2.2. init logger, tensorboard & wandb
    if not coordinator.is_master():
        logger = create_logger(None)
    else:
        print("Training configuration:")
        pprint(cfg._cfg_dict)
        logger = create_logger(exp_dir)
        logger.info(f"Experiment directory created at {exp_dir}")

        tb_writer = create_tensorboard_writer(exp_dir)
        if cfg.wandb:
            wandb.init(project="minisora", name=exp_name, config=cfg._cfg_dict)
        os.makedirs(os.path.join(exp_dir, "previews"), exist_ok=True)

    # h
    # 2.3. initialize ColossalAI booster
    if cfg.plugin == "zero2":
        plugin = LowLevelZeroPlugin(
            stage=2,
            precision=cfg.dtype,
            initial_scale=2**16,
            max_norm=cfg.grad_clip,
            #cpu_offload=True,
        )
        set_data_parallel_group(dist.group.WORLD)
    elif cfg.plugin == "zero2-seq":
        plugin = ZeroSeqParallelPlugin(
            sp_size=cfg.sp_size,
            stage=2,
            precision=cfg.dtype,
            initial_scale=2**16,
            max_norm=cfg.grad_clip,
            #cpu_offload=True
        )
        set_sequence_parallel_group(plugin.sp_group)
        set_data_parallel_group(plugin.dp_group)
    else:
        raise ValueError(f"Unknown plugin {cfg.plugin}")
    booster = Booster(plugin=plugin)

    # ======================================================
    # 3. build dataset and dataloader
    # ======================================================
    if cfg.get("dataset_factory", None): # will be deprecated
        train_dataset = create_instance(cfg.dataset_factory,
                                  # seq_len=cfg.num_frames,
                                  seq_len=cfg.get("test_num_frames", cfg.num_frames),
                                  image_size=cfg.image_size,
                                  training_seq_len=cfg.num_frames,
                                  version=cfg.version,
                                  fps_stride_tuples=cfg.get("fps_stride_tuples", None),
                                  temporal_sampling_scheme=cfg.get("temporal_sampling_scheme", None),
                                  stage="train")
        val_dataset = None
    elif cfg.dataset_version == "NuScenes" and cfg.get("train_dataset", None):
        train_dataset = create_instance_from_config(cfg.train_dataset)
        val_dataset = None
    elif cfg.dataset_version == "NuScenes_interpolated_12Hz" and cfg.get("train_dataset", None):
        pipeline = [
            LoadMultiViewImageFromFiles(camera_list=cfg.camera_list),
            Collect3D(
                keys=["img", 'description'],
                meta_lis_keys=["filename"]
            )
        ]
        train_dataset_args = cfg.train_dataset.copy()
        train_dataset_args.pop('_class_name')
        train_dataset_args.pop('transform_name')
        train_dataset = NuScenesTDataset(**train_dataset_args,
                                        pipeline=pipeline)
        val_dataset = None
    else:
        train_dataset = DatasetFromCSV(
            cfg.data_path,
            transform=(
                get_transforms_video(cfg.image_size[0])
                if not cfg.use_image_transform
                else get_transforms_image(cfg.image_size[0])
            ),
            num_frames=cfg.num_frames,
            frame_interval=cfg.frame_interval,
            root=cfg.root,
        )
        val_dataset = None

    
    train_dataloader, sampler_to_io= prepare_dataloader(
        train_dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        seed=cfg.get("seed", 42),
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        process_group=get_data_parallel_group(),
    )
    

    # ======================================================
    # 4. build model
    # ======================================================
    # 4.1. build model

    logger.info("Building models...")
    # == build text-encoder and vae ==
    text_encoder = build_module(cfg.get("text_encoder", None), MODELS, device=device, dtype=dtype)
    if text_encoder is not None:
        text_encoder_output_dim = text_encoder.output_dim
        text_encoder_model_max_length = text_encoder.model_max_length    
    else:
        text_encoder_output_dim = cfg.get("text_encoder_output_dim", 4096)
        text_encoder_model_max_length = cfg.get("text_encoder_model_max_length", 300)

    # == build vae ==
    vae = build_module(cfg.get("vae", None), MODELS)
    if vae is not None:
        vae = vae.to(device, dtype).eval()
    if vae is not None:
        input_size = (cfg.num_frames, *cfg.full_size)
        latent_size = vae.get_latent_size(input_size)
        vae_out_channels = vae.out_channels
    else:
        latent_size = (None, None, None)
        vae_out_channels = cfg.get("vae_out_channels", 4)
    
    # == build diffusion model ==
    model = (
        build_module(
            cfg.model,
            MODELS,
            input_size=latent_size,
            in_channels=vae_out_channels,
            caption_channels=text_encoder_output_dim,
            model_max_length=text_encoder_model_max_length,
            enable_sequence_parallelism=cfg.get("sp_size", 1) > 1,
        )
        .to(device, dtype)
        .train()
    )
    model_numel, model_numel_trainable = get_model_numel(model)
    text_encoder.y_embedder = model.y_embedder
    
    
    
    logger.info(
        f"Trainable model params: {format_numel_str(model_numel_trainable)}, Total model params: {format_numel_str(model_numel)}"
    )
    # == build ema for diffusion model ==
    ema = deepcopy(model).to(torch.float32).to(device)
    requires_grad(ema, False)
    ema_shape_dict = record_model_param_shape(ema)
    ema.eval()
    update_ema(ema, model, decay=0, sharded=False)


    # 4.4. build scheduler
    scheduler = build_module(cfg.scheduler, SCHEDULERS)

    # 4.5. setup optimizer
    optimizer = HybridAdam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.lr,
        weight_decay=0,
        adamw_mode=True,
    )
    warmup_steps = cfg.get("warmup_steps", None)

    if warmup_steps is None:
        lr_scheduler = None
    else:
        lr_scheduler = LinearWarmupLR(optimizer, warmup_steps=cfg.get("warmup_steps"))

    # 4.6. prepare for training
    if cfg.grad_checkpoint:
        set_grad_checkpoint(model)
    #model.train()
    if cfg.mask_ratios is not None:
        mask_generator = MaskGenerator(cfg.mask_ratios)

    # =======================================================
    # 5. boost model for distributed training with colossalai
    # =======================================================
    torch.set_default_dtype(dtype)
    model, optimizer, _, train_dataloader, lr_scheduler = booster.boost(
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        dataloader=train_dataloader,
    )
    torch.set_default_dtype(torch.float)
    logger.info("Boost model for distributed training")
    if cfg.train_dataset._class_name == "VariableVideoTextDataset":
        num_steps_per_epoch = train_dataloader.batch_sampler.get_num_batch() // dist.get_world_size()
    else:
        num_steps_per_epoch = len(train_dataloader)

    # =======================================================
    # 6. training loop
    # =======================================================
    start_epoch = start_step = log_step = sampler_start_idx = acc_step = 0
    running_loss = 0.0
    #sampler_to_io = train_dataloader.batch_sampler if cfg.train_dataset._class_name == "VariableVideoTextDataset" else None
    # == resume ==
    if cfg.get("load", None) is not None:
        logger.info("Loading resumed checkpoint")
        ret = load(
            booster,
            cfg.load,
            model=model,
            ema=ema,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            sampler=None if cfg.get("start_from_scratch", False) else sampler_to_io,
        )
        if not cfg.get("start_from_scratch", False):
            start_epoch, start_step = ret
        logger.info("Loaded checkpoint %s at epoch %s step %s", cfg.load, start_epoch, start_step)

    logger.info(f"Training for {cfg.epochs} epochs with {num_steps_per_epoch} steps per epoch")

    train_dataloader.sampler.reset()
    model_sharding(ema)

    # 6.2. training loop
    
    dist.barrier()
    
    for epoch in range(start_epoch, cfg.epochs):
        #train_dataloader.sampler.set_epoch(epoch)
        sampler_to_io.set_epoch(epoch)
        dataloader_iter = iter(train_dataloader)
        logger.info(f"Beginning epoch {epoch}...")
        with tqdm(
            enumerate(dataloader_iter, start=start_step),
            desc=f"Epoch {epoch}",
            disable=not coordinator.is_master(),
            initial=start_step,
            total=num_steps_per_epoch,
        ) as pbar:
            for step, batch in pbar:
                x = batch.pop("video").to(device, dtype)  # [B, V, C, T, H, W]
                # x = torch.cat([x[:,i] for i in range(V)], dim=4) # [B, C, T, H, V*W]
                x0 = x.clone()
                x = x0[:,:,:,: cfg.num_frames]
                V = x0.shape[1]
                x0 = torch.cat([x0[:,i] for i in range(V)], dim=4) # [B, C, T, H, V*W]
                y = batch.pop("text")
                # Visual and text encoding
                global_step = epoch * num_steps_per_epoch + step
                log_step += 1    
                          

                
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
                        # V = box_conditions.shape[1]
                        # box_conditions = torch.cat([box_conditions[:,i] for i in range(V)], dim=4) # [B, T, C, H, V*W]
                        model_args["bbox"] = box_conditions
                        
                        
                    elif k == "hdmap":
                        map_conditions = value.to(device, dtype)
                        # V = map_conditions.shape[1]
                        # map_conditions = torch.cat([map_conditions[:,i] for i in range(V)], dim=4) # [B, T, C, H, V*W]
                        model_args["hdmap"] = map_conditions
                    elif k == "traj":
                        traj_conditions = value.to(device, dtype)
                        # V = map_conditions.shape[1]
                        # map_conditions = torch.cat([map_conditions[:,i] for i in range(V)], dim=4) # [B, T, C, H, V*W]
                        model_args["traj"] = traj_conditions # [B, V, T, C, H, W]
                    elif k == 'img_path':
                        
                        T = len(value)               # Number of time steps
                        V = len(value[0])            # Number of views
                        B = len(value[0][0])         # Number of batch items

                        # Reorganize the list from [T[V[B]]] to [B, T, V]
                        model_args["img_path"] = [[[
                            value[t][v][b] for v in range(V)
                        ] for t in range(T)] for b in range(B)]

                # Mask   
                

                if cfg.mask_ratios is not None and cfg.get("mask_training", True):
                # if cfg.mask_ratios is not None:
                    mask = mask_generator.get_masks(x)
                    model_args["x_mask"] = mask # [B,T]
                else:
                    mask = None
                    
                
                # Video info
                with torch.no_grad():
                    if not cfg.debug and (global_step == 0 or global_step in [2 ** n for n in range(4, 8)] or global_step % cfg.get("test_every", 1000) == 0):
                    # if not cfg.debug and (global_step ==0 or global_step % 2 == 0):

                        start_eval = time.time()
                        model.eval()  
                        
                        refs_x =[]
                        condition_frame_length = cfg.condition_frame_length
                        x0_1 = x0[:1]
                        
                        refs_x = [t for t in torch.unbind(x[:1], dim=0)] # [[C, T, H/P, W/P]]

                    
                        refs_x = [refs_x] # [[[C, T, H/P, W/P]]]
                        mask_strategy=["0"]
                        
                        video_clips =[]
                        loop = cfg.loop
                        for loop_i in range(loop):
                            #define z
                            masks = None
                            #v1-1: refs_x is list of [B, ref_num, C, T, H, W]
                            # here B=1
                            if loop_i > 0:
                                # red x is [B, C, T, H, W]
                                # video_clips[-1] - [B,C,T,H,W*V]
                                views = torch.split(video_clips[-1], video_clips[-1].shape[4] // V, dim=4)  
                                encoded_views = [vae.encode(view) for view in views]
                                ref_x = torch.cat(encoded_views, dim=4) # [B,C,T,H,6W]
                                
                                # ref_x  = vae.encode(video_clips[-1])
                                for j, refs in enumerate(refs_x):
                                    # refs is [ref_num, C, T, H, W]
                                    if refs is None:
                                        refs_x[j] = [ref_x[j]]
                                    # refs_x is list of [B, ref_num, C, T, H, W]
                                    else:
                                        refs.append(ref_x[j])
                                
                                    if mask_strategy[j] is None:
                                        mask_strategy[j] = ""
                                    else:
                                        mask_strategy[j] += ";"
                                    mask_strategy[
                                        j
                                    ] += f"{loop_i},{len(refs)-1},-{condition_frame_length},0,{condition_frame_length}"

                            
                            z = torch.randn(1, vae.out_channels, *latent_size, device=device) # [1, C, T, H/P, W/P*view]
                            z_0 = z.clone()
                            if cfg.mask_ratios is not None and cfg.get("mask_training", True):
                                #mask_strategy = ["0"]
                                masks = apply_mask_strategy(z, refs_x, mask_strategy, loop_i)
                                model_args["x_mask"] = torch.ones_like(model_args["x_mask"][:1], dtype=torch.bool, device=x.device)
                                model_args["x_mask"][0][0] = 0
                            else:
                                masks = None
                            
                            
                            if cfg.model.get("enable_condition", False):
                                # downsampling + encode 
                                model_args["layout_conditions"] = model_args["hdmap"][:1,:,loop_i*cfg.num_frames:(loop_i+1)*cfg.num_frames].to(device, dtype)+\
                                    model_args["bbox"][:1,:,loop_i*cfg.num_frames:(loop_i+1)*cfg.num_frames].to(device, dtype)
                                
                                
                                # TODO: traj, first-frame(1/16) set as 0 / every iter
                                model_args["traj"][:1, :, 0] = torch.zeros(model_args["traj"][:1, :, 0].shape, dtype=model_args["traj"][:1, :, 0].dtype, device=model_args["traj"][:1, :, 0].device) #  model_args["traj"]: [B, V, T, C, H, W]
                                model_args["traj_conditions"] = model_args["traj"][:1,:,loop_i*cfg.num_frames:(loop_i+1)*cfg.num_frames].to(device, dtype)# [B,V,T,C,H,W]
                       
                        
                                if loop_i>0:
                                    model_args["layout_conditions"] = model_args["hdmap"][:1,:,loop_i*(cfg.num_frames-condition_frame_length):loop_i*(cfg.num_frames-condition_frame_length)+cfg.num_frames].to(device, dtype)+\
                                        model_args["bbox"][:1,:,loop_i*(cfg.num_frames-condition_frame_length):loop_i*(cfg.num_frames-condition_frame_length)+cfg.num_frames].to(device, dtype)
                                    
                                    model_args["traj"][:1, :, loop_i*(cfg.num_frames-condition_frame_length)] = torch.zeros(model_args["traj"][:1, :, 0].shape, dtype=model_args["traj"][:1, :, 0].dtype, device=model_args["traj"][:1, :, 0].device) #  model_args["traj"]: [B, V, T, C, H, W]
                                    model_args["traj_conditions"] = model_args["traj"][:1,:,loop_i*(cfg.num_frames-condition_frame_length):loop_i*(cfg.num_frames-condition_frame_length)+cfg.num_frames].to(device, dtype)
                                    
                                    
                                if cfg.model.get("condition_downsample", False):
                                    B, V, T, _, _, _ = model_args["layout_conditions"].shape
                                    downsampled_layout = []
                                    for i in range(V):
                                        downsampled = rearrange(model_args["layout_conditions"][:,i], 'B T C H W -> (B T) C H W', B=B, T=T)
                                        downsampled = F.interpolate(downsampled, scale_factor=cfg.model.scale_factor, mode='nearest')
                                        downsampled = rearrange(downsampled, '(B T) C H W -> B T C H W', B=B, T=T)
                                        downsampled_layout.append(downsampled)
                                    model_args["layout_conditions"] = torch.stack(downsampled_layout, dim=1) # [B,V,T,C,H',W']
                                    
                                    # traj
                                    downsampled_traj = []
                                    for i in range(V):
                                        downsampled = rearrange(model_args["traj_conditions"][:,i], 'B T C H W -> (B T) C H W', B=B, T=T)
                                        downsampled = F.interpolate(downsampled, scale_factor=cfg.model.scale_factor, mode='nearest')
                                        downsampled = rearrange(downsampled, '(B T) C H W -> B T C H W', B=B, T=T)
                                        downsampled_traj.append(downsampled)
                                    model_args["traj_conditions"] = torch.stack(downsampled_traj, dim=1) # [B,V,T,C,H',W']
                                            
                                with torch.no_grad():
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
                                z = z,
                                z_0 = z_0,
                                prompts = y[:1],
                                device = device,
                                additional_args = model_args,
                                progress = True,
                                mask = masks,
                                enable_condition=cfg.model.get("enable_condition", False)
                            )   # [B,C,T,H,W*V]
                            
                            
                            views = torch.split(samples, samples.shape[4] // V, dim=4)  # Split along width axis (dim=4)
                            decoded_views = [vae.decode(view.to(dtype), num_frames=cfg.num_frames) for view in views]
                            sample = torch.cat(decoded_views, dim=4) # [B,C,T,H,W*V]

                            video_clips.append(sample)
                            
                            if loop_i==0:
                                sample_0 = sample[0]                            
                            if cfg.get("mask_training", False):

                                if loop_i==loop-1:
                                    if coordinator.is_master():
                                        for idx in range(len(video_clips[0])):
                                            
                                            video_clips_i = [video_clips[0][idx]] + [
                                                video_clips[i][idx][:, condition_frame_length:] for i in range(1, loop)
                                            ]
                                            
                                            video = torch.cat(video_clips_i, dim=1)
                                            video = torch.cat([video,torch.zeros(3,x0[0].shape[1]-video.shape[1],video.shape[2],video.shape[3]).to(device,dtype)],dim=1)
                                            # video = torch.cat(video_clips_i, dim=1)
                                            
                                            
                                            save_path = os.path.join(exp_dir, "previews", f"step_{global_step:08d}_sample_idx_{idx}_loop")
                                            if cfg.model.get("enable_condition", False):
                                                condition_images = model_args["hdmap"].to(device, dtype)[0].detach()+\
                                                                        model_args["bbox"].to(device, dtype)[0].detach()+\
                                                                        model_args["traj"].to(device, dtype)[0].detach() # [V,T,C,H,W]
                                                condition_images = torch.cat([condition_images[i] for i in range(V)], dim=3) # [T,C,H,W*view]
                                                condition_images = condition_images.permute(1,0,2,3) # [C,T,H,6W]
                                                
                                                # cat_img = torch.cat([x0[0], condition_images,video], dim=2)
                                                cat_img = concatenate_tensors([x0[0], condition_images, video], dim=2)
                                                save_sample(cat_img, fps=cfg.save_fps, save_path=save_path, cfg=cfg, divide=True, tag=(condition_images.shape[2],2*condition_images.shape[2]))
                                                with open(os.path.join(exp_dir, "previews", f"step_{global_step:08d}_rank_{dist.get_rank():02d}_prompt.txt"), "w") as fp:
                                                    fp.write(y[0])
                                            else:
                                                # cat_img = torch.cat([x0[0],video], dim=2)
                                                cat_img = concatenate_tensors([x0[0], video], dim=2)
                                                save_sample(cat_img, fps=cfg.save_fps, save_path=save_path, cfg=cfg)
                                                with open(os.path.join(exp_dir, "previews", f"step_{global_step:08d}_rank_{dist.get_rank():02d}_prompt.txt"), "w") as fp:
                                                    fp.write(y[0])
                            else:
                                break
                        
                        
                        # rec = vae.decode(x[:1],num_frames=cfg.num_frames)[0] # [C,T,H,6W]
                        views = torch.split(x[:1], x[:1].shape[4] // V, dim=4)  # Split along width axis (dim=4) x[:1]:[B,C,T,H,6W]
                        decoded_views = [vae.decode(view.to(dtype), num_frames=cfg.num_frames) for view in views]
                        rec = torch.cat(decoded_views, dim=4)[0] # [C,T,H,6W]
                        
                        
                        if coordinator.is_master():

                            if cfg.model.get("enable_condition", False):                        
                                condition_images = model_args["hdmap"][:,:,:cfg.num_frames].to(device, dtype)[0].detach()+\
                                    model_args["bbox"][:,:,:cfg.num_frames].to(device, dtype)[0].detach()+\
                                    model_args["traj"][:,:,:cfg.num_frames].to(device, dtype)[0].detach()# [V,T,C,H,W]
                                condition_images = torch.cat([condition_images[i] for i in range(V)], dim=3) # [T,C,H,W*view]
                                condition_images = condition_images.permute(1,0,2,3) # [C,T,H,6W]
                                # b x c x t x h x w
                                cat_img = torch.cat([x0[0], rec, condition_images, sample_0], dim=2)
                                save_sample(cat_img, fps=cfg.save_fps, save_path=os.path.join(exp_dir, "previews", f"step_{global_step:08d}_cat"), cfg=cfg, divide=True, tag=(2*condition_images.shape[2],3*condition_images.shape[2]))
                                
                                
                                # save imgs
                                if cfg.save_imgs:
                                    directory = os.path.join(exp_dir, f"img")
                                    save_imgs(x0[0], 'gt', directory, cfg.camera_list, model_args['img_path'][0])
                                    save_imgs(rec, 'rec', directory, cfg.camera_list, model_args['img_path'][0])
                                    save_imgs(condition_images, 'cond', directory, cfg.camera_list, model_args['img_path'][0])
                                    save_imgs(sample_0, 'pred', directory, cfg.camera_list, model_args['img_path'][0])

                            else:
                                cat_img = torch.cat([x0[0], rec, sample_0], dim=2)
                                save_sample(cat_img, fps=cfg.save_fps, save_path=os.path.join(exp_dir, "previews", f"step_{global_step:08d}_cat"), cfg=cfg)
                                # save imgs
                                if cfg.save_imgs:
                                    directory = os.path.join(exp_dir, f"img")
                                    save_imgs(x0[0], 'gt', directory, cfg.camera_list, model_args['img_path'][0])
                                    save_imgs(rec, 'rec', directory, cfg.camera_list, model_args['img_path'][0])
                                    save_imgs(sample_0, 'pred', directory, cfg.camera_list, model_args['img_path'][0])

                                
                        print(f"Sampling time: {time.time() - start_eval}")
                        torch.cuda.empty_cache()
                        model.train()
                if cfg.model.get("enable_condition", False):
                    model_args["layout_conditions"] = model_args["bbox"].to(device, dtype)[:,:,:cfg.num_frames]\
                        + model_args["hdmap"].to(device, dtype)[:,:,:cfg.num_frames]
                         
                    model_args["traj"][:1, :, 0] = torch.zeros(model_args["traj"][:1, :, 0].shape, dtype=model_args["traj"][:1, :, 0].dtype, device=model_args["traj"][:1, :, 0].device)          
                    model_args["traj_conditions"] = model_args["traj"].to(device, dtype)[:,:,:cfg.num_frames]  
                    
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
                        
                        
                    with torch.no_grad():
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
                                           

                # Diffusion
                #t = torch.randint(0, scheduler.num_timesteps, (x.shape[0],), device=device)
                loss_dict = scheduler.training_losses(model, x, model_args, mask=mask)

                # Backward & update
                loss = loss_dict["loss"].mean()
                torch.cuda.empty_cache()
                booster.backward(loss=loss, optimizer=optimizer)
                optimizer.step()
                optimizer.zero_grad()

                # Update EMA
                update_ema(ema, model.module, optimizer=optimizer)

                # Log loss values:
                all_reduce_mean(loss)
                running_loss += loss.item()
                acc_step += 1

                # Log to tensorboard
                # == logging ==
                if coordinator.is_master() and (global_step + 1) % cfg.get("log_every", 1) == 0:
                    avg_loss = running_loss / log_step
                    # progress bar
                    pbar.set_postfix({"loss": avg_loss, "step": step, "global_step": global_step})
                    # tensorboard
                    tb_writer.add_scalar("loss", loss.item(), global_step)
                    # wandb
                    if cfg.get("wandb", False):
                        wandb_dict = {
                            "iter": global_step,
                            "acc_step": acc_step,
                            "epoch": epoch,
                            "loss": loss.item(),
                            "avg_loss": avg_loss,
                            "lr": optimizer.param_groups[0]["lr"],
                        }
                        wandb.log(wandb_dict, step=global_step)


                # == checkpoint saving ==
                ckpt_every = cfg.get("ckpt_every", 0)
                if ckpt_every > 0 and (global_step + 1) % ckpt_every == 0:
                    model_gathering(ema, ema_shape_dict)
                    save_dir = save(
                        booster,
                        exp_dir,
                        model=model,
                        ema=ema,
                        optimizer=optimizer,
                        lr_scheduler=lr_scheduler,
                        sampler=sampler_to_io,
                        epoch=epoch,
                        step=step + 1,
                        global_step=global_step + 1,
                        batch_size=cfg.get("batch_size", None),
                    )
                    if dist.get_rank() == 0:
                        model_sharding(ema)
                    logger.info(
                        "Saved checkpoint at epoch %s, step %s, global_step %s to %s",
                        epoch,
                        step + 1,
                        global_step + 1,
                        save_dir,
                    )

        # the continue epochs are not resumed, so we need to reset the sampler start index and start step
        if cfg.train_dataset._class_name == "VideoTextDataset":
            train_dataloader.sampler.reset()
        elif cfg.train_dataset._class_name == "VariableVideoTextDataset":
            train_dataloader.batch_sampler.set_epoch(epoch + 1)
            print("Epoch done, recomputing batch sampler")
        elif cfg.train_dataset._class_name == "configs.base.nuscenes.get_nuscenes" or cfg.train_dataset._class_name == "opensora.datasets.opendv.MotionOpenDVDataset" or cfg.train_dataset._class_name == "NuscenesTDataset":
            train_dataloader.sampler.reset()
        else:
            raise ValueError(f"Unknown train_dataset class name {cfg.train_dataset._class_name}")
        start_step = 0


if __name__ == "__main__":
    main()