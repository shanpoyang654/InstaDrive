import setproctitle
setproctitle.setproctitle("sora")

from copy import deepcopy
from datetime import timedelta
from datetime import timedelta
from pprint import pprint

import torch
import torch.distributed as dist
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
        logger.info("Loading checkpoint")
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
                view = x.shape[1]
                x = torch.cat([x[:,i] for i in range(view)], dim=4) # [B, C, T, H, view*W]
                y = batch.pop("text")
                # Visual and text encoding
                global_step = epoch * num_steps_per_epoch + step
                log_step += 1                

                
                with torch.no_grad():
                    # Prepare visual inputs
                    x = vae.encode(x)  # [B, C, T, H/P, W/P]
                    # Prepare text inputs
                    model_args = text_encoder.encode(y)
                
                # Mask     
                if cfg.mask_ratios is not None:
                    mask = mask_generator.get_masks(x)
                    model_args["x_mask"] = mask # [B,T]
                else:
                    mask = None
                    
                # Video info
                for k, v in batch.items():
                    if k not in ["bbox", "hdmap"]:
                        model_args[k] = v.to(device, dtype) 
                    elif k == "bbox":
                        box_conditions = v.to(device, dtype)
                        view = box_conditions.shape[1]
                        box_conditions = torch.cat([box_conditions[:,i] for i in range(view)], dim=4) # [B, T, C, H, view*W]
                        # box_conditions = torch.cat([box_conditions[:, 0], box_conditions[:, 1], box_conditions[:, 2], box_conditions[:, 3], box_conditions[:, 4], box_conditions[:, 5]], dim=4)  # [B, V, T, C, H, W] -> [B, T, C, H, 6W]
                        model_args["bbox"] = box_conditions
                    elif k == "hdmap":
                        map_conditions = v.to(device, dtype)
                        view = map_conditions.shape[1]
                        map_conditions = torch.cat([map_conditions[:,i] for i in range(view)], dim=4) # [B, T, C, H, view*W]
                        # map_conditions = torch.cat([map_conditions[:, 0], map_conditions[:, 1], map_conditions[:, 2], map_conditions[:, 3], map_conditions[:, 4], map_conditions[:, 5]], dim=4)  # [B, V, T, C, H, W] -> [B, T, C, H, 6W]
                        model_args["hdmap"] = map_conditions

                # === encode layout conditions ===
                with torch.no_grad():
                    if cfg.model.get("enable_condition", False):
                        encoded_box = vae.encode(model_args["bbox"].permute(0,2,1,3,4)) # [B,c,t,h,6w]
                        encoded_map = vae.encode(model_args["hdmap"].permute(0,2,1,3,4)) # [B,c,t,h,6w]
                        model_args["layout_conditions"] = encoded_box + encoded_map
                        # model_args["layout_conditions"] = vae.encode((model_args["box_conditions"]+model_args["map_conditions"]).permute(0,2,1,3,4)) # [B,c,t,h,6w]
                # === encode layout conditions ===

                with torch.no_grad():
                    if not cfg.debug and (global_step == 0 or global_step in [2 ** n for n in range(4, 8)] or global_step % cfg.get("test_every", 1000) == 0):
                    # if not cfg.debug and (global_step ==0 or global_step % 2 == 0):

                        start_eval = time.time()
                        model.eval()  
                        
                        samples = scheduler.sample(
                            model,
                            text_encoder,
                            z_size = (
                                vae.out_channels, *vae.get_latent_size(
                                    (
                                        cfg.get("num_frames", cfg.num_frames), 
                                        *cfg.full_size # revise to [H, 6W]
                                    ) 
                                )
                            ), # [C',T',H',6W']
                            prompts = y[:1], 
                            device = device,
                            additional_args = model_args,
                            progress = True,
                            # mask = masks, # None
                            enable_condition=cfg.model.get("enable_condition", False)
                        ) # [1, C', T, H/P, 6W/P]
                        sample = vae.decode(samples.to(dtype),num_frames=cfg.num_frames) # [1, C, T, H, 6W]
                        
                
                        rec = vae.decode(x[:1],num_frames=cfg.num_frames)
                        
                        # save sample
                        if coordinator.is_master():
                            with open(os.path.join(exp_dir, "previews", f"step_{global_step:08d}_multiview_prompt.txt"), 'w') as fp:
                                fp.write(y[0] + '\n')
                            condition_images = model_args["hdmap"][0].detach()+model_args["bbox"][0].detach() # [T,C,H,6W]
                            condition_images = condition_images.permute(1,0,2,3) # [C,T,H,6W]
                            cat = torch.cat([rec[0], condition_images, sample[0]], dim=2) # [C,T,3H,6W]
                            save_path = os.path.join(exp_dir, "previews", f"step_{global_step:08d}_layout")
                            save_sample(cat, fps=cfg.save_fps, save_path=save_path, normalize=True)
                                
                        print(f"Sampling time: {time.time() - start_eval}")
                        torch.cuda.empty_cache()
                        model.train()
                        
                            

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