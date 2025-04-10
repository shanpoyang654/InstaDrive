# from configs.base.nuscenes import get_nuscenes
# from functools import partial
exp_name = "nuscenes_osv1_1_t64_16x288x512_learn_tpe"
debug = False
resume_inplace = False
frame_interval = 1
num_sampling_steps = 100
if debug:
    version = "mini"  # mini or trainval
    num_sampling_steps = 50

# Define dataset
# dataset_factory = partial(get_nuscenes, seq_len=num_frames, image_size=image_size)
# dataset_factory = "configs.base.nuscenes.get_nuscenes"

fps_stride_tuples = [(10, 0.4)]
num_frames = 16
test_num_frames = 16
image_size = (288, 512)
full_size = (288, 512 * 6)
version = "trainval"

dataset_version = 'NuScenes_interpolated'
if dataset_version == 'NuScenes':
    # === MotionDataset ===
    train_dataset = dict(
        _class_name="configs.base.nuscenes.get_nuscenes",
        seq_len=test_num_frames,
        fps_stride_tuples=fps_stride_tuples,
        version=version,
        training_seq_len=num_frames,
        image_size=image_size,
        frame_interval=1,
        split="train",
        stage="train",
        additional_image_annotations =[
            "bbox",
            "hdmap"
        ],
        camera_list =["CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_RIGHT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_FRONT_LEFT"]
    )

    val_dataset = dict(
        _class_name="configs.base.nuscenes.get_nuscenes",
        # temporal_sampling_scheme=temporal_sampling_scheme,
        fps_stride_tuples=fps_stride_tuples,
        version=version,
        seq_len=test_num_frames,
        image_size=image_size,
        frame_interval=1,
        split="mini_val",
        stage="test",
        additional_image_annotations =[
            "bbox",
            "hdmap"
        ],
        camera_list =["CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_RIGHT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_FRONT_LEFT"], 
    )
    # === MotionDataset ===
    
elif dataset_version == 'NuScenes_interpolated_12Hz':
    # === NuScenesTDataset ===
    train_dataset = dict(
        _class_name="NuscenesTDataset",
        ann_file='/mnt/iag/user/yangzhuoran/dataset/data/nuscenes_mmdet3d-12Hz/nuscenes_interp_12Hz_infos_train.pkl',
        #pipeline=pipeline,
        modality={
            "use_lidar": False,
            "use_camera": True,
            "use_radar": False,
            "use_map": False,
            "use_external": False
        },
        start_on_firstframe=False,
        start_on_keyframe=False,
        video_length = num_frames,
        image_size = image_size,
        full_size = full_size,
        enable_scene_description = True,
        fps = 12,
        additional_image_annotations = [{'bbox': '/mnt/iag/user/yangzhuoran/dataset/data/3dbox_test'},
                                        {'hdmap': '/mnt/iag/user/yangzhuoran/dataset/data/hdmap_test'}],
    )

    val_dataset = dict(
        _class_name="NuscenesTDataset",
        ann_file='/mnt/iag/user/yangzhuoran/dataset/data/nuscenes_mmdet3d-12Hz/nuscenes_interp_12Hz_infos_val.pkl',
        #pipeline=pipeline,
        modality={
            "use_lidar": False,
            "use_camera": True,
            "use_radar": False,
            "use_map": False,
            "use_external": False
        },
        start_on_firstframe=False,
        start_on_keyframe=False,
        video_length = num_frames,
        image_size = image_size,
        full_size = full_size,
        enable_scene_description = True,
        fps = 12,
        additional_image_annotations = [{'bbox': '/mnt/iag/user/yangzhuoran/dataset/data/3dbox_test'},
                                        {'hdmap': '/mnt/iag/user/yangzhuoran/dataset/data/hdmap_test'}],
    
    )
    # === NuScenesTDataset ===
else:
    train_dataset = None
    val_dataset = None




use_image_transform = False


# Define acceleration
num_workers = 0
num_bucket_build_workers = 16
dtype = "bf16"
grad_checkpoint = True
plugin = "zero2"
sp_size = 1

# Define model
model = dict(
    type="STDiT2-XL/2",
    from_pretrained=None,
    input_sq_size=512,  # pretrained model is trained on 512x512
    qk_norm=True,
    enable_flashattn=True,
    enable_layernorm_kernel=False,
)
vae = dict(
    type="VideoAutoencoderKL",
    from_pretrained="/mnt/iag/user/guoxi/model_zoo/pretrained_models/stabilityai/sd-vae-ft-ema", 
    micro_batch_size=4,
    local_files_only=True,
)
text_encoder = dict(
    type="t5",
    from_pretrained="/mnt/iag/user/guoxi/model_zoo/pretrained_models/t5_ckpts/t5-v1_1-xxl",
    model_max_length=200,
    shardformer=False,
)
scheduler = dict(
    type="iddpm",
    timestep_respacing="",
)

# Others
seed = 42
outputs = "outputs/" + exp_name
wandb = False

epochs = 1000
log_every = 10
ckpt_every = 50
load = None

batch_size = 4
lr = 2e-5
grad_clip = 1.0

# for test and inference
save_fps = 10
test_every = 1000
prompt_path = ""
save_dir = ""
