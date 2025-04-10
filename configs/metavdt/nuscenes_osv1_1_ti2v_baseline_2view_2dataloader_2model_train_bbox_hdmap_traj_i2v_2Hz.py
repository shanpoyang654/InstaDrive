# from configs.base.nuscenes import get_nuscenes
# from functools import partial
exp_name = "nuscenes_osv1_1_ti2v_baseline_2view_2dataloader_2model"
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

fps_stride_tuples = [(10, 0.4)] # for MotionDataset
num_frames = 16
test_num_frames = 16
# image_size = (288, 512)
image_size = (256, 448)
fps = 2

version = "trainval"

camera_list = ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_RIGHT", "CAM_BACK", "CAM_BACK_LEFT"]
# camera_list = ["CAM_FRONT"]
full_size = (image_size[0], image_size[1] * len(camera_list)) # MULTI_VIEW






annotation_version = {"bbox": 'class', # 'cross' 'surface' 'none'
                      "hdmap": 'with_divider',# 'old':['drivable_area', 'road_segment', 'lane', 'ped_crossing', 'walkway', 'stop_line']
                                     # 'with_divider':['drivable_area', 'road_segment', 'road_block', 'lane', 'ped_crossing', 'walkway', 'stop_line', 'carpark_area', 'road_divider', 'lane_divider']
                      "traj": 'world-final'} # 'world' 'ego' 'pixel'
# additional_image_annotations
# Define paths based on annotation version
if annotation_version["bbox"] == 'cross':
    bbox_path = '/mnt/iag/user/yangzhuoran/dataset/data/3dbox_head'
elif annotation_version["bbox"] == 'class':
    bbox_path = '/mnt/iag/user/yangzhuoran/dataset/data/3dbox_class' # cross加粗 + 全类别
elif annotation_version["bbox"] == 'surface':
    bbox_path = '/mnt/iag/user/yangzhuoran/dataset/data/3dbox_head_surface'
elif annotation_version["bbox"] == 'none':
    bbox_path = '/mnt/iag/user/yangzhuoran/dataset/data/3dbox_test'  

if annotation_version["hdmap"] == 'old':
    hdmap_path = '/mnt/iag/user/yangzhuoran/dataset/data/hdmap_test'
elif annotation_version["hdmap"] == 'with_divider':
    hdmap_path = '/mnt/iag/user/yangzhuoran/dataset/data/hdmap_with_divider' 
    

if annotation_version["traj"] == 'ego':
    traj_path = '/mnt/iag/user/yangzhuoran/dataset/data/traj_ego'
elif annotation_version["traj"] == 'world':
    traj_path = '/mnt/iag/user/yangzhuoran/dataset/data/traj_test'
elif annotation_version["traj"] == 'world25':
    traj_path = '/mnt/iag/user/yangzhuoran/dataset/data/traj_world_25'
elif annotation_version["traj"] == 'world-final':
    traj_path = '/mnt/iag/user/yangzhuoran/dataset/data/traj_world_final'
elif annotation_version["traj"] == 'pixel':
    traj_path = '/mnt/iag/user/yangzhuoran/dataset/data/traj_pixel'
elif annotation_version["traj"] == 'pixel_changergd_range80':
    traj_path = '/mnt/iag/user/yangzhuoran/dataset/data/traj_pixel_changergb'
# traj_pixel_changergb

# Combine selected paths into a list of annotations
additional_image_annotations = [
    {'bbox': bbox_path} if bbox_path else {},
    {'hdmap': hdmap_path} if hdmap_path else {},
    {'traj': traj_path} if traj_path else {},
]



infer_mode = 'i2v' # TODO: no first frame  't2v' 'i2v'
# mask_training = False # no first frame
loop = 1 # TODO:
dataset_version = 'NuScenes_interpolated_12Hz'
if dataset_version == 'NuScenes':
    # === MotionDataset ===
    train_dataset = dict(
        _class_name="configs.base.nuscenes.get_nuscenes",
        seq_len=test_num_frames,
        fps_stride_tuples=fps_stride_tuples,
        version=version,
        training_seq_len=num_frames,
        image_size=image_size,
        full_size = full_size,
        frame_interval=1,
        split="train",
        stage="train",
        additional_image_annotations =[
            "bbox",
            "hdmap"
        ],
        camera_list = camera_list,
        metadata_dir = '/mnt/iag/user/yangzhuoran/local/stable/metavdt/metadata_' + str(len(camera_list))
    )

    val_dataset = dict(
        _class_name="configs.base.nuscenes.get_nuscenes",
        # temporal_sampling_scheme=temporal_sampling_scheme,
        fps_stride_tuples=fps_stride_tuples,
        version=version,
        seq_len=test_num_frames,
        image_size=image_size,
        full_size = full_size,
        frame_interval=1,
        split="mini_val",
        stage="test",
        additional_image_annotations =[
            "bbox",
            "hdmap"
        ],
        camera_list = camera_list,
        metadata_dir = '/mnt/iag/user/yangzhuoran/local/stable/metavdt/metadata_' + str(len(camera_list))
    )
    # === MotionDataset ===


elif dataset_version == 'NuScenes_interpolated_12Hz':
    # === NuScenesTDataset ===
    train_dataset = dict(
        _class_name="NuscenesTDataset",
        # ann_file='/mnt/iag/user/yangzhuoran/dataset/data/nuscenes_mmdet3d-12Hz/nuscenes_interp_12Hz_infos_train.pkl',
        ann_file='/mnt/iag/user/yangzhuoran/dataset/data/nuscenes_mmdet3d_2/nuscenes_infos_train.pkl',
        #pipeline=pipeline,
        step=num_frames*loop,
        modality={
            "use_lidar": False,
            "use_camera": True,
            "use_radar": False,
            "use_map": False,
            "use_external": False
        },
        start_on_firstframe=False,
        start_on_keyframe=False,
        video_length = num_frames*loop,
        image_size = image_size,
        full_size = full_size,
        enable_scene_description = True,
        fps = 2,
        additional_image_annotations = additional_image_annotations,
        annotation={"hdmap":True,
                    "bbox":True,
                    "traj":True},
    )

    val_dataset = dict(
        _class_name="NuscenesTDataset",
        # ann_file='/mnt/iag/user/yangzhuoran/dataset/data/nuscenes_mmdet3d-12Hz/nuscenes_interp_12Hz_infos_val.pkl',
        ann_file='/mnt/iag/user/yangzhuoran/dataset/data/nuscenes_mmdet3d_2/nuscenes_infos_val.pkl',
        #pipeline=pipeline,
        step=num_frames*loop, # TODO:
        modality={
            "use_lidar": False,
            "use_camera": True,
            "use_radar": False,
            "use_map": False,
            "use_external": False
        },
        start_on_firstframe=False,
        start_on_keyframe=False,
        video_length = num_frames*loop,
        image_size = image_size,
        full_size = full_size,
        enable_scene_description = True,
        fps = 2,
        additional_image_annotations = additional_image_annotations,
        annotation={"hdmap":True,
                    "bbox":True,
                    "traj":True},

    )
    # === NuScenesTDataset ===
else:
    train_dataset = None
    val_dataset = None

use_image_transform = False


# Define acceleration
num_workers = 8 # reduce memory consumption 
num_bucket_build_workers = 16
dtype = "bf16"
grad_checkpoint = True
plugin = "zero2"
sp_size = 1

# Define model
model = dict(
    type="STDiT2-XL/2",
    architecture='ControlDiT',  # 'Non-Control' 'ControlDiT' 'BiMoT' 'BiMoT-VAE'
    from_pretrained=None,
    input_sq_size=512,  # pretrained model is trained on 512x512
    qk_norm=True,
    enable_flashattn=True,
    enable_layernorm_kernel=False,
    enable_condition = True,
    freeze=None, # 'blocks'
    use_separate_scales=True, # pos_emb
    cyclic_pos_emb=False, # pos_emb
    condition_downsample=False,
    scale_factor=0.25,
    controlnet='traj_dive_pos_one_pipeline', # 'dive' 'controlnet' 'sd'
    duplicated_depth=14,
    duplicated_same=False,
    # depth=28,
    task=2,
)
vae = dict(
    type="VideoAutoencoderKL",
    from_pretrained="/mnt/iag/user/guoxi/model_zoo/pretrained_models/stabilityai/sd-vae-ft-ema",
    micro_batch_size=4,
    local_files_only=True,
    freeze_vae_2d=True,
)
text_encoder = dict(
    type="t5",
    from_pretrained="/mnt/iag/user/guoxi/model_zoo/pretrained_models/t5_ckpts/t5-v1_1-xxl",
    model_max_length=200,
    # shardformer=True, # freeze
    freeze_t5=True,
)
scheduler = dict(
    type="iddpm",
    num_sampling_steps=num_sampling_steps,
    timestep_respacing=None,
)

# scheduler = dict(
#     type="rflow",
#     use_timestep_transform=False,
#     sample_method="logit-normal",
# )

mask_ratios = {
    "random": 0.005,
    "intepolate": 0.002,
    "quarter_random": 0.007,
    "quarter_head": 0.002,
    "quarter_tail": 0.002,
    "quarter_head_tail": 0.002,
    "image_random": 0.0,
    "image_head": 0.22,
    "image_tail": 0.005,
    "image_head_tail": 0.005,
}

test_mask_mode = ["image_head"]


# Others
seed = 42
outputs = "outputs/" + exp_name
wandb = False

epochs = 1000
log_every = 100
ckpt_every = 1000
load = None
batch_size = 1
lr = 2e-5
grad_clip = 1.0

# for test and inference
save_fps = 10
test_every = 1000
prompt_path = ""
save_dir = "outputs/" + "infer/" + exp_name 
condition_frame_length = 1
multi_resolution = 'STDiT2'

save_imgs = True
edit_ratio = 0.0