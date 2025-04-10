# from configs.base.nuscenes import get_nuscenes
# from functools import partial
exp_name = "nuscenes_t64_16x288x512_learn_tpe"
debug = False
resume_inplace = False
frame_interval = 1
num_sampling_steps = 100
if debug:
    version = "mini"  # mini or trainval
    num_sampling_steps = 50

# Define dataset
# dataset_factory = partial(get_nuscenes, seq_len=num_frames, image_size=image_sizme)
# dataset_factory = "configs.base.nuscenes.get_nuscenes"

fps_stride_tuples = [(10, 0.4)]
num_frames = 16
test_num_frames = 16
image_size = (288, 512)
version = "trainval"



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
        "3dbox",
        "hdmap"
    ],
    camera_list =["CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_RIGHT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_FRONT_LEFT"],
    metadata_dir ="/mnt/iag/user/zhangxin/worldsim/code/metavdt_dev/meta_data"
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
    metadata_dir ="/mnt/iag/user/zhangxin/worldsim/code/metavdt_dev/meta_data"
)
use_image_transform = False


# Define acceleration
num_workers = 4
num_bucket_build_workers = 16
dtype = "bf16"
grad_checkpoint = True
plugin = "zero2"
sp_size = 1

# Define model
model = dict(
    type="STDiT3-XL/2",
    from_pretrained=None,
    input_sq_size=512,  # pretrained model is trained on 512x512
    qk_norm=True,
    enable_flash_attn=True,
    enable_layernorm_kernel=False,
    freeze_y_embedder=True,
    enable_condition = True
)


vae = dict(
    type="OpenSoraVAE_V1_2",
    from_pretrained="/mnt/iag/user/zhangxin/model_zoo/OpenSora-VAE-v1.2/model.safetensors",
    micro_frame_size=17,
    micro_batch_size=4,
    #local_files_only=True,
)
text_encoder = dict(
    type="t5",
    from_pretrained="./pretrained_models/t5_ckpts/t5-v1_1-xxl",#/mnt/iag/user/guoxi/model_zoo/pretrained_models/t5_ckpts/t5-v1_1-xxl
    model_max_length=300,
    shardformer=False,
)
'''
scheduler = dict(
    type="iddpm",
    timestep_respacing="",
)
'''
scheduler = dict(
    type="rflow",
    use_timestep_transform=False,
    sample_method="logit-normal",
)
'''
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
'''

# Others
seed = 42
outputs = "outputs/" + exp_name
wandb = False

epochs = 1000
log_every = 10
ckpt_every = 1000
load = None

batch_size =8
lr = 2e-5
grad_clip = 1.0

# for test and inference
save_fps = 5
test_every = 50
prompt_path = ""
save_dir = ""

mask_training = False
