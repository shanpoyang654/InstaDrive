# from configs.base.nuscenes import get_nuscenes
# from functools import partial
exp_name = "nuscenes_128x288x512"
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
num_frames = 128
test_num_frames = 128
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
    stage="train"
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
    type="STDiT2-XL/2",
    from_pretrained=None,
    input_sq_size=512,  # pretrained model is trained on 512x512
    qk_norm=True,
    enable_flashattn=True,
    enable_layernorm_kernel=False,
)
vae = dict(
    type="VideoAutoencoderKL",
    from_pretrained="stabilityai/sd-vae-ft-ema",
    micro_batch_size=4,
    local_files_only=True,
)
text_encoder = dict(
    type="t5",
    from_pretrained="./pretrained_models/t5_ckpts/t5-v1_1-xxl",
    model_max_length=200,
    shardformer=False,
)
scheduler = dict(
    type="iddpm",
    timestep_respacing="",
)

# mask_ratios = {
#     "mask_no": 0.75,
#     "mask_quarter_random": 0.025,
#     "mask_quarter_head": 0.025,
#     "mask_quarter_tail": 0.025,
#     "mask_quarter_head_tail": 0.05,
#     "mask_image_random": 0.025,
#     "mask_image_head": 0.025,
#     "mask_image_tail": 0.025,
#     "mask_image_head_tail": 0.05,
# }
# Others
seed = 42
outputs = "/mnt/iag/user/wangchiyu/gitprojects/metavdt/outputs/" + exp_name
wandb = False

epochs = 1000
log_every = 10
ckpt_every = 1000
load = None

batch_size = 1
lr = 2e-5
grad_clip = 1.0

# for test and inference
save_fps = 8
test_every = 500
prompt_path = ""
save_dir = ""
