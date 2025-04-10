# from configs.base.nuscenes import get_nuscenes
# from functools import partial
exp_name = "nuscenes_t128_16x288x512_seq_sampling"
debug = False
resume_inplace = False
frame_interval = 1  # deprecated
data_path = None  # deprecated
num_sampling_steps = 100
if debug:
    version = "mini"  # mini or trainval
    num_sampling_steps = 50

# Define dataset
# dataset_factory = partial(get_nuscenes, seq_len=num_frames, image_size=image_sizme)
# dataset_factory = "configs.base.nuscenes.get_nuscenes"
temporal_sampling_scheme = {f"stride_{frame_stride}": 0.1 for frame_stride in range(1, 9)}
temporal_sampling_scheme['random'] = 0.2
fps_stride_tuples = [(10, 0.4)]
num_frames = 16
test_num_frames = 128
image_size = (288, 512)
version = "trainval"
train_dataset = dict(
    _class_name="configs.base.nuscenes.get_nuscenes",
    seq_len=test_num_frames,
    temporal_sampling_scheme=temporal_sampling_scheme,
    fps_stride_tuples=fps_stride_tuples,
    version=version,
    training_seq_len=num_frames,
    image_size=image_size,
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
    split="mini_val",
    stage="test",
)
use_image_transform = False
num_workers = 4

# Define acceleration
dtype = "bf16"
grad_checkpoint = True
plugin = "zero2"
sp_size = 1

# Define model
model = dict(
    type="STDiT-XL/2",
    space_scale=1.0,
    time_scale=1.0,
    from_pretrained="PixArt-XL-2-512x512.pth",
    enable_flashattn=True,
    enable_layernorm_kernel=False,
    test_num_frames=test_num_frames
)
vae = dict(
    type="VideoAutoencoderKL",
    from_pretrained="stabilityai/sd-vae-ft-ema",
)
text_encoder = dict(
    type="t5",
    from_pretrained="./pretrained_models/t5_ckpts",
    model_max_length=120,
    shardformer=False,
)
scheduler = dict(
    type="iddpm",
    # timestep_respacing="",
    num_sampling_steps=num_sampling_steps,
    # diffusion_steps=diffusion_steps
)

# Others
seed = 42
outputs = "outputs/" + exp_name
wandb = False

epochs = 1000
log_every = 10
ckpt_every = 1000
load = None

batch_size = 8
lr = 2e-5
grad_clip = 1.0

# for test and inference
save_fps = 10
test_every = 500
prompt_path = "./assets/texts/t2v_samples.txt"
save_dir = "./outputs/samples/"
