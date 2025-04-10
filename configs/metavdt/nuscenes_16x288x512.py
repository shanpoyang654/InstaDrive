# from configs.base.nuscenes import get_nuscenes
# from functools import partial
exp_name = "nuscenes_16x288x512"
import os
debug = False
num_frames = 16
frame_interval = 1 # deprecated
data_path = None # deprecated
image_size = (288, 512)
fps_stride_tuples = [(10, 0.4)]
version = "trainval"

diffusion_steps = 100
if debug:
    version = "mini" # mini or trainval
    diffusion_steps=50

# Define dataset
# dataset_factory = partial(get_nuscenes, seq_len=num_frames, image_size=image_size)
dataset_factory = "configs.base.nuscenes.get_nuscenes"
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
    num_sampling_steps=100,
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
save_fps = 5
test_every = 100
prompt_path = "./assets/texts/t2v_samples.txt"
save_dir = "./outputs/nuscenes_16x288x512/samples/"