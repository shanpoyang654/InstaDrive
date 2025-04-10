num_frames = 16
frame_interval = 3
fps = 24
image_size = (240, 426)
multi_resolution = "STDiT2"

# Define model
model = dict(
    type="STDiT2-XL/2",
    from_pretrained="/mnt/iag/user/zhangxin/worldsim/code/metavdt_devV2/pretrained_models/OpenSora-STDiT-v2-stage3/model.safetensors",
    input_sq_size=512,
    qk_norm=True,
    enable_flash_attn=True,
    enable_layernorm_kernel=False,
    enable_condition = False,
)
vae = dict(
    type="VideoAutoencoderKL",
    from_pretrained="/mnt/iag/user/zhangxin/worldsim/code/metavdt_devV2/pretrained_models/stabilityai/sd-vae-ft-ema",
    micro_batch_size=4,
    local_files_only=True,
)
text_encoder = dict(
    type="t5",
    from_pretrained="/mnt/iag/user/zhangxin/worldsim/code/metavdt_devV2/pretrained_models/t5_ckpts/t5-v1_1-xxl",
    model_max_length=200,
    shardformer=False,

)
scheduler = dict(
    type="iddpm",
    num_sampling_steps=100,
    cfg_scale=7.0,
    cfg_channel=3,  # or None
)
dtype = "bf16"

# Condition
#prompt_path = "/mnt/iag/user/zhangxin/worldsim/code/metavdt_devV2/assets/texts/t2v_samples.txt"
prompt = None  # prompt has higher priority than prompt_path
prompt=["A soaring drone footage captures the majestic beauty of a coastal cliff, its red and yellow stratified rock faces rich in color and against the vibrant turquoise of the sea. Seabirds can be seen taking flight around the cliff's precipices. As the drone slowly moves from different angles, the changing sunlight casts shifting shadows that highlight the rugged textures of the cliff and the surrounding calm sea. The water gently laps at the rock base and the greenery that clings to the top of the cliff, and the scene gives a sense of peaceful isolation at the fringes of the ocean. The video captures the essence of pristine natural beauty untouched by human structures."]
# Others
batch_size = 1
seed = 42
save_dir = "/mnt/iag/user/zhangxin/worldsim/code/metavdt_devV2/samples/samples/"
loop = 4

num_sample=1

condition_frame_length =4
