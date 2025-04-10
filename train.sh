export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# bash run/train_local.sh configs/metavdt/nuscenes_osv1_1_ti2v_baseline_2view_2dataloader_2model.py 1 /mnt/iag/user/guoxi/projects/metavdt/outputs/nuscenes_osv1_1_ti2v_far/012-STDiT2-XL-2/epoch165-global_step55000/model/model.safetensors


# /mnt/iag/user/yangzhuoran/local/stable/metavdt/outputs/nuscenes_osv1_1_ti2v_baseline_2view_2dataloader_2model/103-STDiT2-XL-2/epoch0-global_step48000/model/model.safetensors
# bash run/train_local.sh configs/metavdt/nuscenes_osv1_1_ti2v_baseline_2view_2dataloader_2model.py 1 /mnt/iag/user/yangzhuoran/local/stable/metavdt/outputs/nuscenes_osv1_1_ti2v_baseline_2view_2dataloader_2model/103-STDiT2-XL-2/epoch2-global_step48000/model/model.safetensors
# /mnt/iag/user/yangzhuoran/local/stable/metavdt/outputs/nuscenes_osv1_1_ti2v_baseline_2view_2dataloader_2model/099-STDiT2-XL-2/epoch2-global_step57000/model/model.safetensors # controlnet
# controlnet + traj
# bash run/train_local.sh configs/metavdt/nuscenes_osv1_1_ti2v_baseline_2view_2dataloader_2model.py 8 /mnt/iag/user/yangzhuoran/local/stable/metavdt/outputs/nuscenes_osv1_1_ti2v_baseline_2view_2dataloader_2model/099-STDiT2-XL-2/epoch2-global_step57000/model/model.safetensors
# dive 
bash run/train_local.sh configs/metavdt/nuscenes_osv1_1_ti2v_baseline_2view_2dataloader_2model.py 4 /mnt/iag/user/yangzhuoran/local/stable/metavdt/outputs/nuscenes_osv1_1_ti2v_baseline_2view_2dataloader_2model/103-STDiT2-XL-2/epoch2-global_step39000/model/model.safetensors


# /mnt/iag/user/yangzhuoran/local/stable/metavdt/outputs/nuscenes_osv1_1_ti2v_baseline_2view_2dataloader_2model/103-STDiT2-XL-2/epoch6-global_step132000/model/model.safetensors
# bash run/train_local.sh configs/metavdt/nuscenes_osv1_1_ti2v_baseline_2view_2dataloader_2model.py 8 /mnt/iag/user/yangzhuoran/local/stable/metavdt/outputs/nuscenes_osv1_1_ti2v_baseline_2view_2dataloader_2model/103-STDiT2-XL-2/epoch6-global_step132000/model/model.safetensors


# traj_dive_pos_one_pipeline
# /mnt/iag/user/yangzhuoran/local/stable/metavdt/outputs/nuscenes_osv1_1_ti2v_baseline_2view_2dataloader_2model/157-STDiT2-XL-2/epoch19-global_step24000/model/model.safetensors
# bash run/train_local.sh configs/metavdt/nuscenes_osv1_1_ti2v_baseline_2view_2dataloader_2model.py 8 /mnt/iag/user/yangzhuoran/local/stable/metavdt/outputs/nuscenes_osv1_1_ti2v_baseline_2view_2dataloader_2model/164-STDiT2-XL-2/epoch7-global_step10000/model/model.safetensors

# /mnt/iag/user/yangzhuoran/local/stable/metavdt/outputs/nuscenes_osv1_1_ti2v_baseline_2view_2dataloader_2model/162-STDiT2-XL-2/epoch3-global_step4000/model/model.safetensors
# bash run/train_local.sh configs/metavdt/nuscenes_osv1_1_ti2v_baseline_2view_2dataloader_2model.py 1 /mnt/iag/user/yangzhuoran/local/stable/metavdt/outputs/nuscenes_osv1_1_ti2v_baseline_2view_2dataloader_2model/162-STDiT2-XL-2/epoch3-global_step4000/model/model.safetensors


# bash run/train_local.sh configs/metavdt/nuscenes_osv1_1_ti2v_baseline_2view_2dataloader_2model.py 4 /mnt/iag/user/yangzhuoran/local/stable/metavdt/outputs/nuscenes_osv1_1_ti2v_baseline_2view_2dataloader_2model/051-STDiT2-XL-2/epoch1-global_step29000/model/model.safetensors
# /mnt/iag/user/yangzhuoran/local/stable/metavdt/outputs/nuscenes_osv1_1_ti2v_baseline/101-STDiT2-XL-2/epoch1-global_step31000/model/model.safetensors
# bash run/train_local.sh configs/metavdt/nuscenes_osv1_1_ti2v_baseline_2view_2dataloader_2model.py 8 /mnt/iag/user/yangzhuoran/local/stable/metavdt/outputs/nuscenes_osv1_1_ti2v_baseline_2view_2dataloader_2model/007-STDiT2-XL-2/epoch0-global_step7000/model/model.safetensors # bimot
# bash run/train_local.sh configs/metavdt/nuscenes_osv1_1_ti2v_baseline_2view_2dataloader_2model.py 8 /mnt/iag/user/yangzhuoran/local/stable/metavdt/outputs/nuscenes_osv1_1_ti2v_baseline_2view_2dataloader_2model/011-STDiT2-XL-2/epoch0-global_step2000/model/model.safetensors # controldit
# bash run/train_local.sh configs/metavdt/nuscenes_osv1_1_ti2v_baseline_2view_2dataloader_2model.py 4  /mnt/iag/user/guoxi/model_zoo/OpenSora-STDiT-v2-stage3/model.safetensors # controldit finetune
# bash run/train_local.sh configs/metavdt/nuscenes_osv1_1_ti2v_baseline_2view_2dataloader_2model.py 4  /mnt/iag/user/yangzhuoran/local/stable/metavdt/outputs/nuscenes_osv1_1_ti2v_baseline_2view_2dataloader_2model/010-STDiT2-XL-2/epoch0-global_step2000/model/model.safetensors # bimot finetune
# bash run/train_local.sh configs/metavdt/nuscenes_osv1_1_ti2v_baseline_2view_2dataloader_2model.py 1  /mnt/iag/user/yangzhuoran/local/stable/metavdt/outputs/nuscenes_osv1_1_ti2v_baseline_2view_2dataloader_2model/017-STDiT2-XL-2/epoch0-global_step4000/model/model.safetensors # 1view bimot 12Hz
# bash run/train_local.sh configs/metavdt/nuscenes_osv1_1_ti2v_baseline_2view_2dataloader_2model.py 1  /mnt/iag/user/guoxi/model_zoo/OpenSora-STDiT-v2-stage3/model.safetensors # 1view controldit 12Hz

