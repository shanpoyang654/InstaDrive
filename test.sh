export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# origin
# bash run/test_local.sh configs/metavdt/nuscenes_osv1_1_ti2v_baseline_2view_2dataloader_2model.py 1 /mnt/iag/user/guoxi/model_zoo/OpenSora-STDiT-v2-stage3/model.safetensors
# dive
# bash run/test_local.sh configs/metavdt/nuscenes_osv1_1_ti2v_baseline_2view_2dataloader_2model_infer.py 1 /mnt/iag/user/yangzhuoran/local/stable/metavdt/outputs/nuscenes_osv1_1_ti2v_baseline_2view_2dataloader_2model/157-STDiT2-XL-2/epoch19-global_step24000/model/model.safetensors

# dive
# /mnt/iag/user/yangzhuoran/local/stable/metavdt/outputs/nuscenes_osv1_1_ti2v_baseline_2view_2dataloader_2model/103-STDiT2-XL-2/epoch2-global_step39000/model/model.safetensors
bash run/test_local.sh configs/metavdt/nuscenes_osv1_1_ti2v_baseline_2view_2dataloader_2model_infer.py 8 /mnt/iag/user/yangzhuoran/local/stable/metavdt/outputs/nuscenes_osv1_1_ti2v_baseline_2view_2dataloader_2model/103-STDiT2-XL-2/epoch2-global_step39000/model/model.safetensors


# traj_dive_pos_one_pipeline
# /mnt/iag/user/yangzhuoran/local/stable/metavdt/outputs/nuscenes_osv1_1_ti2v_baseline_2view_2dataloader_2model/157-STDiT2-XL-2/epoch8-global_step11000/model/model.safetensors
# bash run/test_local.sh configs/metavdt/nuscenes_osv1_1_ti2v_baseline_2view_2dataloader_2model_infer.py 8 /mnt/iag/user/yangzhuoran/local/stable/metavdt/outputs/nuscenes_osv1_1_ti2v_baseline_2view_2dataloader_2model/157-STDiT2-XL-2/epoch19-global_step24000/model/model.safetensors
# /mnt/iag/user/yangzhuoran/local/stable/metavdt/outputs/nuscenes_osv1_1_ti2v_baseline_2view_2dataloader_2model/162-STDiT2-XL-2/epoch3-global_step4000/model/model.safetensors
# bash run/test_local.sh configs/metavdt/nuscenes_osv1_1_ti2v_baseline_2view_2dataloader_2model_infer.py 8 /mnt/iag/user/yangzhuoran/local/stable/metavdt/outputs/nuscenes_osv1_1_ti2v_baseline_2view_2dataloader_2model/162-STDiT2-XL-2/epoch3-global_step4000/model/model.safetensors


# /mnt/iag/user/yangzhuoran/local/stable/metavdt/outputs/nuscenes_osv1_1_ti2v_baseline_2view_2dataloader_2model/162-STDiT2-XL-2/epoch3-global_step4000/model/model.safetensors
# bash run/test_local.sh configs/metavdt/nuscenes_osv1_1_ti2v_baseline_2view_2dataloader_2model_infer.py 8 /mnt/iag/user/yangzhuoran/local/stable/metavdt/outputs/nuscenes_osv1_1_ti2v_baseline_2view_2dataloader_2model/174-STDiT2-XL-2/epoch13-global_step17000/model/model.safetensors

