config=$1
gpus=$2
pretrain=$3
DIR=/mnt/iag/user/zhangxin/worldsim/code/metavdt_devV2
cd $DIR

ENV_FOUNDATION="/mnt/iag/user/zhangxin/env/metavdt"
ENV_FJ="/mnt/iag/user/guoxi/envs/metavdt"

if [ -d "$ENV_FOUNDATION" ]; then
    echo "Activating environment at $ENV_FOUNDATION"
    source activate $ENV_FOUNDATION
elif [ -d "$ENV_FJ" ]; then
    echo "Activating environment at $ENV_FJ"
    source activate $ENV_FJ
else
    echo "No valid environment found."
fi

export PYTHONPATH=${DIR}:$PYTHONPATH
OTHER_ARGS=${@:4}
#rm -r  ~/.cache/colossalai/torch_extensions/


torchrun --nnodes=1 --nproc_per_node=${gpus}  scripts/train.py ${config} --ckpt-path ${pretrain} ${OTHER_ARGS} #--resume_inplace 