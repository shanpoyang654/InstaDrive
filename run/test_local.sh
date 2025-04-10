config=$1
gpus=$2
pretrain=$3
DIR=/mnt/iag/user/yangzhuoran/local/stable_bbox_embed/stable_new/metavdt
cd $DIR

ENV_FOUNDATION="/mnt/iag/user/zhangxin/env/metavdt" 
ENV_FJ="/mnt/iag/user/guoxi/envs/metavdt"
ENV_FZ="/mnt/iag/user/zhangxin/env/metavdt"

if [ -d "$ENV_FJ" ]; then
    echo "Activating environment at $ENV_FJ"
    source activate $ENV_FJ
elif [ -d "$ENV_FOUNDATION" ]; then
    echo "Activating environment at $ENV_FOUNDATION"
    source activate $ENV_FOUNDATION
elif [ -d "$ENV_FZ" ]; then
    echo "Activating environment at $ENV_FZ"
    source activate $ENV_FZ
else
    echo "No valid environment found."
fi

export PYTHONPATH=${DIR}:${DIR}/third_party:$PYTHONPATH
OTHER_ARGS=${@:4}

torchrun --master_port=29506 --nnodes=1 --nproc_per_node=${gpus} scripts/inference-unify-multi.py \
${config} --ckpt-path ${pretrain}
