#!/usr/bin/env bash

CONFIG=$1
WORKDIR=$2
GPUS=$3
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

echo 'CONFIG===========' $CONFIG
echo 'WORKDIR===========' $WORKDIR
echo 'GPUS===========' $GPUS
echo 'NNODES===========' $NNODES
echo 'PORT===========' $PORT
#echo '${@:4}======'${@:4}
#echo 'pytorch ${@:4}========' pytorch ${@:4}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/train.py \
    --config=$CONFIG \
    --work-dir=$WORKDIR \
    --seed 0 \
    --launcher pytorch ${@:4}
