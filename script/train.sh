#!/bin/bash


MASTER_ADDR=localhost
MASTER_PORT=29500
RANK=0  
nnodes=2
nproc_per_node=2

DISTRIBUTED_ARGS="
    --nproc_per_node $nproc_per_node \
    --nnodes $nnodes \
    --node_rank $RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"
torchrun \
    --nproc_per_node=$nproc_per_node \
    --nnodes=$nnodes \
    --node_rank=$RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    train_ddp.py 


