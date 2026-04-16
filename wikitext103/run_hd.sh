#!/bin/bash

export CUDA_VISIBLE_DEVICES=  # your gpu ids

datapath=./data/wt103  # your path to wk103 dataset

export PD=1
export WNEG=0
export ANEG=0.5

echo 'Run training...'
python train.py \
    --cuda \
    --data $datapath \
    --dataset wt103 \
    --adaptive \
    --n_layer 16 \
    --d_model 410 \
    --n_head 10 \
    --d_head 41 \
    --d_inner 2100 \
    --dropout 0.05 \
    --dropatt 0.0 \
    --optim adam \
    --lr 0.00025 \
    --warmup_step 500 \
    --max_step 200000 \
    --tgt_len 150 \
    --mem_len 0 \
    --eval_tgt_len 156 \
    --batch_size 60 \
    --multi_gpu \
    --gpu0_bsz 30 \
    --scheduler cosine \
    --seed 8888 \
    --attn_type 2 \
    --work_dir ./out/hd
