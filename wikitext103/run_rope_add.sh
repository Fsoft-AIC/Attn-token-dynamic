
export CUDA_VISIBLE_DEVICES=  # your gpu ids

echo cuda: $CUDA_VISIBLE_DEVICES

export ADD=0.05

datapath=./data/wt103  # your path to wk103 dataset


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
    --d_rot 16 \
    --optim adam \
    --lr 0.00025 \
    --warmup_step 1000 \
    --max_step 200000 \
    --tgt_len 150 \
    --mem_len 0 \
    --eval_tgt_len 156 \
    --multi_gpu \
    --batch_size 60 \
    --gpu0_bsz 30 \
    --scheduler cosine \
    --attn_type 305 \
    --work_dir ./out/rope_add
