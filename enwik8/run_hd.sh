export CUDA_VISIBLE_DEVICES=  # your gpu id
export JOB_NAME=hd

export LR=0.0006
export PD=1
export ANEG=0.5
export WNEG=0

python train_enwik8.py
