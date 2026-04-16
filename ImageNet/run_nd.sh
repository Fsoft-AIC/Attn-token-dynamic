n_gpu=4
bz=256

export CUDA_VISIBLE_DEVICES=  # your cuda_ids
echo $CUDA_VISIBLE_DEVICES

export PD=1
export ANEG=1
export WNEG=0

datapath=  # your path to imagenet dataset

min=2050
max=9999
rand=$(od -An -N2 -tu2 /dev/urandom | tr -d ' ')
port=$(( rand % (max - min + 1) + min ))

job_name=deit_nd

python -m torch.distributed.launch --nproc_per_node=$n_gpu --master_port $port \
--use_env main.py --model deit_tiny_patch16_224 --batch-size $bz --data-path $datapath  \
--output_dir ./out/$job_name-$port
