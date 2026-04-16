# Wikitext103 experiments

## Requirements
- python 3.11
- install torch: `pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124`
- wandb==0.19.10
- timm==0.4.12
- download ImageNet-1K dataset
- Hardware: 4 GPUs, each at least 12 GB

## How to run
- define `CUDA_VISIBLE_DEVICES` in each script
- define `datapath` at each script
```
  bash run_pd.sh  # positive-definite scenario
  bash run_hd.sh  # intermediate scenario
  bash run_nd.sh  # negative-definite scenario
```
