# Wikitext103 experiments

## Requirements
- python 3.9
- install pytorch: `pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121`
- wandb
- Wikitext103 dataset at ./data/wt103
- Hardware: 4 GPUs, each at least 12 GB

## How to run
- define `CUDA_VISIBLE_DEVICES` in each script
```
  bash run_pd.sh  # positive-definite scenario
  bash run_hd.sh  # intermediate scenario
  bash run_nd.sh  # negative-definite scenario

  bash run_rope_add.sh  # rope-add-diagonal-component
```
