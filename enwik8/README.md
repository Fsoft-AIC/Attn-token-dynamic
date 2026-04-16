# Wikitext103 experiments

## Requirements
- python 3.9
- install pytorch: `pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121`
- install packages: `pip install packaging==25.0 loguru==0.7.3 einx==0.3.0 tqdm==4.67.1`
- enwik8 dataset as enwik8.gz at ./data
- Hardware: 1 GPU, at least 16 GB

## How to run
- define CUDA_VISIBLE_DEVICES at each script
- run:
```
  bash run_pd.sh  # positive-definite scenario
  bash run_hd.sh  # intermediate scenario
  bash run_nd.sh  # negative-definite scenario
```
