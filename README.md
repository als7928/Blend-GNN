# IJCAI-2025 Submission - Local-Global Blending Graph Neural ODE Network for Graph Classification

## Setup
```
$ conda create -n blend python=3.10
$ conda activate blend
$ conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia
$ pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-1.13.1+cu116.html
$ pip install -r requirements.txt
```
---
## Run
```
$ python main.py --dataset $dataset$ --epochs $epochs$ --hidden_dim $d_k$ --time $T$  --step_size $tau$
```
