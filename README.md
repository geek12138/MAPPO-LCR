# MAPPO-LCR
MAPPO-LCR: Multi-Agent Proximal Policy Optimization with Local Cooperation Reward in Spatial Public Goods Games 


## Requirements
It is worth mentioning that because python runs slowly, we use cuda library to improve the speed of code running.

```
* Python Version 3.12.2
* CUDA Version: 12.8
* torch Version: 2.2.1
* numpy Version: 1.26.4
* pandas Version: 2.2.3
```

## Installation
```bash
conda env create -f environment.yaml
```

## Usage
```bash
sh run_one_MAPPO_LCR.sh
```

## Citation

If you use our codebase or models in your research, please cite this work.

```
@article{YANG2026117948,
title = {MAPPO-LCR: Multi-Agent Proximal Policy Optimization with Local Cooperation Reward in spatial public goods games},
journal = {Chaos, Solitons & Fractals},
volume = {206},
pages = {117948},
year = {2026},
issn = {0960-0779},
doi = {https://doi.org/10.1016/j.chaos.2026.117948},
url = {https://www.sciencedirect.com/science/article/pii/S0960077926000895},
author = {Zhaoqilin Yang and Axin Xiang and Kedi Yang and Tianjun Liu and Youliang Tian}
}
```

Thanks https://github.com/Tychema/Learning-And-Propagation
