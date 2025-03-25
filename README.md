
<h1> FACTR: Force-Attending Curriculum Training for Contact-Rich Policy Learning </h1>



#### [Jason Jingzhou Liu](https://jasonjzliu.com)<sup>\*</sup>, [Yulong Li](https://yulongli42.github.io)<sup>\*</sup>, [Kenneth Shaw](https://kennyshaw.net), [Tony Tao](https://tony-tao.com), [Ruslan Salakhutdinov](https://www.cs.cmu.edu/~rsalakhu/), [Deepak Pathak](https://www.cs.cmu.edu/~dpathak/)
_Carnegie Mellon University_

[Project Page](https://jasonjzliu.com/factr/) | [arXiV](https://arxiv.org/abs/2502.17432)

<h1> </h1>
<img src="assets/main_teaser.jpg" alt="teaser" width="750"/>

<br>

## Catalog
- [Environment](#environment)
- [Dataset](#dataset)
- [Training and Testing](#training-and-testing)
- [Acknowledgements](#acknowledgements)
- [Citation](#citation)
  
## Environment
```bash
conda env create -f env.yaml
conda activate factr
```
We make use of the pretrained features from [data4robotics](https://github.com/SudeepDasari/data4robotics) project. Download the pretrained features as follows:
```bash
bash scripts/download_features
```
## Dataset
Collect and process data.
```bash

```

## Training

To train a BC policy, the following script gives an example of the script and command-line arguments. The default parameters are set in [cfg/train_bc.yaml]

```bash
bash scripts/train_bc.sh
```
There are several important configs to set up to train your own policy:
- setup a task configuration file under [cfg/task](cfg/task) which specifies the observation dimensions, action dimensions, and camera indices for visual inputs
- path to the dataset
- the curriculum parameteters



## Policy Rollout
```bash

```

## Citation
If you find this codebase useful, feel free to cite our work!
<div style="display:flex;">
<div>

```bibtex
@article{factr,
  title={FACTR: Force-Attending Curriculum Training for Contact-Rich Policy Learning},
  author={Liu, Jason Jingzhou and Li, Yulong and Shaw, Kenneth and Tao, Tony and Salakhutdinov, Ruslan and Pathak, Deepak},
  journal={arXiv preprint arXiv:2502.17432},
  year={2025}
}
```
