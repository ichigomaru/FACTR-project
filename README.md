
<h1> FACTR: Force-Attending Curriculum Training for Contact-Rich Policy Learning </h1>



#### [Jason Jingzhou Liu](https://jasonjzliu.com)<sup>\*</sup>, [Yulong Li](https://yulongli42.github.io)<sup>\*</sup>, [Kenneth Shaw](https://kennyshaw.net), [Tony Tao](https://tony-tao.com), [Ruslan Salakhutdinov](https://www.cs.cmu.edu/~rsalakhu/), [Deepak Pathak](https://www.cs.cmu.edu/~dpathak/)
_Carnegie Mellon University_

[Project Page](https://jasonjzliu.com/factr/) | [arXiV](https://arxiv.org/abs/2502.17432)

<h1> </h1>
<img src="assets/main_teaser.jpg" alt="teaser" width="750"/>

<br>

## Catalog
- [Environment](#environment)
- [Data Collection and Processing](#data-collection-and-processing)
- [Training and Testing](#training)
- [Policy Rollout](#policy-rollout)
- [Citation](#citation)
  
## Environment
```bash
conda env create -f env.yaml
conda activate factr
```
We make use of the pretrained features from [data4robotics](https://github.com/SudeepDasari/data4robotics) project. Download the pretrained features as follows:
```bash
bash scripts/download_features.sh
```
## Data Collection and Processing
We provide instructions and sample data collection scripts in ROS2 at [factr_teleop](insert_link). You might need your custom nodes for robots and sensors to run the system. In our case, the collected data is saved in following format:
### Data Structure
Each trajectory is saved as a separate pickle file. Each pickle file contains a dictionary with the following structure:
```
trajectory.pkl
├── "data" : dict
│   ├── "topic_name_1" : list[data_points]
│   ├── "topic_name_2" : list[data_points]
│   └── ...
└── "timestamps" : dict
    ├── "topic_name_1" : list[timestamps]
    ├── "topic_name_2" : list[timestamps]
    └── ...
```
### Key Components:

- **data**: A dictionary where:
  - Keys are the data source names (ROS topic names in our implementation)
  - Values are lists containing the actual data points (low-dimensional states or images)

- **timestamps**: A dictionary where:
  - Keys are the same data source names as in the "data" dictionary
  - Values are lists containing the timestamps when each corresponding data point was recorded

*Note*: Different data sources may log at different frequencies, resulting in varying list lengths across topics. The timestamps are crucial for properly aligning and post-processing the data.
While ROS provides synchronization APIs, we chose to record raw timestamps and perform post-processing to allow for greater flexibility in data analysis and alignment.
```python
# Example of a trajectory structure
{
    "data": {
        "/camera/rgb/image_raw": [image1, image2, ...],
        "/joint_states": [state1, state2, ...],
        "/robot/end_effector_pose": [pose1, pose2, ...]
    },
    "timestamps": {
        "/camera/rgb/image_raw": [1615420323.45, 1615420323.55, ...],
        "/joint_states": [1615420323.40, 1615420323.50, ...],
        "/robot/end_effector_pose": [1615420323.42, 1615420323.52, ...]
    }
}
```
### Data Processing
If the data is in the aforementioned format, we provide sample scripts to process data to create data in [robobuf](https://github.com/AGI-Labs/robobuf) format that our learning pipeline takes in.
```bash
python process_data/process_data.py
```
Please checkout the [config file](process_data/cfg/default.yaml) for details about configurations.

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
We provide instructions and sample policy rollout scripts in ROS2 at [factr_teleop](insert_link). Again, you might need your custom nodes for robots and sensors to run the system.

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
