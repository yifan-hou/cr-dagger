# CR-DAgger
Official implementation for paper "Compliant Residual DAgger: Improving Real-World Contact-Rich Manipulation with Human Corrections", by Xiaomeng Xu*, Yifan Hou*, Chendong Xin, Zeyi Liu, and Shuran Song.

[Link to project page](https://compliant-residual-dagger.github.io/)

# Installation
## Setup Virtual Environments
The following is tested on Ubuntu 24.04.

1. Install [mamba](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html)
2. Clone this repo.
``` sh
git clone git@github.com:yifan-hou/cr-dagger.git
```
3. Create a virtual env called `pyrite`:
``` sh
cd PyriteML
# Note 1: If env create gets stuck, you can create an empty environment, then install pytorch/torchvision/torchaudio following official pytorch installation instructions, then install the rest via mamba.
# Note 2: zarr 3 changed many interfaces and does not work for PyriteML. We recommend to use zarr 2.18
mamba env create -f conda_environment.yaml
# after finish, activate it using
mamba activate pyrite
# a few pip installs
pip install robotmq # for cross-process communication
pip install v4l2py
pip install toppra
pip install atomics
pip install vit-pytorch # Need at least 1.7.12, which was not available in conda
pip install imagecodecs # Need at least 2023.9.18, which caused lots of conflicts in conda

# Install local packages
cd PyriteUtilities
pip install -e .
```
4. Setup environment variables: add the following to your .bashrc or .zshrc, edit according to your local path.
``` sh
# where the collected raw data folders are
export PYRITE_RAW_DATASET_FOLDERS=$HOME/data/real
# where the post-processed data folders are
export PYRITE_DATASET_FOLDERS=$HOME/data/real_processed
# Each training session will create a folder here.
export PYRITE_CHECKPOINT_FOLDERS=$HOME/training_outputs
# Hardware configs.
export PYRITE_HARDWARE_CONFIG_FOLDERS=$HOME/hardware_interfaces/workcell/ur_test_bench/config
# Logging folder.
export PYRITE_CONTROL_LOG_FOLDERS=$HOME/data/control_log
```

## Setup robot controllers
We provide an admittance controller implementation based on our [force_control](https://github.com/yifan-hou/force_control) package.

### Setup conda packages
Make sure the conda packages are visible to c++ linkers. Create a .sh file with the following content:
``` sh
# clib_path_activate.sh
export LD_LIBRARY_PATH=/home/yifanhou/miniforge3/envs/pyrite/lib/:$LD_LIBRARY_PATH
```
at `${CONDA_PREFIX}/etc/conda/activate.d/`, e.g. `$HOME/miniforge3/envs/pyrite/etc/conda/activate.d` if you install miniforge at the default location.

### Build the controller
Pull the following packages:
``` sh
# https://github.com/yifan-hou/cpplibrary
git clone git@github.com:yifan-hou/cpplibrary.git
# https://github.com/yifan-hou/force_control important: use the 'tracking' branch
git clone -b tracking git@github.com:yifan-hou/force_control.git
# https://github.com/yifan-hou/hardware_interfaces important: use the 'tracking' branch
git clone -b tracking git@github.com:yifan-hou/hardware_interfaces.git
```
Then build & install following their readme.

After building the `hardware_interfaces` package, a pybind library is generated under `hardware_interfaces/workcell/table_top_manip/python/`. This library contains a c++ multi-thread server that maintains low-latency communication and data/command buffers with all involved hardware. It also maintains an admittance controller. We will launch a python script (the actor) that communicates with the hardware server, while the python script itself does not need multi-processing.


### (Optional) Install to a local path
I recommend to install to a local path for easy maintainence, also you don't need sudo access. To do so, replace the line
``` sh
cmake ..
```
with
``` sh
cmake -DCMAKE_INSTALL_PREFIX=$HOME/.local  ..
```
when building packages above. Here `$HOME/.local` can be replaced with any local path.
Then you need to tell gcc to look for binaries/headers from your local path by adding the following to your .bashrc or .zshrc:
``` sh
export PATH=$HOME/.local/bin:$PATH
export C_INCLUDE_PATH=$HOME/.local/include/:$C_INCLUDE_PATH
export CPLUS_INCLUDE_PATH=$HOME/.local/include/:$CPLUS_INCLUDE_PATH
export LD_LIBRARY_PATH=$HOME/.local/lib/:$LD_LIBRARY_PATH
```
You need to run `source .bashrc` or reopen a terminal for those to take effect.

# Usage

## Overview
There are two main components when running CR-DAgger:

**The Actor**: A python process that maintains the policy inference loops. It has three functionalities: 
1. It maintains communication with the hardware. The actor loads the base policy and optionally the residual policy, reads robot feedback from the hardware, and sends out policy outputs to the hardware. When launching the actor, the low-level controllers (hardware drivers, admittance controllers) are automatically instantiated and runs in the background.
2. Send data to learner. The actor can send the collected time-stamped real data, including human correction motion data, to the Learner for processing and training.
3. Receive residual policy weights. The actor checks whether new policy weights are available at the beginning of every base policy loop.

**The Learner**: A python process that maintains data processing and policy training. It listens to the actor for available data, and sends back updated residual policy weights.

A few useful features of this setup:
* The actor and the learner could be running on different machines and communicate via network, thanks to the [robot-message-queue](https://github.com/yihuai-gao/robot-message-queue) package. This means that you can use one desktop dedicated to online inference, while performing training/data processing on a server, as long as the two machines can ping each other.
* Both actor and learner reads configurations from the single config file, reducing chance of mistake.


## Configurations
### (a) DAgger config
`PyriteML/online_learning/configs/config_v1.py`

This is the top level config file shared across learner and actor. It controls actor and learner behavior, and their communication. Actor and learner code will load this file by importing this python file. This file contains paths to the other two config files.

### (b) Hardware config
`hardware_interfaces/workcell/table_top_manip/config/`

Configs specific to your robot hardware, such as robot ip, calibrations. Note that the robot stiffness parameter under `admittance_controller` will be overriden by the actor node, specified in DAgger config above.


### (c) Training configs
`PyriteML/diffusion_policy/config/train_online_residual_mlp_workspace.yaml` and `PyriteML/diffusion_policy/config/task/*`

Policy training workspace and task configs for the residual policy, following code structure from [UMI](https://github.com/real-stanford/universal_manipulation_interface). Note that you need to specify the path to the base policy checkpoint in the residual workspace config using the `base_policy_ckpt` field.

## Launch

Before launching, check the following:
1. `pyrite` virtual environment is activated.
2. Env variables `PYRITE_CHECKPOINT_FOLDERS`, `PYRITE_HARDWARE_CONFIG_FOLDERS`, `PYRITE_CONTROL_LOG_FOLDERS` are properly set.
 

**Launch the learner**
``` python
python cr-dagger/PyriteML/online_learning/learners/residual_learner_v1.py
```

**Launch the actor**
``` python
python cr-dagger/PyriteEnvSuites/env_runners/residual_online_learning_env_runner.py
```

## Example use cases
### I. Collect correction data on top of the base policy
1. Set `residual_ckpt_path` to `None` in (a) DAgger config
2. Set `send_transitions_to_server` to `True` in (a) DAgger config
3. If you don't want to start training, set `num_episodes_before_first_training` to a large number in (a) DAgger config
3. Make sure the `base_policy_ckpt` points to the correct base policy in (c) training config
4. Launch actor and learner in separate terminals. Follow on-screen instructions of the actor.

### II. Collect correction data on top of base + residual policy
1. Set `residual_ckpt_path` to the correct path in (a) DAgger config
2. Same as I.

### III. Resume correction data collection
1. Make sure `data_folder_path` in (a) DAgger config points to the folder with existing data.
2. Same as above.

### IV. Collect correction data and run training at the same time
Learner will repeatedly 1. process all available new data, 2. launch training, 3. send updated weights to actor.
1. set `num_episodes_before_first_training` to a suitable number in (a) DAgger config.
2. Same as I or II.

### V. Launch residual training using existing data
1. Make sure `data_folder_path` in (a) DAgger config points to the folder with existing data.
2. set `num_episodes_before_first_training` to a number lower than the number of existing episodes in (a) DAgger config.
3. Launch learner.

### VI. Evaluate a residual policy
1. Set `residual_ckpt_path` to the correct path in (a) DAgger config
2. Set `send_transitions_to_server` to `False` in (a) DAgger config
3. Set `num_episodes_before_first_training` to a large number in (a) DAgger config
4. Make sure the `base_policy_ckpt` points to the correct base policy in (c) training config
5. Launch actor and learner. 


### VII. Train the base policy
For data collection and training of the base policy, please refer to `base_policy.md`.


# Citation
If you found this repo useful, you can acknowledge us by citing our paper:
```
@inproceedings{CRDagger2025,
  title={Compliant Residual DAgger: Improving Real-World Contact-Rich Manipulation with Human Corrections},
  author={Xiaomeng Xu* and Yifan Hou* and Chendong Xin and Zeyi Liu and Shuran Song},
  year={2025},
  booktitle={The 39th Annual Conference on Neural Information Processing Systems (NeurIPS)},
}
```


## License
This repository is released under the MIT license. 

## Acknowledgement
Special thanks to [@yihuai-gao](https://github.com/yihuai-gao) for lots of customization of the [robot-message-queue](https://github.com/yihuai-gao/robot-message-queue) package for this project.

## Code References
* This code base is built on top of [adaptive-compliance-policy](https://github.com/yifan-hou/adaptive_compliance_policy/tree/main)
* The cross-process architecture is built using [robot-message-queue](https://github.com/yihuai-gao/robot-message-queue).
* The training pipeline is modified from [universal_manipulation_interface](https://github.com/real-stanford/universal_manipulation_interface)
* We include a copy of [multimodal_representation](https://github.com/stanford-iprl-lab/multimodal_representation) for the temporal convolution encoding for forces.