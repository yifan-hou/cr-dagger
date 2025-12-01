# PyriteML
Models and training pipelines.

## Install Pyrite Packages
The following is tested on Ubuntu 22.04.

1. Install [mamba](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html)
2. Clone the pyrite repos.
``` sh
git clone git@github.com:yifan-hou/PyriteEnvSuites.git
git clone git@github.com:yifan-hou/PyriteConfig.git
git clone git@github.com:yifan-hou/PyriteUtility.git
git clone --recursive git@github.com:yifan-hou/PyriteML.git
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

## Train a diffusion policy
Train a diffusion policy as the base policy.

1. Set path to your zarr data in your task config under PyriteML/diffusion_policy/config/task/stow_no_force.yaml
2. Make sure the above task yaml is being selected in the workspace config at PyriteML/diffusion_policy/config/train_dp_workspace.yaml
3. Launch training with:
``` sh
clear;
cd PyriteML;
HYDRA_FULL_ERROR=1 accelerate launch train.py --config-name=train_dp_workspace
```
Example of multi-gpu training:
``` sh
HYDRA_FULL_ERROR=1 accelerate launch --gpu_ids 4,5 --num_processes=2 --main_process_port=28888  train.py --config-name=train_dp_workspace
```




## 🏷️ License
This repository is released under the MIT license. See [LICENSE](LICENSE) for additional details.

