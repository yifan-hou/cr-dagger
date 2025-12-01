Follow the following steps if you want to collect data and train a base policy by yourself.

You do not need these steps if you are using an existing checkpoint as the base policy.

# Data collection
(Requires robot controller setup)
The data collection pipeline is wrapped in `hardware_interfaces/applications/manipulation_data_collection".
1. Check the correct config file is selected in `hardware_interfaces/applications/manipulation_data_collection/src/main.cc`.
2. Build `hardware_interfaces` follow its readme.
3. On the UR teach pendant, make sure you calibrated the TCP mass. 
4. Edit the config file specified in step 1, make sure you have the correct hardware IP/ID, data saving path, etc.
5. Launch the manipulation_data_collection binary:
``` sh
cd hardware_interfaces/build
./applications/manipulation_data_collection/manipulation_data_collection
```
Then follow the on screen instructions.

Our data collection pipeline saves data episode by episode. The saved data folder looks like this:
```
current_dataset/
    episode_1727294514
    episode_1727294689
    episode_1727308394/
        rgb_0
        rgb_1/
            img_count_timestamp.jpg
            ...
        robot_data_0.json
        wrench_data_0.json
    ...
```
Within an episode, each file/folder corresponds to a device. Every frame of data is saved with a timestamp that was calibrated across all devices. For rgb images, its timestamp is saved in its file name, e.g.
```
img_000695_29345.186724_ms
```
means that this image is the 695th frame saved in this episode, and it is saved at 29345.186724ms since the program launched. 

## Data postprocessing
We postprocess the data to match the data format for training (zarr).
``` python
python cr-dagger/PyriteUtility/data_pipeline/real_data_processing.py
```
Specify `id_list`, `input_dir`, `output_dir` then run the script. This script will compress the images into a [zarr](https://zarr.dev/) database, then generate meta data for the whole dataset. This step creates a new folder at `output_dir`.

# Launch Training
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
