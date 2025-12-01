import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Cursor
import matplotlib.pyplot as plt
from typing import Tuple, List

import sys
import os
from typing import Dict, Callable, Tuple, List

SCRIPT_PATH = "/home/yifanhou/git/PyriteML/scripts"
sys.path.append(os.path.join(SCRIPT_PATH, '../'))

# from wrench_calibration import MLP
from PyriteUtility.spatial_math import spatial_utilities as su
from PyriteUtility.computer_vision.imagecodecs_numcodecs import register_codecs
register_codecs()

import numpy as np
import zarr

if "PYRITE_DATASET_FOLDERS" not in os.environ:
    raise ValueError("Please set the environment variable PYRITE_DATASET_FOLDERS")
dataset_folder_path = os.environ.get("PYRITE_DATASET_FOLDERS")


show_correction = True

dataset_paths = []


# dataset_paths.append("/shared_local/data/processed/belt_assembly/")
# dataset_paths.append("/shared_local/data/processed/online_stow_nb_v5_50/processed")
# dataset_paths.append("/shared_local/data/processed/online_stow_nb_v6_take_over/processed")

dataset_paths.append("/shared_local/data/processed/belt_assembly/")
dataset_paths.append("/shared_local/data/processed/online_belt_v5_50/processed")
dataset_paths.append("/shared_local/data/processed/online_belt_takeover/processed")
dataset_names = ["Base","On Policy Delta", "Take Over Correction"]


def plot_trajectory(fig, axes, pos_all: np.ndarray, # (N_episodes, N_points, 3)
                           color_id = 0, name=""):
    for dim in range(3):
        ax = axes[dim]
        # Plot the segment
        ax.plot(pos_all[:, :, dim].T,
            color=colors[color_id], alpha=0.7, linewidth=1)

def plot_range(fig, axes, pos_mean: np.ndarray, pos_std: np.ndarray,
                       color_id = 0):
    for dim in range(3):
        ax = axes[dim]
        # Plot the segment
        N = pos_mean.shape[0]
        ax.plot(np.arange(N)/N, pos_mean[:, dim],
            color=colors[color_id], alpha=1.0, linewidth=2, label=dataset_names[color_id])
        ax.fill_between(np.arange(N)/N,
                        pos_mean[:, dim] - pos_std[:, dim],
                        pos_mean[:, dim] + pos_std[:, dim],
                        color=colors[color_id], alpha=0.3)

# Create figure with subplots for each dimension
fig, axes = plt.subplots(3, 1, figsize=(6, 4))
# fig.suptitle('Fingertip Trajectory for Belt Assembly', fontsize=16)


colors = ["#0088FF", "#00CC00", "#FF0000", "#FF8800", "#8800FF"]

dt = 0.002

N_points = 1000
for data_id in range(len(dataset_paths)):
    dataset_path = dataset_paths[data_id]
    print("Loading dataset from: ", dataset_path)
    # load the zarr dataset from the path
    # ‘r’ means read only (must exist); ‘r+’ means read/write (must exist); ‘a’ means read/write (create if doesn’t exist); ‘w’ means create (overwrite if exists); ‘w-’ means create (fail if exists).
    buffer = zarr.open(dataset_path, mode="r+")

    N_episodes = len(buffer["data"].keys())
    pos_all = np.zeros((N_episodes, N_points, 3))

    ep_count = 0
    for ep, ep_data in buffer["data"].items():
        print(ep_count, ": ", ep)

        ts_pose_fb_0 = np.array(buffer["data"][ep]["ts_pose_fb_0"])
        robot_time_stamps_0 = np.array(buffer["data"][ep]["robot_time_stamps_0"])

        # scale robot_time_stamps_0 to total_duration
        unit_progress_0 = robot_time_stamps_0 / robot_time_stamps_0[-1]

        # convert to tip position
        SE3_Ttip = np.eye(4)
        SE3_Ttip[0:3, 3] = np.array([0.0, 0.0, 0.297])
        SE3_fb_WT = su.pose7_to_SE3(ts_pose_fb_0)
        SE3_fb_Wtip = SE3_fb_WT @ SE3_Ttip
        ts_tip_fb_0 = su.SE3_to_pose7(SE3_fb_Wtip)
        pos = ts_tip_fb_0[:, 0:3]
        
        uniform_progress = np.linspace(0, 1, N_points)
        
        pos_uniform_sampled_ids = np.searchsorted(unit_progress_0, uniform_progress, side="left")
        pos = pos[pos_uniform_sampled_ids, :]

        pos_all[ep_count, :, :] = pos
        ep_count += 1


    # compute statistics
    pos_mean = np.mean(pos_all, axis=0)  # (N_points, 3)
    pos_std = np.std(pos_all, axis=0)    # (N_points,

    # plot all traj in this dataset
    name = dataset_names[data_id]
    plot_range(fig, axes, pos_mean, pos_std, color_id=data_id)
    # plot_trajectory(fig, axes, pos_all, color_id=data_id)


dimension_labels = ['X', 'Y', 'Z']
axes[0].grid(True, alpha=0.3)
# axes[0].legend(bbox_to_anchor=(1.05, 1), loc='center right', borderaxespad=0.)
axes[0].set_ylim(0, 0.55)
axes[1].set_ylim(0.4, 0.95)
axes[2].set_ylim(0, 0.55)

for dim in range(3):
    ax = axes[dim]
    ax.set_xlim(0, 1)
    ax.set_ylabel(f'{dimension_labels[dim]} Position (m)')
    ax.grid(True, alpha=0.3)

axes[2].set_xlabel('Progress')

plt.tight_layout()
plt.show()





