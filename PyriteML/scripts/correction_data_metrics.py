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
dataset_paths.append("/shared_local/data/processed/online_stow_nb_v5_50/processed")
dataset_paths.append("/shared_local/data/processed/online_stow_nb_v6_take_over/processed")
# dataset_paths.append("/shared_local/data/processed/online_belt_v5_50/processed")
# dataset_paths.append("/shared_local/data/processed/online_belt_takeover/processed")
dataset_names = ["On-Policy Delta", "Take-Over"]


def plot_trajectory_segments(fig, axes, val: np.ndarray, val_timestamp: np.ndarray, 
                           key_timestamps: np.ndarray, color_id = 0, window_size = 3, name="Dataset"):
    # Assumes there are two axes.
    # One for correction start, one for correction end
    
    time_offset_correction_start = -200  # in milliseconds
    time_offset_correction_end = 520  # in milliseconds

    for i, key_time in enumerate(key_timestamps):
        # Find indices within the time window
        if i % 2 == 0:
            # correction start
            key_time += time_offset_correction_start
        else:
            # correction end
            key_time += time_offset_correction_end
        time_mask = (val_timestamp >= key_time) & \
                    (val_timestamp <= key_time + window_size)
        
        if not np.any(time_mask):
            print(f"Warning: No data points found for key moment {key_time} "
                    f"within ±{window_size}s window")
            continue

        # Extract segment data
        segment_times = val_timestamp[time_mask]
        segment_val = val[time_mask, :]
        segment_val = np.linalg.norm(segment_val, axis=1)

        # Convert to relative time (centered at key moment)
        relative_times = segment_times - key_time

        # Plot the segment
        if i == 0:
            # print legend only once
            axes[0].plot(relative_times, segment_val, 
                    label=name, 
                    color=colors[color_id], alpha=0.7, linewidth=1.0)
        else:
            if i % 2 is 0:
                axes[0].plot(relative_times, segment_val, 
                    color=colors[color_id], alpha=0.7, linewidth=1.0)
            else:
                axes[1].plot(relative_times, segment_val, 
                    color=colors[color_id], alpha=0.7, linewidth=1.0)



# Create figure with subplots for each dimension
fig, axes = plt.subplots(2, 1, figsize=(6, 5))

colors = ["#0088FF", "#FF0000"]

dt = 0.002
window_size = 500  # in milliseconds

for data_id in range(len(dataset_paths)):
    dataset_path = dataset_paths[data_id]
    print("Loading dataset from: ", dataset_path)
    # load the zarr dataset from the path
    # ‘r’ means read only (must exist); ‘r+’ means read/write (must exist); ‘a’ means read/write (create if doesn’t exist); ‘w’ means create (overwrite if exists); ‘w-’ means create (fail if exists).
    buffer = zarr.open(dataset_path, mode="r+")

    ep_count = 0
    for ep, ep_data in buffer["data"].items():
        print(ep_count, ": ", ep)
        ep_count += 1

        ts_pose_fb_0 = np.array(buffer["data"][ep]["ts_pose_fb_0"])
        robot_time_stamps_0 = np.array(buffer["data"][ep]["robot_time_stamps_0"])
        key_event_0 = np.array(buffer["data"][ep]["key_event_0"])
        key_event_timestamps_0 = np.array(buffer["data"][ep]["key_event_time_stamps_0"])

        pos = ts_pose_fb_0[:, 0:3]
        vel = (pos[1:] - pos[:-1]) / dt

        vel_padded = np.zeros_like(pos)
        vel_padded[1:] = vel  # padd the first and last frame
        vel_padded[0] = vel[0]  # pad the first frame with
        
        acc = (vel[1:] - vel[:-1]) / dt
        acc_padded = np.zeros_like(pos)
        acc_padded[1:-1] = acc  # padd the first and last frame
        acc_padded[0] = acc[0]  # pad the first frame with the first acc value
        acc_padded[-1] = acc[-1]  # pad the last frame with the last acc value
        
        acc_mag = np.linalg.norm(acc_padded, axis=1)
        

        if ep_count == len(buffer["data"]):
            name = dataset_names[data_id]
        else:
            name = None
        plot_trajectory_segments(fig, axes, vel_padded, robot_time_stamps_0, key_event_timestamps_0, color_id=data_id, window_size=window_size, name=name)

        # # convert to tip position
        # SE3_Ttip = np.eye(4)
        # SE3_Ttip[0:3, 3] = np.array([0.0, 0.0, 0.297])
        # SE3_fb_WT = su.pose7_to_SE3(ts_pose_fb_0)
        # SE3_fb_Wtip = SE3_fb_WT @ SE3_Ttip
        # ts_tip_fb_0 = su.SE3_to_pose7(SE3_fb_Wtip)
    
    axes[1].set_xlabel('Time (ms)')
    for i in range(2):
        ax = axes[i]
        # Customize subplot
        ax.set_ylabel('Velocity (m/s)')
        ax.grid(True, alpha=0.3)
        # Set x-axis limits to show the full window
        ax.set_xlim(0, window_size)
        ax.set_ylim(0, 0.7)
        # ax.legend(bbox_to_anchor=(0.9, 1), loc='upper right', borderaxespad=1, frameon=False)

plt.tight_layout()
plt.savefig('figure.png')
print("Done")


# plt.show()





