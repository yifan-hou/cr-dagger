import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Cursor

import sys
import os
from typing import Dict, Callable, Tuple, List

SCRIPT_PATH = "/local/real/hjchoi92/repo/PyriteML/scripts"
sys.path.append(os.path.join(SCRIPT_PATH, '../'))
sys.path.append(os.path.join(SCRIPT_PATH, '../../PyriteUtility'))

from PyriteUtility.spatial_math import spatial_utilities as su


import numpy as np
import zarr

if "PYRITE_DATASET_FOLDERS" not in os.environ:
    raise ValueError("Please set the environment variable PYRITE_DATASET_FOLDERS")
dataset_folder_path = os.environ.get("PYRITE_DATASET_FOLDERS")

# dataset_path = dataset_folder_path + "/umi-ft/WBW/acp_replay_buffer_gripper.zarr"
dataset_path = "/store/real/hjchoi92/data/real_processed/umift/WBW-iph-b0/processed_data/all/acp_replay_buffer_gripper.zarr/"

print("Loading dataset from: ", dataset_path)
# load the zarr dataset from the path
# ‘r’ means read only (must exist); ‘r+’ means read/write (must exist); ‘a’ means read/write (create if doesn’t exist); ‘w’ means create (overwrite if exists); ‘w-’ means create (fail if exists).
buffer = zarr.open(dataset_path, mode="r+")

# for ep, ep_data in buffer["data"].items():
#     print(ep)
    
ep = "episode_2"
print("Loading episode: ", ep)


print(buffer.tree())

# preprocess pose data into relative pose
pose_IG = np.array(buffer["data"][ep]["ts_pose_fb_0"])
pose_IG0 = pose_IG[0]
SE3_IG0 = su.pose7_to_SE3(pose_IG0)
SE3_G0I = su.SE3_inv(SE3_IG0)
SE3_G0G = np.zeros((pose_IG.shape[0], 4, 4))
for i in range(pose_IG.shape[0]):
    SE3_IGi = su.pose7_to_SE3(pose_IG[i])
    SE3_G0G[i] = SE3_G0I @ SE3_IGi

print("finished processing pose data")

# compute net wrench
wrench_left = np.array(buffer["data"][ep]["wrench_left_0"][..., :3])
wrench_right = np.array(buffer["data"][ep]["wrench_right_0"][..., :3])
# wrench_left_timestamps = np.array(buffer["data"][ep]["wrench_time_stamps_left_0"])
# wrench_right_timestamps = np.array(buffer["data"][ep]["wrench_time_stamps_right_0"])
# # Find closest matching timestamps
# indices = np.searchsorted(wrench_right_timestamps, wrench_left_timestamps, side="left")
# indices = np.clip(indices, 0, len(wrench_right_timestamps) - 1)  # Ensure indices are valid
# wrench_right_aligned = wrench_right[indices]
wrench_TCP = wrench_left + wrench_right

print("finished processing wrench data")

images = np.array(buffer["data"][ep]["rgb_0"])
print("got images")
data1 = np.array(SE3_G0G[..., :3, 3])
print("got data1")
data2 = np.array(buffer["data"][ep]["gripper_0"])
print("got data2")
data3 = wrench_TCP
print("got data3")
data4 = np.array(buffer["data"][ep]["stiffness_0"])
# change data4 from (N,) to (N, 1)
data4 = data4.reshape(-1, 1)

print("finish distributing data for plotting")

time_images = buffer["data"][ep]["rgb_time_stamps_0"]
time_data1 = buffer["data"][ep]["robot_time_stamps_0"]
time_data2 = buffer["data"][ep]["gripper_time_stamps_0"]
time_data3 = buffer["data"][ep]["wrench_time_stamps_left_0"]
time_data4 = buffer["data"][ep]["robot_time_stamps_0"]

titles = ["RGB Image", "Pose", "Gripper", "Wrench", "Stiffness"]

print("Finished loading data")

# # Example Data (Replace with actual data)
# N0, H, W = 10, 100, 100  # Example image count and size
# N1, D1 = 50, 2
# N2, D2 = 60, 3
# N3, D3 = 40, 2
# N4, D4 = 70, 4

# time_images = np.sort(np.random.rand(N0, 1) * 10)  # Timestamps for images
# time_data1 = np.sort(np.random.rand(N1, 1) * 10, axis=0)
# time_data2 = np.sort(np.random.rand(N2, 1) * 10, axis=0)
# time_data3 = np.sort(np.random.rand(N3, 1) * 10, axis=0)
# time_data4 = np.sort(np.random.rand(N4, 1) * 10, axis=0)

# images = np.random.rand(N0, H, W, 3)  # Random images
# data1 = np.random.rand(N1, D1)
# data2 = np.random.rand(N2, D2)
# data3 = np.random.rand(N3, D3)
# data4 = np.random.rand(N4, D4)


# Initialize figure and axes
fig, axes = plt.subplots(5, 1, figsize=(10, 18), gridspec_kw={'height_ratios': [3, 1, 1, 1, 1], 'hspace': 0.5})
ax_img, ax1, ax2, ax3, ax4 = axes
ax_img.axis("off")  # Hide image axis

# Set plot title
fig.suptitle(f"Plotting for {ep}")

# Plot data
ax1.plot(time_data1, data1, '.-')
ax2.plot(time_data2, data2, '.-')
ax3.plot(time_data3, data3, '.-')
ax4.plot(time_data4, data4, '.-')

# Set titles
ax_img.set_title(titles[0])
ax1.set_title(titles[1])
ax2.set_title(titles[2])
ax3.set_title(titles[3])
ax4.set_title(titles[4])

# set legends for plot 1 and 3
ax1.legend(["x", "y", "z"])
ax3.legend(["x", "y", "z"])

# Shared vertical cursor
cursor_time = np.min(time_images)  # Initialize cursor at first timestamp
cursor_line = [ax1.axvline(cursor_time, color='r', linestyle='--'),
               ax2.axvline(cursor_time, color='r', linestyle='--'),
               ax3.axvline(cursor_time, color='r', linestyle='--'),
               ax4.axvline(cursor_time, color='r', linestyle='--')]

# Function to update the displayed image
def update_image(time):
    closest_idx = np.argmin(np.abs(time_images - time))
    ax_img.clear()
    ax_img.imshow(images[closest_idx])
    ax_img.axis("off")
    fig.canvas.draw_idle()

update_image(cursor_time)

# Keyboard event handler
def on_key(event):
    global cursor_time
    step = 0.1  # Time step
    if event.key == "right":
        cursor_time += step
    elif event.key == "left":
        cursor_time -= step
    else:
        return
    
    print("Showing time: ", cursor_time)
    print("Image shape:", images.shape)
    for line in cursor_line:
        line.set_xdata([cursor_time])
    update_image(cursor_time)
    fig.canvas.draw_idle()

fig.canvas.mpl_connect("key_press_event", on_key)
plt.show()
