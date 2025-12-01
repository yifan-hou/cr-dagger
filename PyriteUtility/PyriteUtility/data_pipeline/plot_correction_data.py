import os
import zarr
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import PyriteUtility.spatial_math.spatial_utilities as su

def load_episode_data(episode_folder):
    """Load data from a given episode folder."""
    episode_data = {}
    
    # Load policy inference data
    policy_inference_path = os.path.join(episode_folder, "policy_inference.zarr")
    if os.path.exists(policy_inference_path):
        policy_data = zarr.open(policy_inference_path, mode='r')
        episode_data['ts_targets'] = policy_data[f'ts_targets_0'][:].reshape(-1, 7)
        episode_data['timestamps'] = policy_data['timestamps_s'][:].reshape(-1)
    
    # Load robot data
    robot_data_path = os.path.join(episode_folder, "robot_data_0.json")
    rgb_folder = os.path.join(episode_folder, "rgb_0")
    
    if os.path.exists(robot_data_path):
        episode_data['robot_data'] = pd.read_json(robot_data_path)
    
    if os.path.exists(rgb_folder):
        episode_data['rgb_files'] = sorted([os.path.join(rgb_folder, f) for f in os.listdir(rgb_folder) if f.endswith('.jpg') or f.endswith('.png')])
    
    return episode_data

def create_video_from_rgb(rgb_files, output_dir="", fps=30):
    """Create a video from RGB images."""
    if not rgb_files:
        print("No RGB images found.")
        return
    
    frame = cv2.imread(rgb_files[0])
    height, width, _ = frame.shape
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(os.path.join(output_dir, "video.mp4"), fourcc, fps, (width, height))
    
    for file in rgb_files:
        frame = cv2.imread(file)
        video_writer.write(frame)
    
    video_writer.release()

def plot_policy_vs_robot_data(robot_data, ts_targets, timestamps, output_dir=""):
    """Plot policy inference data with robot data for comparison, aligned by timestamps."""

    # Align timestamps
    robot_timestamps = robot_data['robot_time_stamps'].to_numpy()/ 1000.0
    predicted_ts_id = np.searchsorted(timestamps, robot_timestamps)
    Npolicy = len(timestamps)
    predicted_ts_id = np.minimum(predicted_ts_id, Npolicy - 1)
    
    robot_data = np.vstack(robot_data['ts_pose_fb'])
    predicted_data = ts_targets[predicted_ts_id]

    # transform from pose7 to pose3
    feedback_pose = su.pose7_to_SE3(robot_data)
    predicted_pose = su.pose7_to_SE3(predicted_data)
    # for i, column in enumerate(['x', 'y', 'z']):  # Assuming first 3 columns are position data
    #     plt.plot(timestamps, aligned_robot_data[column], label=f'Robot {column}', linestyle='dashed')
    #     plt.plot(timestamps, ts_targets[:, i], label=f'Policy {column}', linestyle='solid')

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter(feedback_pose[:, 0], feedback_pose[:, 1], feedback_pose[:, 2], label='Robot Position', color='red', s=1)
    ax.scatter(predicted_pose[:, 0], predicted_pose[:, 1], predicted_pose[:, 2], label='Policy Position', color='blue', s=1)
    
    plt.title("Policy Inference vs Robot Data")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(output_dir, "policy_vs_robot.png"))

def visualize_episode(episode_folder):
    """Load and visualize an episode."""
    print(f"Loading episode from {episode_folder}")
    data = load_episode_data(episode_folder)
    
    if 'robot_data' in data and 'ts_targets' in data and 'timestamps' in data:
        plot_policy_vs_robot_data(data['robot_data'], data['ts_targets'], data['timestamps'], output_dir='.')
    
    if 'rgb_files' in data:
        create_video_from_rgb(data['rgb_files'], output_dir='.')

if __name__ == "__main__":
    dataset_folder = '/shared_local/data/raw/correction_new'
    # episode_folders = [os.path.join(dataset_folder, ep) for ep in os.listdir(dataset_folder)]
    episode_folders = [os.path.join(dataset_folder, ep) for ep in sorted(os.listdir(dataset_folder))[:1]]
    
    for ep in episode_folders:
        visualize_episode(ep)
