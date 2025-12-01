import numpy as np
import copy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec

# Initialize current time index
current_rgb_idx = 0
max_rgb_idx = 0

def draw_timed_traj_and_rgb(rgb_images,
                            rgb_time_stamps,
                            data: dict,
                            data_time_stamps: dict,
                            additional_data: dict = None,
                            additional_data_time_stamps: dict = None,
                            key_events=None,
                            key_event_timestamps=None,
                            elev=20, azim=280):
    global current_rgb_idx, max_rgb_idx
    current_rgb_idx = 0
    max_rgb_idx = len(rgb_time_stamps) - 1

    # colors
    colors = [(0.7, 0, 0), (0, 0, 0.7), (0, 0.7, 0)]
    dark_colors = [(1, 0, 0), (0, 0, 1), (0, 1, 0)]


    # process key event
    if key_events is not None:
        correction_start_indices = np.where(key_events == 1)[0]
        assert(len(correction_start_indices) > 0)
        correction_start_timestamps = key_event_timestamps[correction_start_indices]
        correction_end_indices = np.where(key_events == 0)[0]
        correction_end_timestamps = key_event_timestamps[correction_end_indices]

        data_is_correction = {}
        for key, value in data.items():
            flag_is_correction = np.zeros(value.shape[0], dtype=bool)
            for start, end in zip(correction_start_timestamps, correction_end_timestamps):
                start_id = np.searchsorted(data_time_stamps[key], start)
                end_id = np.searchsorted(data_time_stamps[key], end, side="left")
                flag_is_correction[start_id:end_id] = True
            data_is_correction[key] = flag_is_correction
        
        additional_data_is_correction = {}
        if additional_data is not None:
            for key, value in additional_data.items():
                flag_is_correction = np.zeros(value.shape[0], dtype=bool)
                for start, end in zip(correction_start_timestamps, correction_end_timestamps):
                    start_id = np.searchsorted(additional_data_time_stamps[key], start)
                    end_id = np.searchsorted(additional_data_time_stamps[key], end, side="left")
                    flag_is_correction[start_id:end_id] = True
                additional_data_is_correction[key] = flag_is_correction


    def find_closest_index(time_stamps, target_time):
        """Find the index of the closest timestamp."""
        return np.argmin(np.abs(time_stamps - target_time))

    def update_plot():
        """Update the plot based on the current time index."""
        global current_rgb_idx, max_rgb_idx

        # Get the current pose and image
        current_time = rgb_time_stamps[current_rgb_idx]

        # Update RGB image
        ax_image.clear()
        ax_image.imshow(rgb_images[current_rgb_idx])
        ax_image.set_title(f"Time: {current_time:.2f}ms, rgb index: {current_rgb_idx}")
        ax_image.axis('off')

        # Update 3D trajectory with highlighted point
        ax_3d.clear()
        i = 0
        for key, value in data.items():
            color = colors[i]
            dark_color = dark_colors[i]
            i = i + 1
            # plot the whole trajectory
            ax_3d.plot(value[:, 0], value[:, 1], value[:, 2], alpha=0.6, label=key, color=color)
            # find the closest data point to the current
            closest_data_idx = find_closest_index(data_time_stamps[key], current_time)
            closest_data = value[closest_data_idx]
            ax_3d.scatter(closest_data[0],
                        closest_data[1],
                        closest_data[2],
                        s=10, label="_nolegend_", color=dark_color)
            # plot the points that are marked as correction
            if key_events is not None:
                flag_is_correction = data_is_correction[key]
                correction_indices = np.where(flag_is_correction)[0]
                if len(correction_indices) > 0:
                    ax_3d.scatter(value[correction_indices, 0],
                                value[correction_indices, 1],
                                value[correction_indices, 2],
                                s=10, alpha=0.2, label="_nolegend_", color=color)
                    
                    if flag_is_correction[closest_data_idx]:
                        # redraw the "current" node
                        ax_3d.scatter(closest_data[0],
                            closest_data[1],
                            closest_data[2],
                            s=45, label="_nolegend_", color=dark_color)
                      
        ax_3d.set_title("3D Trajectory")
        # ax_3d.legend(traj_names)
        ax_3d.legend()
        ax_3d.set_xlabel('X')
        ax_3d.set_ylabel('Y')
        ax_3d.set_zlabel('Z')
        
        # Update additional data if provided
        if additional_data is not None:
            ax_2d.clear()
            # traj_names = []
            title = ""
            for key, value in additional_data.items():
                # plot the whole trajectory
                ax_2d.plot(additional_data_time_stamps[key], value[:, 0], alpha=0.7, label=key + " X")
                ax_2d.plot(additional_data_time_stamps[key], value[:, 1], alpha=0.7, label=key + " Y")
                ax_2d.plot(additional_data_time_stamps[key], value[:, 2], alpha=0.7, label=key + " Z")
                # find the closest data point to the current
                closest_data_idx = find_closest_index(additional_data_time_stamps[key], current_time)
                closest_data = value[closest_data_idx]
                title = title + key + f": {closest_data[0]:.2f}, {closest_data[1]:.2f}, {closest_data[2]:.2f}\n"
                ax_2d.scatter(current_time, closest_data[0], s=50, label="Current " + key + " X")
                ax_2d.scatter(current_time, closest_data[1], s=50, label="Current " + key + " Y")
                ax_2d.scatter(current_time, closest_data[2], s=50, label="Current " + key + " Z")
            ax_2d.set_title(title)
            ax_2d.legend()
            ax_2d.set_xlabel('time')
            ax_2d.set_ylabel('N')

        plt.draw()

    def on_key(event):
        """Handle key press events to move time forward or backward."""
        global current_rgb_idx, max_rgb_idx

        if event.key == 'right':
            current_rgb_idx = min(current_rgb_idx + 5, max_rgb_idx)
        elif event.key == 'left':
            current_rgb_idx = max(current_rgb_idx - 5, 0)
        elif event.key == 'up':
            current_rgb_idx = min(current_rgb_idx + 25, max_rgb_idx)
        elif event.key == 'down':
            current_rgb_idx = max(current_rgb_idx - 25, 0)

        update_plot()

    # Create figure and axes
    fig = plt.figure(figsize=(8, 8))
    if additional_data == None:
        gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1.5])
        ax_image = plt.subplot(gs[0])
        ax_3d = plt.subplot(gs[1], projection='3d')
        ax_3d.view_init(elev=elev, azim=azim)
    else:
        gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1.5])
        ax_image = plt.subplot(gs[0,0])
        ax_3d = plt.subplot(gs[1,0], projection='3d')
        ax_3d.view_init(elev=elev, azim=azim)
        ax_2d = plt.subplot(gs[1,1])

    # Initial plot
    update_plot()

    # Connect key press event
    fig.canvas.mpl_connect('key_press_event', on_key)

    plt.show()


def main():
    # Sample data for demonstration (replace these with your actual data)
    N, M = 100, 200
    rgb_images = np.random.randint(0, 255, (N, 100, 100, 3), dtype=np.uint8)
    rgb_time_stamps = np.linspace(0, 10, N)
    trajectory_points = np.cumsum(np.random.randn(M, 3), axis=0)
    pose_time_stamps = np.linspace(0, 10, M)

    data = {"trajectory": trajectory_points}
    data_time_stamps = {"trajectory": pose_time_stamps}

    draw_timed_traj_and_rgb(rgb_images, rgb_time_stamps, data, data_time_stamps)

    print("Done!")

if __name__ == "__main__":
    main()