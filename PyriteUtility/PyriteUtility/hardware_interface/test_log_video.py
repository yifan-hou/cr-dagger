import sys
import os

sys.path.append(os.path.join(sys.path[0], "../.."))  # PyriteUtility

import cv2
import json
import time
import numpy as np
from multiprocessing.managers import SharedMemoryManager

from PyriteUtility.umi_utils.usb_util import (
    reset_all_elgato_devices,
    get_sorted_v4l_paths,
)
from PyriteUtility.hardware_interface.multi_uvc_camera import MultiUvcCamera
from PyriteUtility.hardware_interface.multi_camera_visualizer import (
    MultiCameraVisualizer,
)
from PyriteUtility.hardware_interface.video_recorder import VideoRecorder

save_video_path = "/home/yifanhou/data/experiment_log"
v4l_paths = [
    "/dev/v4l/by-id/usb-Elgato_Cam_Link_4K_A29YB41521ZMA3-video-index0",
    "/dev/v4l/by-id/usb-Elgato_Game_Capture_HD60_X_00000001-video-index0",
]

fps = 60
resolution = (1920, 1080)

# resolution = [(3840, 2160), (3840, 2160)]


def test():

    # Find and reset all Elgato capture cards.
    # Required to workaround a firmware bug.
    reset_all_elgato_devices()

    # Wait for all v4l cameras to be back online
    time.sleep(0.1)

    video_recorder = [
        VideoRecorder.create_hevc_nvenc(
            fps=fps, input_pix_fmt="bgr24", bit_rate=6000 * 1000
        )
        for v in v4l_paths
    ]

    with SharedMemoryManager() as shm_manager:
        with MultiUvcCamera(
            dev_video_paths=v4l_paths,
            shm_manager=shm_manager,
            resolution=resolution,
            capture_fps=fps,
            video_recorder=video_recorder,
            verbose=False,
        ) as camera:
            print("Started camera")
            with MultiCameraVisualizer(
                camera=camera, row=2, col=1, vis_fps=fps, rgb_to_bgr=False
            ) as multi_cam_vis:

                cv2.setNumThreads(1)
                video_path = save_video_path + f"/{time.strftime('%Y%m%d_%H%M%S')}/"
                rec_start_time = time.time() + 1
                camera.start_recording(video_path, start_time=rec_start_time)
                camera.restart_put(rec_start_time)
                time.sleep(1.5)

                print("Recording started")
                while True:
                    time.sleep(0.5)
                    if time.time() - rec_start_time > 20:
                        print("----------Time is up!----------")
                        break

                camera.stop_recording()
                camera.stop()


if __name__ == "__main__":
    test()
