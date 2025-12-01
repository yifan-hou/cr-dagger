import sys
import os

sys.path.append(os.path.join(sys.path[0], "../../"))

import zarr
import re
import shutil
from dataclasses import dataclass
from typing import Union, Optional, List, Dict
import imageio
import numpy as np
from PyriteUtility.computer_vision.imagecodecs_numcodecs import (
    register_codecs,
    Jpeg2k,
    JpegXl,
)
import PyriteUtility.spatial_math.spatial_utilities as su
import numcodecs
import concurrent.futures

from plotly.subplots import make_subplots
import plotly.graph_objs as go
import plotly.io as pio
import plotly.express as px

pio.templates.default = "plotly_dark"
pio.renderers.default = "browser"

register_codecs()


@dataclass
class VideoData:
    rgb: any
    depth: Optional[any] = None
    segmentation: Optional[any] = None
    camera_id: Optional[int] = None

    @property
    def length(self) -> int:
        return len(self.rgb)

    @classmethod
    def stack(cls, video_data_list: List["VideoData"]) -> "VideoData":
        # Concatenate rgb
        stacked_rgb = np.stack([video_data.rgb for video_data in video_data_list])

        # Concatenate depth
        if all(video_data.depth is not None for video_data in video_data_list):
            stacked_depth = np.stack(
                [video_data.depth for video_data in video_data_list]
            )
        else:
            stacked_depth = None

        # Concatenate segmentation
        if all(video_data.segmentation is not None for video_data in video_data_list):
            stacked_segmentation = np.stack(
                [video_data.segmentation for video_data in video_data_list]
            )
        else:
            stacked_segmentation = None

        camera_id = video_data_list[0].camera_id

        return cls(
            rgb=stacked_rgb,
            depth=stacked_depth,
            segmentation=stacked_segmentation,
            camera_id=camera_id,
        )

    def to_mp4(self, path: str, fps: int = 30):
        imageio.mimwrite(path, self.rgb, fps=fps)


@dataclass
class EpisodeData:
    camera_datas: List[VideoData]
    js_command: Optional[any] = None
    js_fb: Optional[any] = None
    ts_pose_command: Optional[any] = None
    ts_pose_fb: Optional[any] = None
    ft_sensor_pose_fb: Optional[any] = None
    low_dim_state: Optional[any] = None
    qpos: Optional[any] = None
    qvel: Optional[any] = None
    js_force: Optional[any] = None
    wrench: Optional[any] = None
    wrench_filtered: Optional[any] = None
    visual_time_stamps: Optional[any] = None
    low_dim_time_stamps: Optional[any] = None
    info: Optional[any] = None

    @property
    def length(self) -> int:
        return len(self.js_command)


def img_copy(zarr_arr, zarr_idx, np_array, np_idx):
    # print(zarr_arr.shape, np_array.shape, zarr_arr.dtype,
    #       np_array.dtype, zarr_idx, np_idx)
    try:
        zarr_arr[zarr_idx] = np_array[np_idx]
        # make sure we can successfully decode
        _ = zarr_arr[zarr_idx]
        return True
    except Exception as e:
        print(e)
        return False


def img_copy_single(zarr_arr, zarr_idx, np_array):
    try:
        zarr_arr[zarr_idx] = np_array
        # make sure we can successfully decode
        _ = zarr_arr[zarr_idx]
        return True
    except Exception as e:
        print(e)
        return False


##
## Use:
##   Call once: create_zarr_groups_for_episode()
##   Call once: save_video_for_episode()
##   Call once: save_low_dim_for_episode()
## Exactly one of the following two parameters should be provided:
##      episode_id is provide: create store at initialization, work on one episode
##      data is provided: use the given data, work on multiple episodes
## store_path should always be provided, it is used for saving the mp4 video.
class EpisodeDataBuffer:
    def __init__(
        self,
        store_path,
        camera_ids,
        max_workers=32,
        save_video=True,
        save_video_fps=30,
        episode_id=None,
        data=None,
    ) -> None:

        if episode_id is None:
            assert data is not None
            self.episode_id = None
            self.store = None
            self.root = data
        else:
            assert data is None
            self.episode_id = episode_id
            self.store = zarr.DirectoryStore(path=self.store_path)
            # self.store = zarr.LMDBStore(path=self.store_path)
            self.root = zarr.open(store=self.store, mode="a")

        self.store_path = store_path
        self.camera_ids = camera_ids
        self.max_workers = max_workers
        self.save_video = save_video
        self.save_video_fps = save_video_fps

    def reset(self):
        keys = list(self.root.group_keys())
        for key in keys:
            del self.root[key]

    def find_max_eps(self, root):
        keys = list(root.group_keys())
        if len(keys) == 0:
            return -1
        else:
            return max([int(re.findall(r"\d+", key)[0]) for key in keys])

    def create_zarr_groups_for_episode(self, rgb_shapes, id_list, episode_id=None):
        assert id_list == self.camera_ids

        if self.episode_id is None:
            assert episode_id is not None
        else:
            assert episode_id == None
            episode_id = self.episode_id

        if self.store_path is not None:
            # check if the episode path already exists
            # This has only been useful for simulation data generation
            episode_path = f"{self.store_path}/data/episode_{episode_id}"
            if os.path.exists(episode_path):
                # input(f"The path {episode_path} already exists. Press Enter to delete it.")
                shutil.rmtree(episode_path)
        data = self.root.require_group("data")
        episode_data = data.create_group(f"episode_{episode_id}")
        # create groups for rgb data
        if len(rgb_shapes) > 0:
            for id in self.camera_ids:
                this_compressor = JpegXl(level=80, numthreads=1)
                n, h, w, c = rgb_shapes[id]
                episode_data.require_dataset(
                    f"rgb_{id}",
                    shape=(n, h, w, c),
                    chunks=(1, h, w, c),
                    dtype=np.uint8,
                    compressor=this_compressor,
                )
                episode_data.create_group(f"rgb_time_stamps_{id}")
                episode_data[f"rgb_time_stamps_{id}"] = zarr.array(np.zeros(n))
        # low dim data groups will be created when calling save_low_dim_for_episode

    def save_low_dim_for_episode(
        self,
        js_command: Optional[any] = None,
        js_fb: Optional[any] = None,
        ts_pose_command: Optional[any] = None,
        ts_pose_fb: Optional[any] = None,
        robot_wrench: Optional[any] = None,
        gripper_fb: Optional[any] = None,
        policy_pose_command: Optional[any] = None,
        policy_gripper_command: Optional[any] = None,
        low_dim_state: Optional[any] = None,
        wrench: Optional[any] = None,
        wrench_filtered: Optional[any] = None,
        item_poses: Optional[any] = None,
        key_event: Optional[any] = None,
        robot_time_stamps: Optional[any] = None,
        gripper_time_stamps: Optional[any] = None,
        wrench_time_stamps: Optional[any] = None,
        policy_time_stamps: Optional[any] = None,
        item_time_stamps: Optional[any] = None,
        key_event_time_stamps: Optional[any] = None,
        episode_id=None,
        masks=None,
    ):
        if self.episode_id is None:
            assert episode_id is not None
        else:
            assert episode_id == None
            episode_id = self.episode_id

        episode_data = self.root["data"][f"episode_{episode_id}"]

        if js_command is not None and len(js_command) > 0:
            for i, js in enumerate(js_command):
                episode_data[f"js_command_{i}"] = zarr.array(js)

        if js_fb is not None and len(js_fb) > 0:
            for i, js in enumerate(js_fb):
                episode_data[f"js_fb_{i}"] = zarr.array(js)

        if ts_pose_command is not None and len(ts_pose_command) > 0:
            for i, ts in enumerate(ts_pose_command):
                episode_data[f"ts_pose_command_{i}"] = zarr.array(ts)

        if ts_pose_fb is not None and len(ts_pose_fb) > 0:
            for i, ts in enumerate(ts_pose_fb):
                episode_data[f"ts_pose_fb_{i}"] = zarr.array(ts)
        
        if robot_wrench is not None and len(robot_wrench) > 0:
            for i, ts in enumerate(robot_wrench):
                episode_data[f"robot_wrench_{i}"] = zarr.array(ts)

        if gripper_fb is not None and len(gripper_fb) > 0:
            for i, ts in enumerate(gripper_fb):
                episode_data[f"gripper_{i}"] = zarr.array(ts)

        if policy_pose_command is not None and len(policy_pose_command) > 0:
            for i, ts in enumerate(policy_pose_command):
                episode_data[f"policy_pose_command_{i}"] = zarr.array(ts)
        
        if policy_gripper_command is not None and len(policy_gripper_command) > 0:
            for i, ts in enumerate(policy_gripper_command):
                episode_data[f"policy_gripper_command_{i}"] = zarr.array(ts)

        if low_dim_state is not None and len(low_dim_state) > 0:
            episode_data["low_dim_state"] = zarr.array(low_dim_state)

        if wrench is not None and len(wrench) > 0:
            for i, w in enumerate(wrench):
                episode_data[f"wrench_{i}"] = zarr.array(w)

        if wrench_filtered is not None and len(wrench_filtered) > 0:
            for i, w in enumerate(wrench_filtered):
                episode_data[f"wrench_filtered_{i}"] = zarr.array(w)

        if item_poses is not None and len(item_poses) > 0:
            for i, w in enumerate(item_poses):
                episode_data[f"item_poses_{i}"] = zarr.array(w)

        if robot_time_stamps is not None and len(robot_time_stamps) > 0:
            for i, ts in enumerate(robot_time_stamps):
                episode_data[f"robot_time_stamps_{i}"] = zarr.array(ts)
        
        if gripper_time_stamps is not None and len(gripper_time_stamps) > 0:
            for i, ts in enumerate(gripper_time_stamps):
                episode_data[f"gripper_time_stamps_{i}"] = zarr.array(ts)

        if wrench_time_stamps is not None and len(wrench_time_stamps) > 0:
            for i, ts in enumerate(wrench_time_stamps):
                episode_data[f"wrench_time_stamps_{i}"] = zarr.array(ts)

        if item_time_stamps is not None and len(item_time_stamps) > 0:
            for i, ts in enumerate(item_time_stamps):
                episode_data[f"item_time_stamps_{i}"] = zarr.array(ts)

        if policy_time_stamps is not None and len(policy_time_stamps) > 0:
            for i, ts in enumerate(policy_time_stamps):
                episode_data[f"policy_time_stamps_{i}"] = zarr.array(ts)

        if key_event is not None and len(key_event) > 0:
            for i, ts in enumerate(key_event):
                episode_data[f"key_event_{i}"] = zarr.array(ts)
        if key_event_time_stamps is not None and len(key_event_time_stamps) > 0:
            for i, ts in enumerate(key_event_time_stamps):
                episode_data[f"key_event_time_stamps_{i}"] = zarr.array(ts)

        if masks is not None and len(masks) > 0:
            for i, mask in enumerate(masks):
                episode_data[f"mask_{i}"] = zarr.array(mask)

    def save_video_for_episode(
        self,
        visual_observations: Dict[int, VideoData],
        visual_time_stamps: List,
        episode_id=None,
    ):
        if self.episode_id is None:
            assert episode_id is not None
        else:
            assert episode_id == None
            episode_id = self.episode_id

        episode_data = self.root["data"][f"episode_{episode_id}"]
        for camera_id in self.camera_ids:
            rgb_arr = episode_data[f"rgb_{camera_id}"]
            n, h, w, c = visual_observations[camera_id].rgb.shape
            # with parallelization
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=self.max_workers
            ) as executor:
                futures = set()
                for i in range(n):
                    futures.add(
                        executor.submit(
                            img_copy, rgb_arr, i, visual_observations[camera_id].rgb, i
                        )
                    )
                completed, futures = concurrent.futures.wait(futures)
                for f in completed:
                    if not f.result():
                        raise RuntimeError("Failed to encode image!")

            if self.save_video:
                visual_observations[camera_id].to_mp4(
                    f"{self.store_path}/data/episode_{episode_id}/camera{camera_id}_rgb.mp4",
                    fps=self.save_video_fps,
                )

        for id in self.camera_ids:
            episode_data[f"rgb_time_stamps_{id}"] = zarr.array(visual_time_stamps[id])

    def delete_episode_data(self, episode_id):
        episode_path = f"{self.store_path}/data/episode_{episode_id}"
        if os.path.exists(episode_path):
            shutil.rmtree(episode_path)

    def __repr__(self) -> str:
        return str(self.root.tree())

    def plot_low_dim(self, episode_id=None):
        if self.episode_id is None:
            assert episode_id is not None
        else:
            assert episode_id == None
            episode_id = self.episode_id

        episode_data = self.root["data"][f"episode_{episode_id}"]

        ts_pose_command = episode_data["ts_pose_command"]
        ts_pose_fb = episode_data["ts_pose_fb"]
        # ft_sensor_pose_fb = episode_data["ft_sensor_pose_fb"]
        # low_dim_state = episode_data["low_dim_state"]
        # qpos = episode_data["qpos"]
        # qvel = episode_data["qvel"]
        # js_force = episode_data["js_force"]
        wrench = episode_data["wrench"]
        wrench_filtered = episode_data["wrench_filtered"]
        times = episode_data["low_dim_time_stamps"]

        # compute wrench in world frame
        wrench_world = np.zeros_like(wrench)
        wrench_world_filtered = np.zeros_like(wrench_filtered)
        for i in range(len(wrench)):
            SE3_WT = su.pose7_to_SE3(ts_pose_fb[i])
            adj_WT = su.SE3_to_adj(SE3_WT)
            wrench_world[i] = adj_WT @ wrench[i]
            wrench_world_filtered[i] = adj_WT @ wrench_filtered[i]

        fig = make_subplots(
            rows=6,
            cols=2,
            shared_xaxes=True,
            subplot_titles=(
                "Px",
                "Fx(world)",
                "Py",
                "Fy(world)",
                "Pz",
                "Fz(world)",
                "Qw",
                "Tx(world)",
                "Qx",
                "Ty(world)",
                "Qy",
                "Tz(world)",
            ),
        )

        fig.add_trace(
            go.Scatter(x=times, y=ts_pose_command[:, 0], name="ts_pose_command_0"),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(x=times, y=ts_pose_command[:, 1], name="ts_pose_command_1"),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(x=times, y=ts_pose_command[:, 2], name="ts_pose_command_2"),
            row=3,
            col=1,
        )
        fig.add_trace(
            go.Scatter(x=times, y=ts_pose_command[:, 3], name="ts_pose_command_3"),
            row=4,
            col=1,
        )
        fig.add_trace(
            go.Scatter(x=times, y=ts_pose_command[:, 4], name="ts_pose_command_4"),
            row=5,
            col=1,
        )
        fig.add_trace(
            go.Scatter(x=times, y=ts_pose_command[:, 5], name="ts_pose_command_5"),
            row=6,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=times, y=ts_pose_fb[:, 0], line=dict(dash="dot"), name="ts_pose_fb0"
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=times, y=ts_pose_fb[:, 1], line=dict(dash="dot"), name="ts_pose_fb1"
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=times, y=ts_pose_fb[:, 2], line=dict(dash="dot"), name="ts_pose_fb2"
            ),
            row=3,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=times, y=ts_pose_fb[:, 3], line=dict(dash="dot"), name="ts_pose_fb3"
            ),
            row=4,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=times, y=ts_pose_fb[:, 4], line=dict(dash="dot"), name="ts_pose_fb4"
            ),
            row=5,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=times, y=ts_pose_fb[:, 5], line=dict(dash="dot"), name="ts_pose_fb5"
            ),
            row=6,
            col=1,
        )

        fig.add_trace(
            go.Scatter(x=times, y=wrench_world[:, 0], name="wrench0"), row=1, col=2
        )
        fig.add_trace(
            go.Scatter(x=times, y=wrench_world[:, 1], name="wrench1"), row=2, col=2
        )
        fig.add_trace(
            go.Scatter(x=times, y=wrench_world[:, 2], name="wrench2"), row=3, col=2
        )
        fig.add_trace(
            go.Scatter(x=times, y=wrench_world[:, 3], name="wrench3"), row=4, col=2
        )
        fig.add_trace(
            go.Scatter(x=times, y=wrench_world[:, 4], name="wrench4"), row=5, col=2
        )
        fig.add_trace(
            go.Scatter(x=times, y=wrench_world[:, 5], name="wrench5"), row=6, col=2
        )

        fig.add_trace(
            go.Scatter(x=times, y=wrench_world_filtered[:, 0], name="wrench0_filtered"),
            row=1,
            col=2,
        )
        fig.add_trace(
            go.Scatter(x=times, y=wrench_world_filtered[:, 1], name="wrench1_filtered"),
            row=2,
            col=2,
        )
        fig.add_trace(
            go.Scatter(x=times, y=wrench_world_filtered[:, 2], name="wrench2_filtered"),
            row=3,
            col=2,
        )
        fig.add_trace(
            go.Scatter(x=times, y=wrench_world_filtered[:, 3], name="wrench3_filtered"),
            row=4,
            col=2,
        )
        fig.add_trace(
            go.Scatter(x=times, y=wrench_world_filtered[:, 4], name="wrench4_filtered"),
            row=5,
            col=2,
        )
        fig.add_trace(
            go.Scatter(x=times, y=wrench_world_filtered[:, 5], name="wrench5_filtered"),
            row=6,
            col=2,
        )

        fig.update_layout(
            height=1400, width=900, title_text="Episode " + str(episode_id)
        )
        fig.show()
        # fig.write_html('output.html')


##
## Usage:
##   Call at your desired rate: save_one_img_frame()
##   Call once: save_low_dim_for_episode()
##   Call once: save_video_to_file()
##
class EpisodeDataIncreImageBuffer(EpisodeDataBuffer):
    def __init__(self, rgb_shape_nhwc, **kwargs) -> None:
        super().__init__(**kwargs)
        self.rgb_data_id = 0
        self.video_saver = {}
        self.create_zarr_groups_for_episode(rgb_shape_nhwc)
        for camera_id in self.camera_ids:
            if self.save_video:
                self.video_saver[f"camera{camera_id}_rgb"] = imageio.get_writer(
                    f"{self.store_path}/data/episode_{self.episode_id}/camera{camera_id}_rgb.mp4",
                    fps=self.save_video_fps,
                )

    def save_one_img_frame(
        self,
        visual_observation: Dict[int, VideoData],
        visual_time_stamp: Optional[any] = None,
    ):
        episode_data = self.root["data"][f"episode_{self.episode_id}"]
        for camera_id in self.camera_ids:
            rgb_arr = episode_data[f"camera{camera_id}_rgb"]
            img_copy_single(rgb_arr, self.rgb_data_id, visual_observation[camera_id])

            if self.save_video:
                self.video_saver[f"camera{camera_id}_rgb"].append_data(
                    visual_observation[camera_id]
                )
        if visual_time_stamp is not None:
            episode_data["visual_time_stamps"][self.rgb_data_id] = visual_time_stamp
        self.rgb_data_id = self.rgb_data_id + 1
        # print(self.rgb_data_id)

    def save_video_to_file(self):
        # resize video buffer to fit actual size
        episode_data = self.root["data"][f"episode_{self.episode_id}"]
        for camera_id in self.camera_ids:
            rgb_arr = episode_data[f"camera{camera_id}_rgb"]
            shape = (self.rgb_data_id,) + rgb_arr.shape[1:]
            rgb_arr.resize(shape)
        # resize visual_time_stamps
        episode_data["visual_time_stamps"].resize(self.rgb_data_id)

        # save mp4 to file
        for camera_id in self.camera_ids:
            self.video_saver[f"camera{camera_id}_rgb"].close()
