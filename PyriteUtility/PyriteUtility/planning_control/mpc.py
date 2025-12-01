import sys
import os

sys.path.append(os.path.join(sys.path[0], ".."))  # PyriteUtility

from einops import rearrange, reduce
import time
import torch
import copy
import numpy as np
from typing import Optional, Callable

# from PyriteConfig.tasks.flip_up import flip_up_type_conversion as task
from PyriteConfig.tasks.common import common_type_conversions as common_task
from PyriteConfig.tasks.umift import umift_type_conversions as umift_task

from PyriteUtility.common import dict_apply
from PyriteUtility.data_pipeline.data_plotting import plot_ts_action, plot_js_action
import PyriteUtility.spatial_math.spatial_utilities as su


def printOrNot(verbose, *args):
    if verbose >= 0:
        print(*args)


# fmt: off
class ModelPredictiveControllerHybrid():
    """Class that maintains IO buffering of a MPC controller with hybrid policy.
    args:
        shape_meta: dict
        policy: torch.nn.Module
        action_to_trajectory: A function that converts waypoints to a continuous trajectory
        execution_horizon: MPC execution horizon in number of steps
    """
    def __init__(self,
        shape_meta,
        id_list,
        policy,
        action_to_trajectory: Callable[[np.ndarray], Callable],
        sparse_execution_horizon=10,
        dense_execution_horizon=2,
        test_sparse_action=False,
        fix_orientation=False,
        dense_execution_offset=0.0,
    ):
        print("[MPC] Initializing")
        self.shape_meta = shape_meta
        self.id_list = id_list

        action_type = ""
        if shape_meta['action']['shape'][0] == 7:
            action_type = "js"
        elif shape_meta['action']['shape'][0] == 9:
            action_type = "pose9"
        elif shape_meta['action']['shape'][0] == 10:
            action_type = "pose9g1"
        elif shape_meta['action']['shape'][0] == 19:
            action_type = "pose9pose9s1"
        elif shape_meta['action']['shape'][0] == 38:
            action_type = "pose9pose9s1"
        elif shape_meta["action"]["shape"][0] == 21:
            action_type = "pose9pose9s1a2"
        elif shape_meta["action"]["shape"][0] == 42:
            action_type = "pose9pose9s1a2"
        else:
            raise RuntimeError('unsupported')

        if action_type == "js":
            action_postprocess = common_task.actionJS_postprocess
        elif action_type == "pose9":
            action_postprocess = common_task.action9_postprocess
        elif action_type == "pose9g1":
            action_postprocess = common_task.action9g1_postprocess
        elif action_type == "pose9pose9s1":
            action_postprocess = common_task.action19_postprocess
        elif action_type == "pose9pose9s1a2":
            action_postprocess = common_task.action21_postprocess
        else:
            raise RuntimeError('unsupported')
        
        self.action_type = action_type
        self.action_postprocess = action_postprocess

        self.policy = policy
        self.action_to_trajectory = action_to_trajectory
        self.fix_orientation = fix_orientation

        # internal variables
        self.time_offset = None
        self.sparse_obs_data = {}
        self.sparse_obs_last_timestamps = {}
        self.horizon_start_time_step = -np.inf
        self.verbose_level = -1

        # for virtual target actions only
        self.sparse_target_mats = None
        self.sparse_vt_mats = None
        self.stiffness = None

        print("[MPC] Done initializing")

    def set_time_offset(self, current_time_s):
        '''
        Set time offset such that timing in this controller is aligned with hardware time.
        '''
        self.time_offset = current_time_s - time.perf_counter()

    def set_observation(self, obs_task):
        for key, attr in self.shape_meta['sample']['obs']['sparse'].items():
            data = obs_task[key]
            horizon = attr['horizon']
            down_sample_steps = attr['down_sample_steps']
            # sample 'horizon' number of latest obs from the queue
            assert len(data) >= (horizon-1) * down_sample_steps + 1
            self.sparse_obs_data[key] = data[-(horizon-1) * down_sample_steps - 1::down_sample_steps]

        # for id in self.id_list:
        #     self.sparse_obs_last_timestamps[f"rgb_time_stamps_{id}"] = obs_task[f"rgb_time_stamps_{id}"][-1]
        #     self.sparse_obs_last_timestamps[f"robot_time_stamps_{id}"] = obs_task[f"robot_time_stamps_{id}"][-1]
        #     self.sparse_obs_last_timestamps[f"wrench_time_stamps_{id}"] = obs_task[f"wrench_time_stamps_{id}"][-1]

    def compute_sparse_control(self, device):
        """ Run sparse model inference once. Does not output control.
        """
        # time_now = time.perf_counter() + self.time_offset
        # for id in self.id_list:
        #     dt_rgb = time_now - self.sparse_obs_last_timestamps[f"rgb_time_stamps_{id}"]
        #     dt_ts_pose = time_now - self.sparse_obs_last_timestamps[f"robot_time_stamps_{id}"]
        #     dt_wrench = time_now - self.sparse_obs_last_timestamps[f"wrench_time_stamps_{id}"]
        #     print(f'[MPC] obs lagging for robot {id}: dt_rgb: {dt_rgb}, dt_ts_pose: {dt_ts_pose}, dt_wrench: {dt_wrench}')

        with torch.no_grad():
            s = time.time()
            obs_sample_np = {}
            obs_sample_np['sparse'], base_ref = common_task.sparse_obs_to_obs_sample(
                obs_sparse=self.sparse_obs_data,
                shape_meta=self.shape_meta,
                reshape_mode='reshape',
                id_list=self.id_list,
                ignore_rgb=False,
            )
            # add batch dimension
            obs_sample_np = dict_apply(obs_sample_np,
                lambda x: rearrange(x, '... -> 1 ...'))
            # convert to torch tensor
            obs_sample = dict_apply(obs_sample_np,
                lambda x: torch.from_numpy(x).to(device))

            result = self.policy.predict_action(obs_sample)
            raw_action = result['sparse'][0].detach().to('cpu').numpy()
            
            if self.action_type == "js":
                action = self.action_postprocess(raw_action, base_ref)
            else:
                action = self.action_postprocess(raw_action, base_ref, self.id_list, self.fix_orientation)
            printOrNot(self.verbose_level, 'Sparse inference latency:', time.time() - s)
            return action
    
    def get_SE3_targets(self):
        return self.sparse_target_mats, self.sparse_vt_mats

# fmt: on

class ModelPredictiveController():
    """Class that maintains IO buffering of a MPC controller.
    args:
        shape_meta: dict
        id_list: list of robot ids. [0] or [0, 1]
        policy: torch.nn.Module
        action_to_trajectory: A function that converts waypoints to a continuous trajectory
        execution_horizon: MPC execution horizon in number of steps
        fix_orientation: whether to only execute xyz in the action
    """
    def __init__(self,
        shape_meta,
        id_list,
        policy,
        action_to_trajectory: Callable[[np.ndarray], Callable],
        execution_horizon=10,
        fix_orientation=False,
    ):
        print("[MPC] Initializing")
        self.shape_meta = shape_meta
        self.id_list = id_list

        action_type = "pose9" # "pose9" or "pose9pose9s1"
        if shape_meta['action']['shape'][0] == 9:
            action_type = "pose9"
        elif shape_meta['action']['shape'][0] == 10:
            action_type = "pose9g1"
        elif shape_meta['action']['shape'][0] == 19:
            action_type = "pose9pose9s1"
        elif shape_meta['action']['shape'][0] == 38:
            action_type = "pose9pose9s1"
        elif shape_meta['action']['shape'][0] == 21:
            action_type = "pose9pose9s1a2"
        elif shape_meta['action']['shape'][0] == 42:
            action_type = "pose9pose9s1a2"
        else:
            raise RuntimeError('unsupported')

        if action_type == "pose9":
            action_postprocess = umift_task.action9_postprocess
        elif action_type == "pose9g1":
            action_postprocess = umift_task.action10_postprocess
        elif action_type == "pose9pose9s1":
            action_postprocess = umift_task.action19_postprocess
        elif action_type == "pose9pose9s1a2":
            action_postprocess = umift_task.action21_postprocess
        else:
            raise RuntimeError('unsupported')
        self.action_type = action_type
        self.action_postprocess = action_postprocess

        self.policy = policy
        self.execution_horizon_time_step = execution_horizon
        self.action_to_trajectory = action_to_trajectory
        self.fix_orientation = fix_orientation

        # internal variables
        self.time_offset = None
        self.obs_data = {}
        self.obs_last_timestamps = {}
        self.SE3_WBase = None
        self.verbose_level = -1

        print("[MPC] Done initializing")


    def set_time_offset(self, current_time_s):
        '''
        Set time offset such that timing in this controller is aligned with hardware time.
        '''
        self.time_offset = current_time_s - time.perf_counter()

    def compute_one_horizon_action(self, obs_task, device):
        # sample the data per down_sample_steps and horizon
        obs_sampled = {}
        for key, attr in self.shape_meta['sample']['obs']['sparse'].items():
            data = obs_task[key]
            horizon = attr['horizon']
            down_sample_steps = attr['down_sample_steps']
            # sample 'horizon' number of latest obs from the queue
            assert len(data) >= (horizon-1) * down_sample_steps + 1
            obs_sampled[key] = data[-(horizon-1) * down_sample_steps - 1::down_sample_steps]

        # # report latency
        # time_now = time.perf_counter() + self.time_offset
        # for id in self.id_list:
        #     dt_rgb = time_now - obs_task[f"rgb_time_stamps_{id}"][-1]
        #     dt_ts_pose = time_now - obs_task[f"robot_time_stamps_{id}"][-1]
        #     dt_wrench = time_now - obs_task[f"wrench_time_stamps_{id}"][-1]
        #     dt_gripper = time_now - obs_task[f"gripper_time_stamps_{id}"][-1]
            # print(f'[MPC] obs lagging for robot {id}: dt_rgb: {dt_rgb}, dt_ts_pose: {dt_ts_pose}, dt_wrench: {dt_wrench}, dt_gripper: {dt_gripper}')

        # run inference
        with torch.no_grad():
            s = time.time()
            obs_sample_np = {}
            obs_sample_np['sparse'], SE3_WBase = umift_task.sparse_obs_to_obs_sample(
                obs_sparse=obs_sampled,
                shape_meta=self.shape_meta,
                reshape_mode='reshape',
                id_list=self.id_list,
                ignore_rgb=False,
            )
            self.SE3_WBase = SE3_WBase
            # add batch dimension
            obs_sample_np = dict_apply(obs_sample_np,
                lambda x: rearrange(x, '... -> 1 ...'))
            # convert to torch tensor
            obs_sample = dict_apply(obs_sample_np,
                lambda x: torch.from_numpy(x).to(device))

            result = self.policy.predict_action(obs_sample)
            if 'sparse' in result:
                raw_action = result['sparse'][0].detach().to('cpu').numpy()
            else:
                raw_action = result['action'][0].detach().to('cpu').numpy()
            
            action = self.action_postprocess(raw_action, SE3_WBase, self.id_list, self.fix_orientation)
            printOrNot(self.verbose_level, 'Sparse inference latency:', time.time() - s)
            return action
    