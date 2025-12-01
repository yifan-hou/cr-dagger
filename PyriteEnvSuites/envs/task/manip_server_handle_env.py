import os
import sys

SCRIPT_PATH = os.path.abspath(os.path.dirname(__file__))
PACKAGE_PATH = os.path.join(SCRIPT_PATH, "../../")
sys.path.append(os.path.join(SCRIPT_PATH, "../../../"))

import numpy as np

from PyriteEnvSuites.envs.task.manip_server_env import ManipServerEnv

class ManipServerHandleEnv(ManipServerEnv):
    """
    This class is a wrapper for the ManipServerEnv class.
    It wraps the get observation function for the stow robot with handle.

    """
    def __init__(self, *args, **kwargs):
        super(ManipServerHandleEnv, self).__init__(*args, **kwargs)

    def get_observation_from_buffer(self):
        obs = super(ManipServerHandleEnv, self).get_sparse_observation_from_buffer()
        return obs

    def start_saving_data_for_a_new_episode(self, episode_name = ""):
        self.server.start_listening_key_events()
        self.server.start_saving_data_for_a_new_episode(episode_name)

    def stop_saving_data(self):
        self.server.stop_saving_data()
        self.server.stop_listening_key_events()


    def get_episode_folder(self):
        return self.server.get_episode_folder()
    