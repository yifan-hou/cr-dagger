import numpy as np
import torch
import time
import dill
import hydra
import pickle
import time
import numpy as np
import numpy.typing as npt
from torch.utils.data import DataLoader
import robotmq as rmq

# Check and receive res network weights
# send transitions to learner

class Actor:
    def __init__(self,
                 network_server_endpoint: str,
                 network_weight_topic: str,
                 transitions_server_endpoint: str,
                 transitions_topic: str,
                 transitions_topic_expire_time_s: int):
        self.network_weight_client = rmq.RMQClient(
            client_name="network_weight_client", server_endpoint=network_server_endpoint
        )
        print("[Actor] network_weight_client created")

        self.transitions_server = rmq.RMQServer(
            server_name="transitions_server", server_endpoint=transitions_server_endpoint
        )
        self.transitions_server.add_topic(transitions_topic, transitions_topic_expire_time_s)
        print("[Actor] transitions_server created")
       
        self.network_weight_topic = network_weight_topic
        self.transitions_topic = transitions_topic

    def receive_network_weights(self, workspace):
        retrieve_start_time = time.time()
        retrieved_data, timestamp = self.network_weight_client.pop_data(topic=self.network_weight_topic, n=-1)
        retrieve_end_time = time.time()

        if retrieved_data:
            data = pickle.loads(retrieved_data[0])
            workspace.model = data["model"] 
            for key, value in data["trainable_obs_encoders"].items():
                workspace.trainable_obs_encoders[key] = value
            workspace.sparse_normalizer = data["sparse_normalizer"]

            print(
                f"[Actor] [receive_network_weights] Received data size: {len(retrieved_data[0]) / 1024**2:.3f}MB. retrieve: {retrieve_end_time - retrieve_start_time:.4f}s, load: {time.time() - retrieve_end_time:.4f}s)"
            )
            return True
        return False

    def send_transitions(self, transitions: dict):
        start_time = time.time()
        pickle_data = pickle.dumps(transitions)
        dump_end_time = time.time()
        self.transitions_server.put_data(self.transitions_topic, pickle_data)
        send_end_time = time.time()
        print(
            f"[Actor][send_transitions] Data size: {len(pickle_data) / 1024:.3f}KB. dump: {dump_end_time - start_time:.4f}s, send: {send_end_time - dump_end_time: .4f}s)"
        )
        # time.sleep(1)
