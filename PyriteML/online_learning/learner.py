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

class Learner:
    def __init__(self,
                 network_server_endpoint: str,
                 network_weight_topic: str,
                 transitions_server_endpoint: str,
                 transitions_topic: str,
                 network_weight_expire_time_s: int):
        self.network_weight_server = rmq.RMQServer(
            server_name="network_weight_server", server_endpoint=network_server_endpoint
        )
        self.network_weight_server.add_topic(network_weight_topic, network_weight_expire_time_s)
        print("[Learner] network_weight_server created")
        self.transitions_client = rmq.RMQClient(
            client_name="transitions_client", server_endpoint=transitions_server_endpoint
        )
        print("[Learner] transitions_client created")

        self.network_weight_topic = network_weight_topic
        self.transitions_topic = transitions_topic
    
    ### payloads: {"model": model, "sparse_normalizer": sparse_normalizer}
    def send_network_weights(self, payloads: dict):
        start_time = time.time()
        pickle_data = pickle.dumps(payloads)
        dump_end_time = time.time()
        self.network_weight_server.put_data(self.network_weight_topic, pickle_data)
        send_end_time = time.time()
        
        print(
            f"[Learner] [send_network_weights] Data size: {len(pickle_data) / 1024**2:.3f}MB. dump: {dump_end_time - start_time:.4f}s, send: {send_end_time - dump_end_time: .4f}s)"
        )
    
    def receive_transitions(self):
        retrieve_start_time = time.time()
        retrieved_data, timestamp = self.transitions_client.pop_data(topic=self.transitions_topic, n=1)
        retrieve_end_time = time.time()
        
        if retrieved_data:
            transitions = pickle.loads(retrieved_data[0])
            
            print(
                f"[Learner] [receive_transitions] Received data size: {len(retrieved_data[0]) / 1024**2:.3f}MB. retrieve: {retrieve_end_time - retrieve_start_time:.4f}s, load: {time.time() - retrieve_end_time:.4f}s)"
            )
            return transitions
        return None