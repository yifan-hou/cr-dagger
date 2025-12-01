import sys
import os

SCRIPT_PATH = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(SCRIPT_PATH, "../../../"))

from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPResidual(nn.Module):
    def __init__(self, input_dim, action_dim, action_horizon, hidden_dims=None, dropout=0.0):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = []
        self.input_layer = nn.Linear(input_dim, hidden_dims[0])
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims)-1):
            self.hidden_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.output_layer = nn.Linear(hidden_dims[-1], action_dim * action_horizon)
        self.activation = nn.GELU()
        self.dropout = dropout

    def forward(self, x):
        h = self.input_layer(x)
        h = self.activation(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        for layer in self.hidden_layers:
            h = layer(h)
            h = self.activation(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.output_layer(h)
        h = h.reshape(-1, self.action_horizon, self.action_dim)
        return h
    
    def compute_loss(self, x, target):
        pred = self.forward(x)
        loss = F.mse_loss(pred, target)
        return loss
    
    