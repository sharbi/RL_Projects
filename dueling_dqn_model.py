import random
import numpy as np

import torch
import torch.nn as nn
import torch.autograd as autograd


class ConvDuelingDQN(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(ConvDuelingDQN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.conv = nn.Sequential(
            nn.Conv2d(input_dim[0], 32, kernel_size=(8, 8), stride=4),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(4,4), stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(3,3), stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.fc_input_dim = self.feature_size()

        self.adv_stream = nn.Sequential(
            nn.Linear(self.fc_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, self.output_dim)
        )

        self.v_stream = nn.Sequential(
            nn.Linear(self.fc_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, state):
        features = self.conv(state)
        features = features.view(features.size(0), -1)
        values = self.v_stream(features)
        advantages = self.adv_stream(features)
        qvals = values + (advantages - advantages.mean())

        return qvals

    def feature_size(self):
        return self.conv(autograd.Variable(torch.zeros(1, *self.input_dim))).view(1, -1).size(1)
