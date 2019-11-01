import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter

import numpy as np

from replay_buffer import BasicBuffer
from dueling_dqn_model import ConvDuelingDQN
from explore_exploit_scheduler import Explore_Exploit_Sched

class DuelingAgent:

    def __init__(self, env, learning_rate=2.5e-4, gamma=0.99, buffer_size=1000000):
        self.env = env
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.replay_buffer = BasicBuffer(max_size=buffer_size)
        self.writer = SummaryWriter()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.main_model = ConvDuelingDQN(env.observation_space.shape, env.action_space.n).to(self.device)
        self.target_model = ConvDuelingDQN(env.observation_space.shape, env.action_space.n).to(self.device)

        self.target_model.load_state_dict(self.main_model.state_dict())
        self.target_model.eval()

        self.optimizer = optim.Adam(self.main_model.parameters())
        self.explore_exploit_sched = Explore_Exploit_Sched(self.main_model, env.action_space.n)

    def get_action(self, state, frame_number, evaluation=False):
        state = torch.FloatTensor(state).float().unsqueeze(0).to(self.device)
        action = self.explore_exploit_sched.get_action(frame_number, state, self.device, evaluation)

        return action

    def compute_loss(self, batch, batch_size):

        states, actions, rewards, next_states, dones = batch
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        actions = actions.view(actions.size(0), 1)
        dones = dones.view(dones.size(0), 1)

        # First get the predicted current Q using the main network
        current_q = self.main_model.forward(states).gather(1, actions)
        next_q = self.target_model.forward(next_states)
        max_next_q = torch.max(next_q, 1)[0]
        max_next_q = max_next_q.view(max_next_q.size(0), 1)

        # Get target value with Bellmann equation. 1-done ensures only reward is used in terminal
        target_q = rewards + (self.gamma*max_next_q * (1-dones))

        #print(current_q)
        #print(target_q)

        # Loss is Hueber loss, clipped between 1 and -1
        loss = F.smooth_l1_loss(current_q, target_q.detach())

        loss_clipped = torch.clamp(loss, min=-1, max=1)

        return loss_clipped

    def run_target_update(self):
        for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data_copy_(0.01 * param + (1 - 0.01) * target_param)

    def update(self, batch_size, write_to_board, episodes):

        batch = self.replay_buffer.sample(batch_size)
        loss = self.compute_loss(batch, batch_size)

        if write_to_board:
            self.writer.add_scalar('Loss/train', loss, episodes)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
