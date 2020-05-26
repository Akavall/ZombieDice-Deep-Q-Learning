
import torch 
from torch import nn 

import torch.nn.functional as F

import numpy as np
import random as rn

from collections import deque 

class RNNRegressor(nn.Module):
    def __init__(self, input_size, action_size):
        super(RNNRegressor, self).__init__()

        self.fc1 = nn.Linear(input_size, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, 24)
        self.fc4 = nn.Linear(24, action_size)

    def forward(self, input):

        fc_1_output = F.relu(self.fc1(input))
        fc_2_output = F.relu(self.fc2(fc_1_output))
        fc_3_output = F.relu(self.fc3(fc_2_output))
        fc_4_output = self.fc4(fc_3_output)

        return fc_4_output

class PyTorchAgent:

    def __init__(self, 
                 input_size, 
                 action_size, 
                 batch_size=100,
                 memories_capacity=1000):
        self.input_size = input_size
        self.action_size = action_size 
        self.model = RNNRegressor(self.input_size, self.action_size)
        self.target_model = RNNRegressor(self.input_size, self.action_size)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0002)
        self.memories = deque([], memories_capacity)
        self.batch_size = batch_size 
        self.epsilon=0.95
        self.epsilon_decay=0.995
        self.epsilon_min=0.01
        self.discount_rate=0.95

    def remember(self, memory):
        self.memories.append(memory)

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return rn.randrange(self.action_size)

        state_torch = torch.from_numpy(state).float()
        act_values = self.model(state_torch)

        return act_values.argmax()

    def replay_memory(self, episode):

        if len(self.memories) < self.batch_size:
            return 

        sample_memories = rn.sample(self.memories, self.batch_size)

        for state, action, reward, new_state, done in sample_memories:

            self.optimizer.zero_grad()

            state_pytorch = torch.from_numpy(state).float()
            action_scores = self.model(state_pytorch)
            action_scores_original = action_scores.clone() # do not detach!

            if done:
                action_scores[0][action] = reward

            else:
                if isinstance(new_state, int):
                    new_action_score = new_state
                else:
                    new_state_pytorch = torch.from_numpy(new_state).float()
                    temp = self.target_model(new_state_pytorch)
                    new_action_score = temp.max()

                # reward is zero in this case, so it doesn't matter 
                action_scores[0][action] = reward + self.discount_rate * new_action_score
                # action_scores[0][action] = self.discount_rate * new_action_score

            loss = self.criterion(action_scores_original, action_scores)
            loss.backward()
            self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_policy_weights(self):

        with torch.no_grad():
            self.target_model.load_state_dict(self.model.state_dict())
