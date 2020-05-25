
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
                 batch_size=32,
                 memories_capacity=1000):
        self.input_size = input_size
        self.action_size = action_size 
        self.model = RNNRegressor(self.input_size, self.action_size)
        self.target_model = RNNRegressor(self.input_size, self.action_size)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
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

        # optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        # criterion = nn.MSELoss()

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
                    # when temp came from target model, the weights were never updated..
                    new_action_score = temp.max()

                # it looks like reward + ... would double count reward
                # but reward id zero in this case, so it doesn't matter 
                action_scores[0][action] = reward + self.discount_rate * new_action_score
                # action_scores[0][action] = self.discount_rate * new_action_score

            loss = self.criterion(action_scores_original, action_scores)
            loss.backward()
            self.optimizer.step()

            print(f"action: {action}")
            print(f"action_scores: {action_scores}, action_scores_original: {action_scores_original}")
            print("fc1 grad abs sum: {}".format(self.model.fc1.weight.grad.abs().sum()))
            print("fc2 grad abs sum: {}".format(self.model.fc2.weight.grad.abs().sum()))
            print("fc3 grad abs sum: {}".format(self.model.fc3.weight.grad.abs().sum()))
            print("fc4 grad abs sum: {}".format(self.model.fc4.weight.grad.abs().sum()))

            # model weights are not being updated :(
            # gradient seems to be zero

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_policy_weights(self):

        with torch.no_grad():
            self.target_model.load_state_dict(self.model.state_dict())

        # import ipdb 
        # ipdb.set_trace()

        # with torch.no_grad():
        #     # every time model
        #     self.target_model.fc1.weight = self.model.fc1.weight
        #     self.target_model.fc2.weight = self.model.fc2.weight
        #     self.target_model.fc3.weight = self.model.fc3.weight
        #     self.target_model.fc4.weight = self.model.fc4.weight

        # import ipdb 
        # ipdb.set_trace()

        # self.target_model.fc1.weight = self.model.fc1.weight 
        # self.target_model.fc2.weight = self.model.fc2.weight 
        # self.target_model.fc3.weight = self.model.fc3.weight 
        # self.target_model.fc4.weight = self.model.fc4.weight









            



    





