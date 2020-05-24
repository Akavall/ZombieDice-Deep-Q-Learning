

from pytorch_agent import RNNRegressor

import torch 
from torch import nn

import numpy as np


train_data = [[0,0,0,0,6,4,3], [4,2,4,0,0,4,3], [7,2,4,0,0,1,3]]
labels = [[0, 2], [4, 2], [7, 4]]

train_data_torch = torch.from_numpy(np.array(train_data)).float()
labels_torch = torch.from_numpy(np.array(labels)).float()

my_model = RNNRegressor(7, 2)
optimizer = torch.optim.Adam(my_model.parameters(), lr=0.01)
criterion = nn.MSELoss()

for epoch in range(10):
    for i in range(len(train_data)):
        optimizer.zero_grad()

        pred = my_model(train_data_torch[i])

        loss = criterion(pred, labels_torch[i])
        loss.backward()
        optimizer.step()

        import ipdb 
        ipdb.set_trace()

# This looks like it does what it is supposed to do
# and least gradient is not all zeros and it weights are 
# being updated 