from pickletools import float8
from turtle import forward
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np

xarr = torch.tensor([[0],[1],[2],[3],[4],[5],[6],[7],[8],[9],[10]],dtype=torch.float32)
yarr = torch.tensor([[0],[1],[4],[6],[8],[10],[12],[14],[16],[18],[20]],dtype=torch.float32)

print(xarr)

class LR(nn.Module):
    def __init__(self):
        super(LR,self).__init__()

        self.linear = nn.Linear(1,1)

    def forward(self,x):
        #print(x)
        return self.linear(x)

model = LR()

loss = nn.MSELoss()
optimzer = torch.optim.SGD(model.parameters(),lr=0.01)

epochs = 1000

for epoch in range(epochs):

    optimzer.zero_grad()

    predicted = model(xarr)

    l = loss(yarr, predicted)
    l.backward()

    optimzer.step()

    print(f'epoch : {epoch}')

test = torch.tensor([[10]], dtype=torch.float32)

output = model(test)
print(output)
