from turtle import forward
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Logistic Regression using multiple features

xarr = torch.tensor([[1],[2],[3],[4],[5],[6],[7],[8],[9]],dtype=torch.float32)
yarr = torch.tensor([[0],[0],[1],[0],[0],[1],[1],[1],[1]],dtype=torch.float32)

class logreg(nn.Module):
    def __init__(self):
        super(logreg,self).__init__()
        self.linear = nn.Linear(1,1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.linear(x)
        x = self.sigmoid(x)
        return x

model = logreg()

epochs = 100000
loss = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(epochs):
    predicted = model(xarr)

    l = loss(yarr,predicted)
    l.backward()

    optimizer.step()


pred = torch.tensor([[7]],dtype=torch.float32)

print(model(pred))