import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd

dataset = pd.read_csv('./data/station_hour.csv')
dataset.dropna(inplace=True)
dataset.drop(["AQI_Bucket", "StationId","Datetime"], axis=1, inplace=True)

class MLEDataset(Dataset):
    def __init__(self,data):
        self.len = data.shape[0]

        self.x = torch.tensor(data.iloc[:, 0:-1].values,dtype=torch.float32)
        self.y = torch.tensor(data.iloc[:, [-1]].values,dtype=torch.float32)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.y)

dataset = MLEDataset(dataset)
train_loader = DataLoader(dataset=dataset,
                          batch_size=32,
                          shuffle=True,
                          num_workers=2)

class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()

        self.linear = nn.Linear(len(list(dataset)),1)

    def forward(self,x):
        return self.linear(x)

model = Model()
epochs = 10000
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(epochs):
    for i,(feature,output) in enumerate(train_loader):
        optimizer.zero_grad()
        
        predicted = model(feature)
        loss = criterion(output,predicted)

        loss.backward()
        optimizer.step()
