import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
import numpy as np
import PIL
from PIL import Image

# used for creating own dataset w/ values
"""# CREATE OWN DATASET
class CNNDataset(Dataset):
    def __init__(self):
        pass
    def __getitem__(self,index):
        pass
    def __len__(self):
        pass"""


device = torch.device("cpu:0" if torch.cuda.is_available() else "cpu")

img = Image.open('./data/brain_tumour_data/test/no/no11.jpg')
img = np.array(img)

tempconvert = transforms.Compose([
        transforms.ToTensor()
    ])

img = tempconvert(img)
mean, std = img.mean([1,2]), img.std([1,2])

data_transform = transforms.Compose([
        #transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Resize((225,225)),
        transforms.Normalize((mean), (std)) # essentially removes noise, increases clarity
    ])

train_loader = DataLoader(ImageFolder('./data/brain_tumour_data/train',transform=data_transform),batch_size=12,shuffle=True)
test_loader = DataLoader(ImageFolder('./data/brain_tumour_data/test',transform=data_transform),batch_size=12,shuffle=True)

# input channels, output channels, filter size
class ConvolutionalNeuralNetworks(nn.Module):
    def __init__(self):
        super(ConvolutionalNeuralNetworks, self).__init__()
        
        # formula to calculate the dimensions of the matrix --> 
        self.conv = nn.Conv2d(3, 12, 6)
        self.conv2 = nn.Conv2d(12, 32, 5)
        self.maxpool = nn.MaxPool2d(2,2)

        self.linear = nn.Linear(89888, 24)
        self.linear2 = nn.Linear(24,12)
        self.linear3 = nn.Linear(12,2)

        self.dropout = nn.Dropout2d(0.25)
    
    def forward(self,x):
        #print(x.size(), "in forward1")
        x = F.relu(self.conv(x))
        x = self.maxpool(x)
        x = self.dropout(x)

        #print(x.size(), "in forward2")
        x = F.relu(self.conv2(x))
        x = self.maxpool(x)
        x = self.dropout(x)

        #print(x.size(), "in forward3")
        x = x.view(x.size(0),-1)    

        #print(x.size(), "in forward4")
        x = F.relu(self.linear(x))

        #print(x.size(), "in forward5")
        x = F.relu(self.linear2(x))

        #print(x.size(), "in forward6")
        return self.linear3(x)


model = ConvolutionalNeuralNetworks().to(device)

lr = 0.01
epochs = 10000
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = lr)

# training loop
print('hi')
for epoch in range(epochs):
    for i, (image, label) in enumerate(train_loader):
        image = image.to(device)
        label = label.to(device)

        optimizer.zero_grad()

        predicted = model(image)

        loss = criterion(predicted, label)
        loss.backward()
        optimizer.step()

        if epoch % 2 == 0:
            print(f"epoch : {epoch}, loss : {loss.item():3f}")


for epoch in range(epochs):
    for i, (image, label) in enumerate(test_loader):
        optimizer.zero_grad()

        predicted = model(image)
        label = torch.tensor(label)

        loss = criterion(label.long(), predicted)
        loss.backward()
        optimizer.step()


