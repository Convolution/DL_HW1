import torch
import numpy as np
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.nn.functional as F


# define the CNN architecture
class LENET5(nn.Module):
    def __init__(self):
        super(LENET5, self).__init__()
        # convolutional layer (sees 28x28x1 image tensor)
        self.conv1 = nn.Conv2d(1, 6, 5, stride=1, padding=2)
        # convolutional layer (sees 14x14x16 tensor)
        self.conv2 = nn.Conv2d(6, 16, 5, stride=1, padding=0)
        # convolutional layer (sees 1X1x120 tensor)
        self.conv3 = nn.Conv2d(16, 120, 5, stride=1, padding=0)
        # max pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        # linear layer (120 * 1 * 1 -> 84)
        self.fc1 = nn.Linear(1 * 1 * 120, 84)
        # linear layer (120 -> 10)
        self.fc2 = nn.Linear(84, 10)
        # dropout layer (p=0.25)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        # add sequence of convolutional and max pooling layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.conv3(x)
        # flatten image input
        x = x.view(-1, 120)
        # add dropout layer
        x = self.dropout(x)
        # add 1st hidden layer, with relu activation function
        x = F.relu(self.fc1(x))
        # add dropout layer
        x = self.dropout(x)
        # add 2nd hidden layer, with relu activation function
        x = F.relu(self.fc2(x))
        return x


# Main
if torch.cuda.is_available():
    device = torch.device('cuda')
    print('GPU available')
else:
    device = torch.device('cpu')
    print('GPU not available, training on CPU.')

print(device)
# number of subprocesses to use for data loading
num_workers = 0
# how many samples per batch to load
batch_size = 20

# Load data
train_data = torchvision.datasets.FashionMNIST(
    root = './data/FashionMNIST',
    train = True,
    download = True,
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
)

test_data = torchvision.datasets.FashionMNIST(
    root = './data/FashionMNIST',
    train = False,
    download = True,
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
)

# prepare data loaders
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=num_workers)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=num_workers)

# obtain one batch of training images
dataiter = iter(train_loader)
images, labels = dataiter.next()
images_np = images.numpy()

img = images_np[0].squeeze()
# Plot image
#plt.imshow(img, cmap='gray')
#plt.show()

# create a complete CNN
model = LENET5()
print(model)

# move tensors to GPU if CUDA is available
model.to(device)

# forward pass images
output = model(images)
print(output.size())