import torch
import numpy as np
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

# define the CNN architecture
class LeNet5(nn.Module):
    def __init__(self, drop = False, decay = False, BN = False):
        super(LeNet5, self).__init__()
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
        self.dropout = nn.Dropout(0.2)
        # batch normalization
        self.BN1 = nn.BatchNorm2d(6, device=device)
        self.BN2 = nn.BatchNorm2d(16, device=device)
        self.BN3 = nn.BatchNorm2d(120, device=device)
        self.BN4 = nn.BatchNorm1d(120, device=device)
        self.BN5 = nn.BatchNorm1d(84, device=device)
        # layer modes
        self.modes = {'drop': drop, 'decay': decay, 'BN': BN}

    def forward(self, x):
        # add sequence of convolutional and max pooling layers
        if self.modes['drop']:
            x = self.pool(F.sigmoid(self.conv1(x)))
            x = self.pool(F.sigmoid(self.conv2(x)))
            x = F.sigmoid(self.conv3(x))
            # flatten image input
            x = x.view(-1, 120)
            # add dropout layer
            x = self.dropout(x)
            # add 1st hidden layer, with relu activation function
            x = F.sigmoid(self.fc1(x))
            # add dropout layer
            x = self.dropout(x)
        #elif:
        elif self.modes['BN']:
            x = self.pool(F.relu(self.BN1(self.conv1(x))))
            x = self.pool(F.relu(self.BN2(self.conv2(x))))
            x = F.relu(self.BN3(self.conv3(x)))
            # flatten image input
            x = self.BN4(x.view(-1, 120))
            # add 1st hidden layer, with relu activation function
            x = F.relu(self.BN5(self.fc1(x)))
        else:
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = F.relu(self.conv3(x))
            # flatten image input
            x = x.view(-1, 120)

            # add 1st hidden layer, with relu activation function
            x = F.relu(self.fc1(x))

        # add 2nd hidden layer, with relu activation function
        x = self.fc2(x)
        return x


def train_model(model, criterion, optimizer, train_loader, test_loader):
    # number of epochs to train the model
    n_epochs = 5

    test_loss_min = np.Inf  # track change in test loss
    train_losses, test_losses, accuracies_test, accuracies_train = [], [], [], []

    for epoch in range(1, n_epochs + 1):

        # keep track of training and test loss
        train_loss = 0.0
        test_loss = 0.0
        train_correct = 0
        test_correct = 0
        ###################
        # train the model #
        ###################

        for data, target in train_loader:
            # move tensors to GPU if CUDA is available
            data, target = data.to(device), target.to(device)
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the batch loss
            loss = criterion(output, target)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # update training loss
            train_loss += loss.item() * data.size(0)

            # get probabilities
            probs = torch.softmax(output, dim=1)
            # get top probability and class
            top_p, top_class = probs.topk(1, dim=1)
            top_p, top_class = top_p.to(device), top_class.to(device)
            equals = top_class == target.view(*top_class.shape)
            # calculate accuracy
            train_correct += equals.sum().item()

        accuracy_train = (train_correct / len(train_loader.dataset))*100
        accuracies_train.append(accuracy_train)


        ######################
        # test the model #
        ######################
        with torch.no_grad():
            model.eval()
            for data, target in test_loader:
                # move tensors to GPU if CUDA is available
                data, target = data.to(device), target.to(device)
                # forward pass: compute predicted outputs by passing inputs to the model
                output = model(data)
                # calculate the batch loss
                loss = criterion(output, target)
                # update average test loss
                test_loss += loss.item() * data.size(0)

                # get probabilities
                probs = torch.softmax(output, dim=1)
                # get top probability and class
                top_p, top_class = probs.topk(1, dim=1)
                top_p, top_class = top_p.to(device), top_class.to(device)
                equals = top_class == target.view(*top_class.shape)
                # calculate accuracy
                test_correct += equals.sum().item()

        model.train()

        # calculate average losses
        train_loss = train_loss / len(train_loader)
        test_loss = test_loss / len(test_loader)

        # print training/test statistics
        train_losses.append(train_loss)
        test_losses.append(test_loss)

        accuracy_test = (test_correct / len(test_loader.dataset))*100
        accuracies_test.append(accuracy_test)
        print("Epoch: {}/{}.. ".format(epoch, n_epochs),
              "Training Loss: {:.3f}.. ".format(train_loss),
              "Test Loss: {:.3f}.. ".format(test_loss),
              "Test Accuracy: {:.3f}%".format(accuracy_test))

        # save model if test loss has decreased
        if test_loss <= test_loss_min:
            print('Test loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                test_loss_min,
                test_loss))
            torch.save(model.state_dict(), 'model_LeNet5.pt')
            test_loss_min = test_loss
    loss_plt = plt.figure(1)
    plt.plot(train_losses, label='Training loss')
    plt.plot(test_losses, label='Validation loss')
    plt.legend(frameon=False)

    accuracy_plt = plt.figure(2)
    plt.plot(accuracies_train, label='Train accuracy')
    plt.plot(accuracies_test, label='Test accuracy')
    plt.legend(frameon=False)
    plt.show()
#%% Main
global device
if torch.cuda.is_available():
    device = torch.device('cuda')
    print('GPU available')
else:
    device = torch.device('cpu')
    print('GPU not available, training on CPU.')

#print(device)

# number of subprocesses to use for data loading
num_workers = 0
# how many samples per batch to load
batch_size = 20

# Load data
train_data = torchvision.datasets.FashionMNIST(
    root = './data/FashionMNIST',
    train = True,
    download = True,
    transform = transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize((0.5,), (0.5,))])
)

test_data = torchvision.datasets.FashionMNIST(
    root = './data/FashionMNIST',
    train = False,
    download = True,
    transform = transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize((0.5,), (0.5,))])
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
model = LeNet5(BN=True)
#print(model)

# move tensors to GPU if CUDA is available
model.to(device)

# forward pass images
output = model(images.to(device))
# print(output.size())

# specify loss function (categorical cross-entropy)
criterion = nn.CrossEntropyLoss()

# specify optimizer
optimizer = optim.Adam(model.parameters(), lr=0.003)

train_model(model, criterion, optimizer, train_loader, test_loader)



