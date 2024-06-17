import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torchvision.transforms as transforms

# create fully connected network
class NN(nn.Module):                                        # inhereted from nn.Module
    def __init__(self, input_size, num_classes):
        super(NN, self).__init__()                          # super calls initializtion of parent class
        self.fc1 = nn.Linear(input_size, 50)                # fc stands for fully connected, 50 is the number of neurons in the layer
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):                                   # computes output Tensors from input Tensors, perfroming the layers that were initialized in the __init__                       
        x = F.relu(self.fc1(x))                              # calling self.fc1 on input x and applying relu
        x = self.fc2(x)                                      # calling self.fc2 on the new x
        return x
    
# model = NN(784, 10)                                        # model with 784 input size and 10 classes
# x = torch.randn(64, 784)                                   # 64 is batch size (examples run simulaniously) and 784 features
# print(model(x).shape)                                      # print the shape of the output, should be 64x10 (10 values for each example (the 10 numbers are probabilities of the example being each of the 10 classes))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

# hyperparameters
input_size = 784
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 1

# load data
# root sets possible directory for new data, if it is already there it does not download, but otherwise it does
# making the data into the tensor format from numpy array
train_dataset = MNIST(root='dataset/', train=True, transform=transforms.ToTensor(), download=True)

# load the data in batches, after going through an epoch, it shuffles the data, different sets are used for each epoch
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# same for test
test_dataset = MNIST(root='dataset/', train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

# initialize network
# to device is used to move the model to the gpu if available
model = NN(input_size=input_size, num_classes=num_classes).to(device)

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# train network
# for each epoch, we go through all the images in the data
for epoch in range(num_epochs):
    # go through each batch, want to see which batch indec
    for batch_idx, (data, targets), in enumerate(train_loader):

        # using the gpu if available
        data - data.to(device=device)
        targets = targets.to(device=device)

        # 64x1x28x28, 64 images (examples), 1 channel (black and white, if color then rgb=3), 28x28 pixels (height and width)
        # print(data.shape)    

        # want to unroll the data so that it is just one long vector fo 28x28=784
        # data.shape[0] is the batch size (keeps it 64), -1 is the rest of the dimensions (will do 1X28x28=784)
        data = data.reshape(data.shape[0], -1)    

        # forward
        scores = model(data)

        # answer model gives and actual answer, automatic in tf
        loss = criterion(scores, targets) 

        # backward
        optimizer.zero_grad()    # set all gradients to zero at the end of each batch
        loss.backward()          # calcuate weights/gradient

        # gradient descent or adam step
        optimizer.step()         # update the weights


# check accuracy 
def check_accuracy(loader, model):
    if loader.dataset.train:
        print("Checking accuracy on training data")

    else:
        print("Checking accuracy on test data")


    num_correct = 0
    num_samples = 0
    model.eval()                   # set model to evaluate because it might impact calculations

    with torch.no_grad():          # don't calculate gradients with evaluation
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            x = x.reshape(x.shape[0], -1)

            scores = model(x)

            # 64x10, because 10 possible classes for each of the 64 examples
            # want to find the max possibility for each example because that is the predicted class
            # do not want the probability, just the class (1-10)
            _, predictions = scores.max(1)

            # prediction that are = to the correct label, take sum
            num_correct += (predictions == y).sum()

            # gives 64, the number of examples
            num_samples += predictions.size(0)

            # print prediction if index is 0
            if num_samples == 0:
                print("prediction:", predictions)

        # making tensor into a floats
        print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}')
        acc = float(num_correct) / float(num_samples)
            
    model.train()                  # set model back to training mode
    return acc

check_accuracy(train_loader, model)
check_accuracy(test_loader, model)





