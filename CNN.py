import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

transform= transforms.ToTensor()

train_data = datasets.MNIST(root='./cnn_data',train=True, download=True, transform=transform)
test_data = datasets.MNIST(root='./cnn_data',train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=10, shuffle=True)
test_loader = DataLoader(test_data, batch_size=10, shuffle=False)

for i,(X_train, y_train) in enumerate(train_data):
    break

class LeNet5(nn.Module): 
    def __init__(self): 
        super(LeNet5, self).__init__() 
          
        # First Convolutional Layer 
        self.conv1 = nn.Conv2d(1, 6, 4, 1) 
          
        # Max Pooling Layer 
        self.pool = nn.MaxPool2d(2, 2) 
          
        # Second Convolutional Layer 
        self.conv2 = nn.Conv2d(6, 16, 4, 1) 
          
        # First Fully Connected Layer 
        self.fc1 = nn.Linear(16 * 4 * 4, 120) 
          
        # Second Fully Connected Layer 
        self.fc2 = nn.Linear(120, 84) 
          
        # Output Layer 
        self.fc3 = nn.Linear(84, 10) 
  
    def forward(self, x): 
        # Pass the input through the first convolutional layer and activation function 
        x = self.pool(F.relu(self.conv1(x))) 
          
        # Pass the output of the first layer through  
        # the second convolutional layer and activation function 
        x = self.pool(F.relu(self.conv2(x))) 
          
        # Reshape the output to be passed through the fully connected layers 
        x = x.view(-1, 16 * 4 * 4) 
          
        # Pass the output through the first fully connected layer and activation function 
        x = F.relu(self.fc1(x)) 
          
        # Pass the output of the first fully connected layer through  
        # the second fully connected layer and activation function 
        x = F.relu(self.fc2(x)) 
          
        # Pass the output of the second fully connected layer through the output layer 
        x = self.fc3(x) 
          
        # Return the final output 
        return x
    
model = LeNet5()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

import time
start_time = time.time()

epochs = 5

train_losses = []
test_losses = []
train_correct = []
test_correct = []

for i in range(epochs):
    trn_corr=0
    tst_corr=0


    for b,(X_train, y_train) in enumerate(train_loader):
        b+=1
        y_pred = model(X_train)
        loss=criterion(y_pred,y_train)

        predicted= torch.max(y_pred.data,1)[1]
        batch_corr= (predicted==y_train).sum()
        trn_corr+=batch_corr

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        if b%600 == 0:
            print(f'Epoch: {i} Batch: {b} Loss: {loss.item()}')

    train_losses.append(loss)
    train_correct.append(trn_corr)


    with torch.no_grad():
        for b,(X_test, y_test) in enumerate(test_loader):
            y_val = model(X_test)
            predicted = torch.max(y_val.data,1)[1]
            tst_corr += (predicted==y_test).sum()

    loss = criterion(y_val,y_test)
    test_losses.append(loss)
    test_correct.append(tst_corr)


current_time = time.time()
total = current_time - start_time
print(f'Training time: {total/60}minutes')


test_load_everything = DataLoader(test_data, batch_size=10000, shuffle=False)
with torch.no_grad():
    correct=0
    for X_test, y_test in test_load_everything:
        y_val=model(X_test)
        predicted=torch.max(y_val,1)[1]
        correct +=(predicted==y_test).sum()
print(f'Accuracy: {correct.item()/len(test_data)*100}')


