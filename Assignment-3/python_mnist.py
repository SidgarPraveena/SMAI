from mnist import MNIST
from torch.utils.data import DataLoader, Dataset, TensorDataset
import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision
from torchvision import datasets
import torch.optim as optim
import sys
import numpy as np

mnist = MNIST(sys.argv[1])
X_train,y_train = mnist.load_training()
X_test, y_test = mnist.load_testing()
X_train = torch.FloatTensor(X_train)
X_train = X_train.to(dtype=torch.float32)
X_train = X_train.reshape(X_train.size(0), -1) 
X_train = X_train/128
y_train = torch.FloatTensor(y_train)
y_train = y_train.to(dtype=torch.long)
bs = 4
X_test = torch.FloatTensor(X_test)
X_test = X_test.to(dtype=torch.float32)
X_test = X_test.reshape(X_test.size(0), -1) 
X_test = X_test/128
y_test = torch.FloatTensor(y_test)
y_test = y_test.to(dtype=torch.long)
train_ds = TensorDataset(X_train, y_train)
train_dl = DataLoader(train_ds, batch_size=bs, drop_last=False, shuffle=True)

test_ds = TensorDataset(X_test, y_test)
test_dl = DataLoader(test_ds, batch_size=bs * 2)

loaders={}
loaders['train'] = train_dl
loaders['test'] = test_dl

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 24, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(24, 48, 5)
        self.fc1 = nn.Linear(48 * 4 * 4, 64)
        self.fc2 = nn.Linear(64, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 48 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()

def return_predictions(pred):
    pred = [j for sub in pred for j in sub]
    return pred

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


for epoch in range(10):
    running_loss = 0.0
    for i, (data,target) in enumerate(loaders['train']):
        optimizer.zero_grad()
        data = data.view(-1, 1, 28, 28)
        outputs = net(data)
        
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        if i % 2000 == 1999:   
            running_loss = 0.0

PATH = './mnist_net.pth'
torch.save(net.state_dict(), PATH)
dataiter = iter(loaders['test'])
images, labels = dataiter.next()
net = Net()
net.load_state_dict(torch.load(PATH))
images = images.view(-1, 1, 28, 28)
outputs = net(images)

correct = 0
total = 0
pred=[]
with torch.no_grad():
    for data in loaders['test']:
        images, labels = data
        images = images.view(-1,1,28,28)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        pred.append(np.array(predicted))

res=return_predictions(pred)
for i in res:
    print(i)

    