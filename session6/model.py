'''
This defines the model architecture
    - 2 Conv2d
    - MP
    - 1 Conv2d
    - MP
    - 1  GAP
    - 1 linear layer
'''

import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    #This defines the structure of the NN.
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(
                                nn.Conv2d(1, 8, kernel_size=3),
                                nn.BatchNorm2d(8),
                                nn.Dropout(0.02),
                                nn.ReLU()

                              )
        self.conv2 = nn.Sequential(
                                nn.Conv2d(8, 16, kernel_size=3),
                                nn.BatchNorm2d(16),
                                nn.Dropout(0.02),
                                nn.ReLU(),
                                nn.MaxPool2d(kernel_size=2, stride=2)
                              )
        self.conv3 = nn.Sequential(
                                nn.Conv2d(16, 32, kernel_size=3),
                                nn.BatchNorm2d(32),
                                nn.Dropout(0.02),
                                nn.ReLU(),
                                nn.MaxPool2d(kernel_size=2, stride=2)
                              )
        self.conv4 = nn.Sequential(
                                nn.Conv2d(32, 40, kernel_size=3),
                                nn.BatchNorm2d(40)
                                #nn.Dropout(0.05),
                                #nn.ReLU(),
                                #nn.MaxPool2d(kernel_size=2, stride=2),
                                #nn.Conv2d(24, 16, kernel_size=3)
                                
                              )
        self.gap =  nn.AdaptiveAvgPool2d(1) # nn.AvgPool2d(8)
        self.fc = nn.Linear(40, 10)
        #self.fc2 = nn.Linear(64, 10)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.gap(x)
        x = x.view(-1, 40)
        #x = F.relu(self.fc1(x))
        x = self.fc(x)
        #x = self.fc2(x)
        return F.log_softmax(x, dim=1)
