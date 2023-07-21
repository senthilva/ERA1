
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
      def __init__(self):
          super(Net, self).__init__()

          # Prep Layer
          self.prepblock = nn.Sequential(
              nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), stride=1,padding = 1),
              nn.BatchNorm2d(64),
              nn.ReLU(),
              #nn.Dropout(0.2)

          ) # output_size = 32


          #Residual blk1 R1
          self.R1 = nn.Sequential(
              nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=1,padding = 1),
              nn.BatchNorm2d(128),
              nn.ReLU(),
              nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=2,padding = 1),
              nn.BatchNorm2d(128),
              nn.ReLU(),

          ) # output


          # Layer 1
          self.convblockx1 = nn.Sequential(
              nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=1,padding = 1),
              nn.MaxPool2d(2, 2),
              nn.BatchNorm2d(128),
              nn.ReLU(),
              nn.Dropout(0.1)
          ) # output_size = 30



        # Layer 2
          self.convblockmid = nn.Sequential(
              nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=1,padding = 1),
              #nn.MaxPool2d(2, 2)
              nn.BatchNorm2d(256),
              nn.ReLU(),
              #nn.Dropout(0.1)
          ) # output_size = 30

          #Residual blk2 R2
          self.R2 = nn.Sequential(
              nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), stride=1,padding = 1),
              nn.BatchNorm2d(512),
              nn.ReLU(),
              nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=2,padding = 1),
              nn.BatchNorm2d(512),
              nn.ReLU(),

          ) # outpu

          # Layer 1
          self.convblockx2 = nn.Sequential(
              nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), stride=1,padding = 1),
              nn.MaxPool2d(2, 2),
              nn.BatchNorm2d(512),
              nn.ReLU(),
              #nn.Dropout(0.1)
          ) # output_size = 30

          self.pool = nn.MaxPool2d(4, 2)

          self.fc1 = nn.Linear(4608, 10)




      def forward(self, x):
          xin = x
          x = self.prepblock(x)
          x = self.R1(x) + self.convblockx1(x)
          x = self.convblockmid(x)
          x = self.R2(x) + self.convblockx2(x)
          x = self.pool(x)
          #print(x.shape)
          x = x.view(-1, 4608)

          x = self.fc1(x)


          return F.log_softmax(x, dim=-1)