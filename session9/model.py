
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
      def __init__(self):
          super(Net, self).__init__()

          # Input Block
          self.convblock1 = nn.Sequential(
              nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(3, 3), bias=False,padding = 1),
              nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), bias=False, padding = 1),
              nn.BatchNorm2d(32),
              nn.ReLU(),
              nn.Dropout(0.2)

          ) # output_size = 32

          #skip connections
          self.skip1 = nn.Sequential(
              nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(1, 1), bias=False),

          ) # output

          # CONVOLUTION BLOCK 1
          self.convblock2 = nn.Sequential(
              nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), bias=False,stride=2),
              nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), bias=False,padding = 1),
              nn.BatchNorm2d(128),
              nn.ReLU(),
              nn.Dropout(0.1)
          ) # output_size = 30

           #skip connections
          self.skip2 = nn.Sequential(
              nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3)),

          ) # out

          # TRANSITION BLOCK 1
          self.convblock3 = nn.Sequential(
              # Dilated convolutions
              nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(3, 3), padding=0, bias=False,dilation=2),
              nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(3, 3), padding=0, bias=False),
              nn.BatchNorm2d(32),
              nn.ReLU(),
              nn.Dropout(0.1)
          ) # output_size = 12

          #self.pool1 = nn.MaxPool2d(2, 2) # output_size = 11


          # CONVOLUTION BLOCK 2
          self.convblock4 = nn.Sequential(
              # Depthwise Seperable convolution
              nn.Conv2d(32, 32, kernel_size=3, padding=1, groups=32),
              nn.Conv2d(32, 16, kernel_size=1),
              nn.BatchNorm2d(16),
              nn.ReLU(),
              nn.Dropout(0.1)
          ) # output_size = 9

          # OUTPUT BLOCK
          self.gap = nn.Sequential(
              nn.AvgPool2d(kernel_size=9)
          ) # output_size = 1

          self.convblock5 = nn.Sequential(
              nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),

          )


      def forward(self, x):
          xin = x
          x = self.convblock1(x) + self.skip1(xin)
          x = self.convblock2(x)
          x = self.convblock3(x)
          x = self.convblock4(x)
          x = self.gap(x)
          x = self.convblock5(x)

          x = x.view(-1, 10)
          return F.log_softmax(x, dim=-1)
