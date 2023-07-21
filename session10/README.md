# Convergence using 1 Cycle LR for CIFAR 10

The goal here is to write a custom ResNet architecture for CIFAR10 to achive 90+ accuracy in less than 24 epochs using 1 cycle LR strategy

## Custom Network built

```
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
```

## Network Summary

Below is the network we have used as a baseline.

```
 ----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 64, 32, 32]           1,792
       BatchNorm2d-2           [-1, 64, 32, 32]             128
              ReLU-3           [-1, 64, 32, 32]               0
            Conv2d-4          [-1, 128, 32, 32]          73,856
       BatchNorm2d-5          [-1, 128, 32, 32]             256
              ReLU-6          [-1, 128, 32, 32]               0
            Conv2d-7          [-1, 128, 16, 16]         147,584
       BatchNorm2d-8          [-1, 128, 16, 16]             256
              ReLU-9          [-1, 128, 16, 16]               0
           Conv2d-10          [-1, 128, 32, 32]          73,856
        MaxPool2d-11          [-1, 128, 16, 16]               0
      BatchNorm2d-12          [-1, 128, 16, 16]             256
             ReLU-13          [-1, 128, 16, 16]               0
          Dropout-14          [-1, 128, 16, 16]               0
           Conv2d-15          [-1, 256, 16, 16]         295,168
      BatchNorm2d-16          [-1, 256, 16, 16]             512
             ReLU-17          [-1, 256, 16, 16]               0
           Conv2d-18          [-1, 512, 16, 16]       1,180,160
      BatchNorm2d-19          [-1, 512, 16, 16]           1,024
             ReLU-20          [-1, 512, 16, 16]               0
           Conv2d-21            [-1, 512, 8, 8]       2,359,808
      BatchNorm2d-22            [-1, 512, 8, 8]           1,024
             ReLU-23            [-1, 512, 8, 8]               0
           Conv2d-24          [-1, 512, 16, 16]       1,180,160
        MaxPool2d-25            [-1, 512, 8, 8]               0
      BatchNorm2d-26            [-1, 512, 8, 8]           1,024
             ReLU-27            [-1, 512, 8, 8]               0
        MaxPool2d-28            [-1, 512, 3, 3]               0
           Linear-29                   [-1, 10]          46,090
================================================================
Total params: 5,362,954
Trainable params: 5,362,954
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 14.29
Params size (MB): 20.46
Estimated Total Size (MB): 34.75
----------------------------------------------------------------

```

## Image Augmentation used



    ```
    train_transform = A.Compose(
    [
        A.RandomCrop(32,32,always_apply=True),
        A.HorizontalFlip(p=0.5),
        A.Cutout(num_holes=1, max_h_size=8, max_w_size=8, fill_value=0, always_apply=False, p=0.5),
        A.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ToTensorV2(),
    ]
)

test_transform = A.Compose(
   [
        A.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ToTensorV2(),
    ]
)
    ```

## 1 Cycle LR used

    ```

        model =  Net().to(device)
                        EPOCHS = 24
                        criterion = nn.CrossEntropyLoss()
                        optimizer = optim.Adam(model.parameters(), lr=1e-3,weight_decay=1e-2)
                        scheduler = OneCycleLR(optimizer, max_lr=1.91E-03,
                                            pct_start = 5/EPOCHS,
                                            steps_per_epoch=len(train_loader),
                                            div_factor = 100,
                                            final_div_factor = 100,
                                            three_phase = False,
                                            epochs=EPOCHS)
    ```

![image](https://github.com/senthilva/ERA1/assets/8141261/34ae5e78-a244-4b7d-9889-67b068abd993)


## Results:

   ```
   EPOCH: 0
Loss=1.581998348236084 Batch_id=97 Accuracy=35.27: 100%|██████████| 98/98 [01:04<00:00,  1.51it/s]

Test set: Average loss: 1.4688, Accuracy: 4799/10000 (47.99%)

EPOCH: 1
Loss=1.2134969234466553 Batch_id=97 Accuracy=51.01: 100%|██████████| 98/98 [01:06<00:00,  1.47it/s]

Test set: Average loss: 1.2797, Accuracy: 5537/10000 (55.37%)

EPOCH: 2
Loss=1.2006698846817017 Batch_id=97 Accuracy=56.83: 100%|██████████| 98/98 [01:06<00:00,  1.47it/s]

Test set: Average loss: 1.1622, Accuracy: 5914/10000 (59.14%)

EPOCH: 3
Loss=1.1510486602783203 Batch_id=97 Accuracy=61.17: 100%|██████████| 98/98 [01:06<00:00,  1.47it/s]

Test set: Average loss: 1.0837, Accuracy: 6219/10000 (62.19%)

EPOCH: 4
Loss=1.048377513885498 Batch_id=97 Accuracy=64.55: 100%|██████████| 98/98 [01:06<00:00,  1.47it/s]

Test set: Average loss: 1.0170, Accuracy: 6476/10000 (64.76%)

EPOCH: 5
Loss=0.9011110663414001 Batch_id=97 Accuracy=67.26: 100%|██████████| 98/98 [01:06<00:00,  1.47it/s]

Test set: Average loss: 0.9664, Accuracy: 6611/10000 (66.11%)

EPOCH: 6
Loss=0.857053279876709 Batch_id=97 Accuracy=69.54: 100%|██████████| 98/98 [01:06<00:00,  1.47it/s]

Test set: Average loss: 0.9201, Accuracy: 6828/10000 (68.28%)

EPOCH: 7
Loss=0.8335208892822266 Batch_id=97 Accuracy=71.54: 100%|██████████| 98/98 [01:06<00:00,  1.47it/s]

Test set: Average loss: 0.8779, Accuracy: 6922/10000 (69.22%)

EPOCH: 8
Loss=0.7967602610588074 Batch_id=97 Accuracy=73.49: 100%|██████████| 98/98 [01:06<00:00,  1.47it/s]

Test set: Average loss: 0.8468, Accuracy: 7051/10000 (70.51%)

EPOCH: 9
Loss=0.712113082408905 Batch_id=97 Accuracy=75.00: 100%|██████████| 98/98 [01:06<00:00,  1.46it/s]

Test set: Average loss: 0.8213, Accuracy: 7144/10000 (71.44%)

EPOCH: 10
Loss=0.7555771470069885 Batch_id=97 Accuracy=76.28: 100%|██████████| 98/98 [01:06<00:00,  1.47it/s]

Test set: Average loss: 0.7994, Accuracy: 7200/10000 (72.00%)

EPOCH: 11
Loss=0.7216870188713074 Batch_id=97 Accuracy=77.71: 100%|██████████| 98/98 [01:06<00:00,  1.47it/s]

Test set: Average loss: 0.7798, Accuracy: 7310/10000 (73.10%)

EPOCH: 12
Loss=0.6423915028572083 Batch_id=97 Accuracy=78.91: 100%|██████████| 98/98 [01:06<00:00,  1.47it/s]

Test set: Average loss: 0.7560, Accuracy: 7429/10000 (74.29%)

EPOCH: 13
Loss=0.6668161749839783 Batch_id=97 Accuracy=80.03: 100%|██████████| 98/98 [01:06<00:00,  1.46it/s]

Test set: Average loss: 0.7513, Accuracy: 7428/10000 (74.28%)

EPOCH: 14
Loss=0.58458012342453 Batch_id=97 Accuracy=81.19: 100%|██████████| 98/98 [01:06<00:00,  1.47it/s]

Test set: Average loss: 0.7260, Accuracy: 7490/10000 (74.90%)

EPOCH: 15
Loss=0.557901918888092 Batch_id=97 Accuracy=82.13: 100%|██████████| 98/98 [01:06<00:00,  1.46it/s]

Test set: Average loss: 0.7114, Accuracy: 7577/10000 (75.77%)

EPOCH: 16
Loss=0.5526317358016968 Batch_id=97 Accuracy=82.99: 100%|██████████| 98/98 [01:06<00:00,  1.46it/s]

Test set: Average loss: 0.6991, Accuracy: 7612/10000 (76.12%)

EPOCH: 17
Loss=0.5495215058326721 Batch_id=97 Accuracy=83.99: 100%|██████████| 98/98 [01:06<00:00,  1.47it/s]

Test set: Average loss: 0.6843, Accuracy: 7669/10000 (76.69%)

EPOCH: 18
Loss=0.49213868379592896 Batch_id=97 Accuracy=84.94: 100%|██████████| 98/98 [01:06<00:00,  1.47it/s]

Test set: Average loss: 0.6888, Accuracy: 7653/10000 (76.53%)

EPOCH: 19
Loss=0.5226911306381226 Batch_id=97 Accuracy=85.87: 100%|██████████| 98/98 [01:06<00:00,  1.47it/s]

Test set: Average loss: 0.6607, Accuracy: 7763/10000 (77.63%)

EPOCH: 20
Loss=0.4739624559879303 Batch_id=97 Accuracy=86.50: 100%|██████████| 98/98 [01:06<00:00,  1.47it/s]

Test set: Average loss: 0.6547, Accuracy: 7772/10000 (77.72%)

EPOCH: 21
Loss=0.4301646053791046 Batch_id=97 Accuracy=87.43: 100%|██████████| 98/98 [01:06<00:00,  1.47it/s]

Test set: Average loss: 0.6476, Accuracy: 7785/10000 (77.85%)

EPOCH: 22
Loss=0.3992057144641876 Batch_id=97 Accuracy=88.29: 100%|██████████| 98/98 [01:06<00:00,  1.47it/s]

Test set: Average loss: 0.6380, Accuracy: 7825/10000 (78.25%)

EPOCH: 23
Loss=0.4293695092201233 Batch_id=97 Accuracy=88.94: 100%|██████████| 98/98 [01:06<00:00,  1.47it/s]

Test set: Average loss: 0.6329, Accuracy: 7837/10000 (78.37%)

   ```
   

