# GradCAM


## Library used

(https://github.com/senthilva/ERA1/tree/main/session11)


## Resnet model used

(https://github.com/senthilva/ERA1/blob/main/session11/models/resnet.py)



## Image Augmentation used



    ```
    
    def transform_trainv2():
    means = [0.4914, 0.4822, 0.4465]
    stds = [0.2470, 0.2435, 0.2616]
    return A.Compose(
    [
       
        A.Normalize(mean=means, std=stds, always_apply=True),
        A.PadIfNeeded(min_height=40, min_width=40, always_apply=True),
        A.RandomCrop(height=32, width=32, always_apply=True),
        A.HorizontalFlip(),
        A.CoarseDropout(max_holes=1, max_height=8, max_width=8, min_holes=1, min_height=8, min_width=8, fill_value=means),
        ToTensorV2(),
    ])

    def transform_testv2():
    means = [0.4914, 0.4822, 0.4465]
    stds = [0.2470, 0.2435, 0.2616]
    return A.Compose(
    [
        A.Normalize(mean=means, std=stds, always_apply=True),
        ToTensorV2(),
    ])
      

    ```



## Results:

   ```
   
   EPOCH: 0
Loss=-5.8687005043029785 Batch_id=97 Accuracy=24.08: 100% 98/98 [00:46<00:00,  2.10it/s]

Test set: Average loss: -7.0005, Accuracy: 3061/10000 (30.61%)

EPOCH: 1
Loss=-9.170592308044434 Batch_id=97 Accuracy=34.42: 100% 98/98 [00:40<00:00,  2.42it/s]

Test set: Average loss: -10.3728, Accuracy: 4038/10000 (40.38%)

EPOCH: 2
Loss=-12.37462043762207 Batch_id=97 Accuracy=42.88: 100% 98/98 [00:40<00:00,  2.41it/s]

Test set: Average loss: -12.8611, Accuracy: 4690/10000 (46.90%)

EPOCH: 3
Loss=-13.692671775817871 Batch_id=97 Accuracy=47.95: 100% 98/98 [00:40<00:00,  2.42it/s]

Test set: Average loss: -14.8659, Accuracy: 5093/10000 (50.93%)

EPOCH: 4
Loss=-16.96373748779297 Batch_id=97 Accuracy=51.52: 100% 98/98 [00:40<00:00,  2.42it/s]

Test set: Average loss: -17.9648, Accuracy: 5440/10000 (54.40%)

EPOCH: 5
Loss=-18.419708251953125 Batch_id=97 Accuracy=54.50: 100% 98/98 [00:40<00:00,  2.41it/s]

Test set: Average loss: -20.3402, Accuracy: 5449/10000 (54.49%)

EPOCH: 6
Loss=-20.17940902709961 Batch_id=97 Accuracy=56.32: 100% 98/98 [00:40<00:00,  2.42it/s]

Test set: Average loss: -22.1473, Accuracy: 5577/10000 (55.77%)

EPOCH: 7
Loss=-21.95830535888672 Batch_id=97 Accuracy=57.79: 100% 98/98 [00:40<00:00,  2.42it/s]

Test set: Average loss: -23.6825, Accuracy: 5909/10000 (59.09%)

EPOCH: 8
Loss=-23.732913970947266 Batch_id=97 Accuracy=59.30: 100% 98/98 [00:40<00:00,  2.42it/s]

Test set: Average loss: -25.6963, Accuracy: 5945/10000 (59.45%)

EPOCH: 9
Loss=-25.68902015686035 Batch_id=97 Accuracy=60.69: 100% 98/98 [00:40<00:00,  2.41it/s]

Test set: Average loss: -27.8724, Accuracy: 6122/10000 (61.22%)

EPOCH: 10
Loss=-27.355358123779297 Batch_id=97 Accuracy=61.69: 100% 98/98 [00:40<00:00,  2.42it/s]

Test set: Average loss: -29.6951, Accuracy: 6265/10000 (62.65%)

EPOCH: 11
Loss=-30.144060134887695 Batch_id=97 Accuracy=62.77: 100% 98/98 [00:40<00:00,  2.42it/s]

Test set: Average loss: -31.2394, Accuracy: 6292/10000 (62.92%)

EPOCH: 12
Loss=-30.645395278930664 Batch_id=97 Accuracy=63.18: 100% 98/98 [00:40<00:00,  2.40it/s]

Test set: Average loss: -33.2861, Accuracy: 6429/10000 (64.29%)

EPOCH: 13
Loss=-32.99705123901367 Batch_id=97 Accuracy=63.99: 100% 98/98 [00:40<00:00,  2.42it/s]

Test set: Average loss: -34.4486, Accuracy: 6401/10000 (64.01%)

EPOCH: 14
Loss=-33.855106353759766 Batch_id=97 Accuracy=64.73: 100% 98/98 [00:40<00:00,  2.42it/s]

Test set: Average loss: -34.5724, Accuracy: 6527/10000 (65.27%)

EPOCH: 15
Loss=-36.07720184326172 Batch_id=97 Accuracy=65.47: 100% 98/98 [00:40<00:00,  2.42it/s]

Test set: Average loss: -37.9240, Accuracy: 6524/10000 (65.24%)

EPOCH: 16
Loss=-37.08926773071289 Batch_id=97 Accuracy=65.74: 100% 98/98 [00:40<00:00,  2.42it/s]

Test set: Average loss: -39.6305, Accuracy: 6634/10000 (66.34%)

EPOCH: 17
Loss=-39.18950653076172 Batch_id=97 Accuracy=66.53: 100% 98/98 [00:40<00:00,  2.42it/s]

Test set: Average loss: -42.1811, Accuracy: 6525/10000 (65.25%)

EPOCH: 18
Loss=-41.895042419433594 Batch_id=97 Accuracy=66.80: 100% 98/98 [00:40<00:00,  2.42it/s]

Test set: Average loss: -42.7465, Accuracy: 6745/10000 (67.45%)

EPOCH: 19
Loss=-43.765907287597656 Batch_id=97 Accuracy=67.60: 100% 98/98 [00:40<00:00,  2.40it/s]

Test set: Average loss: -44.4199, Accuracy: 6823/10000 (68.23%)

   ```






