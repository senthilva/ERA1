## Iteration 1 - reduce paramaters 


### Target

- Reduce the # of parameters ; added 1x1 and removed FC
- Update epcochs to 15
- 

### Results

- Training accuracy was 94.41
- Val accuracy was 74.32
- 7998 parameters

### Analysis

- Still could not hit 99+% in training
- very low Val accuracy
- Definetely over fitting
- RF 14

## Iteration 2 - Improve training accuracy and remain within 8K parameters


### Target
- Review the RF and ensure it close to 24 and update model to reflect that
- Use a stride of 2 for first convolution

### Results

- Training accuracy was 98.74
- Val accuracy was 92.19
- 7998 parameters

### Analysis
- Increasing the RF by adding stride to first convolution made huge difference
- More epochs i think we would have hit the goal of 99.4%


## Iteration 3 - There was a Batch Normalization(BN) and RELU near the final layer.Fixed that.
Enabled dropout to tackle overfitting

### Target
- Expecting Val accuracy to improve > 92.19
- Train accuracy to > 99

### Results

- Training accuracy was 96.16
- Val accuracy was 91.52
- 7978parameters

### Analysis

- Maybe did too much regularization ( BN + Dropout)
- End image size is too small 
- Maybe considering adding padding

## Iteration 4 - Reduce dropout + add image augmentation


### Target
- Reduced dropout
- Added image augmentation
- Expected val accuracy to increase

### Results

- Training accuracy was 96.44
- Val accuracy was 92.72
- 7978parameters

### Analysis

- Still not able to meet the required accuracy under constraints
