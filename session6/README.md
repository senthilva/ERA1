# Part 1 - Back propogation

The objective is train a neural network in excel and show back propogation working on weights

# Proof of work


## Network considered

![image](https://github.com/senthilva/ERA1/assets/8141261/13ce23a6-6f1b-42ba-880e-9edc424fc1c9)



# Calculations

* h1 = w1*i1 + w2*i2		
* h2 = w3*i1 + w4*i2		
* a_h1 = σ(h1) = 1/(1 + exp(-h1))		
* a_h2 = σ(h2)= 1/(1 + exp(-h2))		
* o1 = w5*a_h1 + w6*a_h2		
* o2 = w7*a_h1 + w8*a_h2		
* a_o1 = σ(o1) =  1/(1 + exp(-o1))		
* a_o2 = σ(o2) = 1/(1 + exp(-o2))		
* E_total = E1 + E2		
* E1 = ½ * (t1 - a_o1)²		
* E2 = ½ * (t2 - a_o2)²		


* ∂E_total/∂w5 = ∂(E1 + E2)/∂w5					
* ∂E_total/∂w5 = ∂E1/∂w5 ( E2 not dependent on w5)					
* ∂E_total/∂w5 = ∂E1/∂w5 = ∂E1/∂a_o1*∂a_o1/∂o1*∂o1/∂w5					
* ∂E1/∂a_o1 =  ∂(½ * (t1 - a_o1)²)/∂a_o1 = (a_01 - t1)					
* ∂a_o1/∂o1 =  ∂(σ(o1))/∂o1 = a_o1 * (1 - a_o1)					
* ∂o1/∂w5 = a_h1				

*
* ðE_total/ðw5 = ð(E1+E2)/ðw5	
* ðE_total/ðw5 = ð(E1)/ðw5	
* ðE_total/ðw5 = ð(E1)/ðw5= ðE1/ða_o1*ða_o1/ðo1*ðo1/ðw5	
* ðE1/ða_o1 = ð(1/2*(t1-a_o1)^2)/ða_o1 = -1(t1-a_o1) = a_01-t1	
* ða_o1/ðo1 = ð(1/(1+ exp(-o1)))/ðo1 = a_o1*(1-ao1)	
* ðo1/ðw5 = a_h1	
*
* ∂E_total/∂w5 = (a_01 - t1) * a_o1 * (1 - a_o1) *  a_h1					
* ∂E_total/∂w6 = (a_01 - t1) * a_o1 * (1 - a_o1) *  a_h2					
* ∂E_total/∂w7 = (a_02 - t2) * a_o2 * (1 - a_o2) *  a_h1					
* ∂E_total/∂w8 = (a_02 - t2) * a_o2 * (1 - a_o2) *  a_h2				

* 		
* ∂E1/∂a_h1 = (a_01 - t1) * a_o1 * (1 - a_o1) * w5								
* ∂E2/∂a_h1 = (a_02 - t2) * a_o2 * (1 - a_o2) * w7								
* ∂E_total/∂a_h1 = (a_01 - t1) * a_o1 * (1 - a_o1) * w5 +  (a_02 - t2) * a_o2 * (1 - a_o2) * w7								
* ∂E_total/∂a_h2 = (a_01 - t1) * a_o1 * (1 - a_o1) * w6 +  (a_02 - t2) * a_o2 * (1 - a_o2) * w8						

* ∂E_total/∂w1 = ∂E_total/∂a_h1 * ∂a_h1/∂h1 * ∂h1/∂w1					
* ∂E_total/∂w2 = ∂E_total/∂a_h1 * ∂a_h1/∂h1 * ∂h1/∂w2					
* ∂E_total/∂w3 = ∂E_total/∂a_h2 * ∂a_h2/∂h2 * ∂h2/∂w3				

* ∂E_total/∂w1 = ((a_01 - t1) * a_o1 * (1 - a_o1) * w5 +  (a_02 - t2) * a_o2 * (1 - a_o2) * w7) * a_h1 * (1 - a_h1) * i1												
* ∂E_total/∂w2 = ((a_01 - t1) * a_o1 * (1 - a_o1) * w5 +  (a_02 - t2) * a_o2 * (1 - a_o2) * w7) * a_h1 * (1 - a_h1) * i2												
* ∂E_total/∂w3 = ((a_01 - t1) * a_o1 * (1 - a_o1) * w6 +  (a_02 - t2) * a_o2 * (1 - a_o2) * w8) * a_h2 * (1 - a_h2) * i1												
* ∂E_total/∂w4 = ((a_01 - t1) * a_o1 * (1 - a_o1) * w6 +  (a_02 - t2) * a_o2 * (1 - a_o2) * w8) * a_h2 * (1 - a_h2) * i2									

							

# Excel Calculations

![image](https://github.com/senthilva/ERA1/assets/8141261/06af030d-2141-409a-a33a-fce9de025d01)




## Error vs Lr - 0.1

![image](https://github.com/senthilva/ERA1/assets/8141261/b11d7b0d-32f4-4a98-be5b-8f00632dcfac)

## Error vs Lr - 0.2

![image](https://github.com/senthilva/ERA1/assets/8141261/9c074b41-5deb-4a93-9c01-c873debbda9f)


## Error vs Lr - 0.5

![image](https://github.com/senthilva/ERA1/assets/8141261/abf3a99b-090e-4da0-9fa5-c4e8d02910bb)


## Error vs Lr - 0.8

![image](https://github.com/senthilva/ERA1/assets/8141261/21ce4a0b-d69b-423d-8932-817d166137cc)


## Error vs LR - 1

![image](https://github.com/senthilva/ERA1/assets/8141261/9b3a4820-0a6f-4e84-97c8-7321d3aaf912)

## Error vs LR - 2

![image](https://github.com/senthilva/ERA1/assets/8141261/da126395-5338-42a8-84f1-bd1ac38570be)




# Error vs LR graph



## Observations
 
* As LR increases it converges faster - as it takes larger steps 



# Part 2 


# MNIST

The objective is achieve a test accuracy > 99.4% with less than 20K parameters within 20 epochs.


## Steps taken

### Iteration 1 - reduce paramaters 

- Reduced parameters < 19k by changing kernels and reducing linear layers
- Accuracy was ~12% after 5 spochs 
- Inference was not learning 

### Iteration 2 - increase the kernels and include GAP
- Added Global average pooling
- Size of model further reduced 
- Accuracy improved to 40%
- Validation accuracy was not increasing - potentially overfitting

### Iteration 3 - 

- Add Batch Normalization, Dropout
- 20 epochs reached 86% val accuracy


### Iteration 4

- Reduced dropout 0.02



# Network Design



  ```
   ----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1            [-1, 8, 26, 26]              80
       BatchNorm2d-2            [-1, 8, 26, 26]              16
           Dropout-3            [-1, 8, 26, 26]               0
              ReLU-4            [-1, 8, 26, 26]               0
            Conv2d-5           [-1, 16, 24, 24]           1,168
       BatchNorm2d-6           [-1, 16, 24, 24]              32
           Dropout-7           [-1, 16, 24, 24]               0
              ReLU-8           [-1, 16, 24, 24]               0
         MaxPool2d-9           [-1, 16, 12, 12]               0
           Conv2d-10           [-1, 32, 10, 10]           4,640
      BatchNorm2d-11           [-1, 32, 10, 10]              64
          Dropout-12           [-1, 32, 10, 10]               0
             ReLU-13           [-1, 32, 10, 10]               0
        MaxPool2d-14             [-1, 32, 5, 5]               0
           Conv2d-15             [-1, 40, 3, 3]          11,560
      BatchNorm2d-16             [-1, 40, 3, 3]              80
AdaptiveAvgPool2d-17             [-1, 40, 1, 1]               0
           Linear-18                   [-1, 10]             410
================================================================
Total params: 18,050
Trainable params: 18,050
Non-trainable params: 0
----------------------------------------------------------------
  ```
 
## Training and Loss

* Number of epochs : 20
* Loss - Negative log likehood
* batch_size = 512
* SGB optimizer used

  ```
 Adjusting learning rate of group 0 to 1.0000e-02.
Epoch 1
Train: Loss=1.5075 Batch_id=19 Accuracy=44.47: 100%|██████████| 20/20 [00:02<00:00,  7.78it/s]
Test set: Average loss: 2.0254, Accuracy: 11158/60000 (18.60%)

Adjusting learning rate of group 0 to 1.0000e-02.
Epoch 2
Train: Loss=0.8810 Batch_id=19 Accuracy=76.81: 100%|██████████| 20/20 [00:02<00:00,  8.53it/s]
Test set: Average loss: 1.1745, Accuracy: 39934/60000 (66.56%)

Adjusting learning rate of group 0 to 1.0000e-02.
Epoch 3
Train: Loss=0.5525 Batch_id=19 Accuracy=89.09: 100%|██████████| 20/20 [00:02<00:00,  8.27it/s]
Test set: Average loss: 0.8286, Accuracy: 46259/60000 (77.10%)

Adjusting learning rate of group 0 to 1.0000e-02.
Epoch 4
Train: Loss=0.3714 Batch_id=19 Accuracy=93.07: 100%|██████████| 20/20 [00:02<00:00,  8.67it/s]
Test set: Average loss: 0.6918, Accuracy: 48118/60000 (80.20%)

Adjusting learning rate of group 0 to 1.0000e-02.
Epoch 5
Train: Loss=0.2359 Batch_id=19 Accuracy=94.73: 100%|██████████| 20/20 [00:02<00:00,  8.41it/s]
Test set: Average loss: 0.5213, Accuracy: 51605/60000 (86.01%)

Adjusting learning rate of group 0 to 1.0000e-02.
Epoch 6
Train: Loss=0.2528 Batch_id=19 Accuracy=95.74: 100%|██████████| 20/20 [00:02<00:00,  8.68it/s]
Test set: Average loss: 0.4041, Accuracy: 53651/60000 (89.42%)

Adjusting learning rate of group 0 to 1.0000e-02.
Epoch 7
Train: Loss=0.2055 Batch_id=19 Accuracy=96.43: 100%|██████████| 20/20 [00:02<00:00,  8.36it/s]
Test set: Average loss: 0.3407, Accuracy: 54889/60000 (91.48%)

Adjusting learning rate of group 0 to 1.0000e-02.
Epoch 8
Train: Loss=0.1768 Batch_id=19 Accuracy=96.84: 100%|██████████| 20/20 [00:02<00:00,  8.61it/s]
Test set: Average loss: 0.3337, Accuracy: 54552/60000 (90.92%)

Adjusting learning rate of group 0 to 1.0000e-02.
Epoch 9
Train: Loss=0.1517 Batch_id=19 Accuracy=97.23: 100%|██████████| 20/20 [00:02<00:00,  8.51it/s]
Test set: Average loss: 0.3302, Accuracy: 54440/60000 (90.73%)

Adjusting learning rate of group 0 to 1.0000e-02.
Epoch 10
Train: Loss=0.1205 Batch_id=19 Accuracy=97.53: 100%|██████████| 20/20 [00:02<00:00,  8.54it/s]
Test set: Average loss: 0.2469, Accuracy: 56261/60000 (93.77%)

Adjusting learning rate of group 0 to 1.0000e-02.
Epoch 11
Train: Loss=0.1293 Batch_id=19 Accuracy=97.86: 100%|██████████| 20/20 [00:02<00:00,  8.60it/s]
Test set: Average loss: 0.2576, Accuracy: 55866/60000 (93.11%)

Adjusting learning rate of group 0 to 1.0000e-02.
Epoch 12
Train: Loss=0.1277 Batch_id=19 Accuracy=97.94: 100%|██████████| 20/20 [00:02<00:00,  8.59it/s]
Test set: Average loss: 0.2564, Accuracy: 55795/60000 (92.99%)

Adjusting learning rate of group 0 to 1.0000e-02.
Epoch 13
Train: Loss=0.1270 Batch_id=19 Accuracy=98.24: 100%|██████████| 20/20 [00:02<00:00,  8.24it/s]
Test set: Average loss: 0.2297, Accuracy: 56222/60000 (93.70%)

Adjusting learning rate of group 0 to 1.0000e-02.
Epoch 14
Train: Loss=0.0806 Batch_id=19 Accuracy=98.36: 100%|██████████| 20/20 [00:02<00:00,  8.46it/s]
Test set: Average loss: 0.2405, Accuracy: 55904/60000 (93.17%)

Adjusting learning rate of group 0 to 1.0000e-02.
Epoch 15
Train: Loss=0.0822 Batch_id=19 Accuracy=98.49: 100%|██████████| 20/20 [00:03<00:00,  6.57it/s]
Test set: Average loss: 0.2143, Accuracy: 56332/60000 (93.89%)

Adjusting learning rate of group 0 to 1.0000e-03.
Epoch 16
Train: Loss=0.0539 Batch_id=19 Accuracy=98.75: 100%|██████████| 20/20 [00:02<00:00,  8.28it/s]
Test set: Average loss: 0.1947, Accuracy: 56790/60000 (94.65%)

Adjusting learning rate of group 0 to 1.0000e-03.
Epoch 17
Train: Loss=0.0736 Batch_id=19 Accuracy=98.68: 100%|██████████| 20/20 [00:03<00:00,  5.41it/s]
Test set: Average loss: 0.1870, Accuracy: 57069/60000 (95.11%)

Adjusting learning rate of group 0 to 1.0000e-03.
Epoch 18
Train: Loss=0.1044 Batch_id=19 Accuracy=98.70: 100%|██████████| 20/20 [00:02<00:00,  8.22it/s]
Test set: Average loss: 0.1861, Accuracy: 57019/60000 (95.03%)

Adjusting learning rate of group 0 to 1.0000e-03.
Epoch 19
Train: Loss=0.0811 Batch_id=19 Accuracy=98.72: 100%|██████████| 20/20 [00:02<00:00,  7.48it/s]
Test set: Average loss: 0.1883, Accuracy: 56966/60000 (94.94%)

Adjusting learning rate of group 0 to 1.0000e-03.
Epoch 20
Train: Loss=0.0668 Batch_id=19 Accuracy=98.76: 100%|██████████| 20/20 [00:02<00:00,  8.26it/s]
Test set: Average loss: 0.1864, Accuracy: 57032/60000 (95.05%)

Adjusting learning rate of group 0 to 1.0000e-03.


  ```

## Observations/ Learning

* Total params: 18,050
* Test accuracy achieved : 95.05
* No batch normalization or dropout was used to close to the prediction layer

