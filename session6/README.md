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
-


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
      0%|          | 0/469 [00:00<?, ?it/s]EPOCH : 1
   /usr/local/lib/python3.7/dist-packages/ipykernel_launcher.py:34: UserWarning: Implicit dimension choice for log_softmax has been deprecated. Change the call to include dim=X as an argument.
   loss=0.1951860934495926 batch_id=468: 100%|██████████| 469/469 [00:29<00:00, 15.88it/s]
     0%|          | 0/469 [00:00<?, ?it/s]
   Test set: Average loss: 0.1778, Accuracy: 9515/10000 (95.15%)

   EPOCH : 2
   loss=0.12187216430902481 batch_id=468: 100%|██████████| 469/469 [00:28<00:00, 16.23it/s]
     0%|          | 0/469 [00:00<?, ?it/s]
   Test set: Average loss: 0.1427, Accuracy: 9550/10000 (95.50%)

   EPOCH : 3
   loss=0.041622504591941833 batch_id=468: 100%|██████████| 469/469 [00:30<00:00, 15.44it/s]
     0%|          | 0/469 [00:00<?, ?it/s]
   Test set: Average loss: 0.0894, Accuracy: 9713/10000 (97.13%)

   EPOCH : 4
   loss=0.037716399878263474 batch_id=468: 100%|██████████| 469/469 [00:30<00:00, 15.45it/s]
     0%|          | 0/469 [00:00<?, ?it/s]
   Test set: Average loss: 0.0674, Accuracy: 9797/10000 (97.97%)

   EPOCH : 5
   loss=0.07980864495038986 batch_id=468: 100%|██████████| 469/469 [00:29<00:00, 15.76it/s]
     0%|          | 0/469 [00:00<?, ?it/s]
   Test set: Average loss: 0.0462, Accuracy: 9863/10000 (98.63%)

   EPOCH : 6
   loss=0.025407209992408752 batch_id=468: 100%|██████████| 469/469 [00:28<00:00, 16.26it/s]
     0%|          | 0/469 [00:00<?, ?it/s]
   Test set: Average loss: 0.0442, Accuracy: 9874/10000 (98.74%)

   EPOCH : 7
   loss=0.004418657626956701 batch_id=468: 100%|██████████| 469/469 [00:29<00:00, 16.17it/s]
     0%|          | 0/469 [00:00<?, ?it/s]
   Test set: Average loss: 0.0462, Accuracy: 9849/10000 (98.49%)

   EPOCH : 8
   loss=0.078040212392807 batch_id=468: 100%|██████████| 469/469 [00:29<00:00, 16.14it/s]
     0%|          | 0/469 [00:00<?, ?it/s]
   Test set: Average loss: 0.0454, Accuracy: 9862/10000 (98.62%)

   EPOCH : 9
   loss=0.030806919559836388 batch_id=468: 100%|██████████| 469/469 [00:29<00:00, 16.12it/s]
     0%|          | 0/469 [00:00<?, ?it/s]
   Test set: Average loss: 0.0302, Accuracy: 9907/10000 (99.07%)

   EPOCH : 10
   loss=0.0046560498885810375 batch_id=468: 100%|██████████| 469/469 [00:29<00:00, 16.04it/s]
     0%|          | 0/469 [00:00<?, ?it/s]
   Test set: Average loss: 0.0306, Accuracy: 9912/10000 (99.12%)

   EPOCH : 11
   loss=0.0323166698217392 batch_id=468: 100%|██████████| 469/469 [00:29<00:00, 16.04it/s]
     0%|          | 0/469 [00:00<?, ?it/s]
   Test set: Average loss: 0.0336, Accuracy: 9897/10000 (98.97%)

   EPOCH : 12
   loss=0.012631400488317013 batch_id=468: 100%|██████████| 469/469 [00:29<00:00, 15.94it/s]
     0%|          | 0/469 [00:00<?, ?it/s]
   Test set: Average loss: 0.0457, Accuracy: 9867/10000 (98.67%)

   EPOCH : 13
   loss=0.05438346043229103 batch_id=468: 100%|██████████| 469/469 [00:29<00:00, 15.94it/s]
     0%|          | 0/469 [00:00<?, ?it/s]
   Test set: Average loss: 0.0430, Accuracy: 9865/10000 (98.65%)

   EPOCH : 14
   loss=0.012918871827423573 batch_id=468: 100%|██████████| 469/469 [00:29<00:00, 15.84it/s]
     0%|          | 0/469 [00:00<?, ?it/s]
   Test set: Average loss: 0.0317, Accuracy: 9904/10000 (99.04%)

   EPOCH : 15
   loss=0.007770549971610308 batch_id=468: 100%|██████████| 469/469 [00:29<00:00, 16.14it/s]
     0%|          | 0/469 [00:00<?, ?it/s]
   Test set: Average loss: 0.0297, Accuracy: 9914/10000 (99.14%)

   EPOCH : 16
   loss=0.011336133815348148 batch_id=468: 100%|██████████| 469/469 [00:29<00:00, 16.03it/s]
     0%|          | 0/469 [00:00<?, ?it/s]
   Test set: Average loss: 0.0283, Accuracy: 9912/10000 (99.12%)

   EPOCH : 17
   loss=0.03598247841000557 batch_id=468: 100%|██████████| 469/469 [00:29<00:00, 16.00it/s]
     0%|          | 0/469 [00:00<?, ?it/s]
   Test set: Average loss: 0.0310, Accuracy: 9900/10000 (99.00%)

   EPOCH : 18
   loss=0.01342777069658041 batch_id=468: 100%|██████████| 469/469 [00:29<00:00, 16.04it/s]
     0%|          | 0/469 [00:00<?, ?it/s]
   Test set: Average loss: 0.0340, Accuracy: 9895/10000 (98.95%)

   EPOCH : 19
   loss=0.031836893409490585 batch_id=468: 100%|██████████| 469/469 [00:29<00:00, 16.12it/s]

   Test set: Average loss: 0.0292, Accuracy: 9913/10000 (99.13%)


  ```

## Observations/ Learning

* Total params: 18,050
* Test accuracy achieved : 99.13
* No batch normalization or dropout was used to close to the prediction layer

