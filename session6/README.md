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
