# S5.ipynb
This is main notebook for training the MNIST dataset.
- Train and test transforms are applied as explained in **utils.py**
- Batch size of 512 is considered
- matplotlib libraries in **utils.py** are used to display plots of dataset and track loss and accuracies.
- Neural architecture as defined by **model.py** is used
- Model has 593200 parameters
- Model is trained for 20 epochs using gradient descent - stepLR starting at 0.01
- Loss and accuracy graphs are generated

![image](https://github.com/senthilva/ERA1/assets/8141261/c5a01939-3751-4a51-8478-20a42eb42dc2)




## model.py

This files contains neural architecture
-  2 Conv2d
- Max Pooling
- 2 Conv2d
- Max Pooling
- 2 linear layers
- one hot encoded output for 10 output values

## utils.py

### train_transforms
- Randomly apply centercrop
- resize to 28x28
- random rotation of +-15 degrees
- apply normalization
- convert to tensor

### test_transforms
- 
- apply normalization
- convert to tensor

### plot_train_samples

- plt 12 images from train_loader
- display in 3x4 grid

### plot_loss_accuracy

- plot collected training loss, test loss
- plot collected training accuracy, test accuracy
