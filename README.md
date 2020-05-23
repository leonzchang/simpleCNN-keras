# simpleCNN-keras

This is a simple CNN build with keras and perform how to tuning.

## command
```

simpleCNN.py [-h]  -train TRAIN_DATA_PATH 
                   -test TEST_DATA_PATH 
                   -size ROW_SIZE COLUMN_SIZE 
                   -c CATEGORIES_NUMBER 
                   -af ACTIVATION_FUNCTION
                   [-r WEIGHT_DECAY_RATE]
                   [-d DROPOUT_RATE]

optional arguments:
  -h, --help                    show this help message and exit
  -train TRAIN_DATA_PATH        enter train data path
  -test TEST_DATA_PATH          enter test data path
  -size ROW_SIZE COLUMN_SIZE    enter the image row size and column size
  -c CATEGORIES_NUMBER          enter categories number
  -af ACTIVATION_FUNCTION       enter activation function type
  -r WEIGHT_DECAY_RATE          enter weight decay rate,0~0.1
  -d DROPOUT_RATE               enter dropout rate, 0~0.5
```
* Example
`python simpleCNN.py -train dataset/training_set/ -test dataset/test_set/ -size 128 128 -c 3 -af softmax -r 0.01 -d 0.2`

## methods to prevent overfitting in machine learning

- Overfitting refers to a model that models the training data too well. Instead of learning the genral distribution of the data, the model learns the expected output for every data point.

![overfitting1](https://github.com/leonzchang/simpleCNN-keras/blob/master/assets/overfitting1.png)
![overfitting2](https://github.com/leonzchang/simpleCNN-keras/blob/master/assets/overfitting2.png)
### 1.mini-batch 
- epoch = batchSize * iteration
- epoch: one Epoch is when an ENTIRE dataset is passed forward and backward through the neural network only ONCE.
- batch: divide dataset into Number of Batches or sets or parts.
- batchSize: Total number of training examples present in a single batch.
- iterations: the number of batches needed to complete one epoch.
![minibatch](https://github.com/leonzchang/simpleCNN-keras/blob/master/assets/minibatch.png)
### 2.Dropout
- Neural networks process the information from one layer to the next. The idea is to randomly deactivate either neurons (dropout) or connections (dropconnect) during the training.
![dropout](https://github.com/leonzchang/simpleCNN-keras/blob/master/assets/dropout.jpeg)
### 3.Simplify the model
- On the left, the model is too simple. On the right it overfits.
![SimplifyModel](https://github.com/leonzchang/simpleCNN-keras/blob/master/assets/SimplifyModel.png)
### 4.Data augmentation & Noise

### 5.Regularization(Weight Decay)
- L1 and L2 regularization
- Add a penalty to the loss function,with the penalty, the model is forced to make compromises on its weights, as it can no longer make them arbitrarily large.
![regularization](https://github.com/leonzchang/simpleCNN-keras/blob/master/assets/regularization.png)
### 6.Early Termination 
- In most cases, the model starts by learning a correct distribution of the data, and, at some point, starts to overfit the data. By identifying the moment where this shift occurs, you can stop the learning process before the overfitting happens. As before, this is done by looking at the training error over time.
![EarlyTermination](https://github.com/leonzchang/simpleCNN-keras/blob/master/assets/EarlyTermination.png)

## Ref
https://hackernoon.com/memorizing-is-not-learning-6-tricks-to-prevent-overfitting-in-machine-learning-820b091dc42
