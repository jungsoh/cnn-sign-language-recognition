# Convolutional neural network: Sign language recognition
Recognizing multiple classes of objects from images is a common computer vision task. Here we have a 6-class problem where we want to recognize 6 different digits (0, 1, 2, 3, 4, 5) of [American Sign Language](https://en.wikipedia.org/wiki/American_Sign_Language) given an image of a hand. We build a convolutional neural network for this multi-class classification task using TensorFlow Keras Functional API. I did this project in the [Convolutional Neural Networks](https://www.coursera.org/learn/convolutional-neural-networks) course as part of the [Deep Learning Specialization](https://www.coursera.org/specializations/deep-learning).

## Datasets
We have 1080 training examples and 120 test examples, where each example is of shape (64, 64, 3) with each of RGB channel image is of size 64x64. The examples are labeled as one of 0, 1, 2, 3, 4, and 5 for the corresponding digits. These labels are one-hot encoded to be used as the target output of the recognizer, as shown below.

![One-hot encoding of class labels](images/SIGNS.png)

## Convolutional eural network architecture
We used TensorFlow Keras Functional API to build a neural network model depicted below. 
![convolutional neural network architecture](images/model.png)

The resulting model's `.summary()` method shows the following layers and parameters.
```
Model: "functional_5"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_8 (InputLayer)         [(None, 64, 64, 3)]       0         
_________________________________________________________________
conv2d_12 (Conv2D)           (None, 64, 64, 8)         392       
_________________________________________________________________
re_lu_12 (ReLU)              (None, 64, 64, 8)         0         
_________________________________________________________________
max_pooling2d_8 (MaxPooling2 (None, 8, 8, 8)           0         
_________________________________________________________________
conv2d_13 (Conv2D)           (None, 8, 8, 16)          528       
_________________________________________________________________
re_lu_13 (ReLU)              (None, 8, 8, 16)          0         
_________________________________________________________________
max_pooling2d_9 (MaxPooling2 (None, 2, 2, 16)          0         
_________________________________________________________________
flatten_3 (Flatten)          (None, 64)                0         
_________________________________________________________________
dense_3 (Dense)              (None, 6)                 390       
=================================================================
Total params: 1,310
Trainable params: 1,310
Non-trainable params: 0
_________________________________________________________________
```
We trained the convolutional neural network for 100 epochs with the Keras model's `.fit()` method. Checking into the returned history objecrt to evaluate the performance reveals that the model has the training accuracy of 0.88 and the validation accuracy of 0.82.
