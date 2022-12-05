#  Handwriting Recognition using Convolutional Neural Networks MINST Dataset
Creation of a convolutional neural network in order to recognize handwriting in an image and give its meaning.
Trained with the MINST dataset.
The code is written in Python and use the Keras and Tensorflow frameworks.


## Description

The dataset consists of 70,000 images (60,000 training and 10,000 test).
The images contain images of numbers from 0 to 9 written in handwriting.  

The data is augmented with the help of data augmentation tools to get more training data using rotations and flips among others.  

The model consists of 5 layers.

* The first layer consists of a convolution layer with Weight Decay, also called L2 Regularization. The activation function used is the rectifier (ReLU).
Batch normalization is applied to improve the stability  

* The second layer consists of a convolution layer with L2 Regularization. The activation function used is the rectifier (ReLU).
Batch normalization is applied to improve the stability.
MaxPooling is applied to downsample the input along its spatial dimensions.
We also apply a Dropout of 0.2 which randomly sets input units to 0 with a frequency of rate at each step during training time, which helps prevent overfitting.  

* The third layer consists of a convolution layer with L2 Regularization. The activation function used is (ReLU) with Batch Normalization.  

* The fourth layer consists of a convolution layer with L2 Regularization. The activation function used is (ReLU).
We apply Batch Normalization to improve the stability.
MaxPooling is applied to downsample the input along its spatial dimensions.
We also apply a Dropout of 0.3 which randomly sets input units to 0 with a frequency of rate at each step during training time, which helps prevent overfitting.  

* The last layer is composed of 10 neurons (one for each object class.
The activation function used is softmax.  

* The optimizer used is Adam.

* The mnist.h5 contains the trained model with all the weights.

## Getting Started

### Dependencies

* Python
* Keras with Tensorflow Backend

### Executing program

* Run mnist.py with Python
```
python mnist.py
```

## Author

KARUNAKARAN Nithushan


## License

This project is licensed under the MIT License - see the LICENSE.md file for details

