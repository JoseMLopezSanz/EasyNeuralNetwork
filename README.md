# EasyNeuralNetwork
Visual implementation of the decision boundaries of a neural network

## Description

With this simple implementation using Keras (A very versatile and easy to use library for python to implement neural networks) you can play with the number of classes and parameters of the neural network in a 2 Dimensional space to see the decision boundary of the net.

I hope you find it useful and it helps understand a bit more how they work and how we can use their parameters to find very good classifiers by using deep learning but being careful because overfitting could be a problem.

## Requiremnts 

Keras, matplotlib, sklearn, pandas, argparse

## Examples 

Arguments:

* --dataset or -d: type of dataset (moons or blobs)
* --classes or -c: number of classes to classify (#number of perceptrons in the output layer)
* --layers or -l: number of hidden layers and number of perceptrons per hidden layer.
* --epochs or -e: number of epochs to train the model
* --learning_rate or -lr: learning rate used to train the model with stochastic gradient descend (SGD)
Some examples

Example 1
```
python Neural_Network.py --d moons -l 20 20 20 -e 10 -c 2 -lr 0.001
```
We create a neural network with input layer for 2 features, 3 hidden layers with 20 perceptrons each and an output layer for 3 classes, we train the model for 10 epochs with a learning rate of 0.001 
![alt text](https://raw.githubusercontent.com/JoseMLopezSanz/EasyNeuralNetwork/master/Figure_1.png)


Example 2
```
python Neural_Network.py --dataset  moons -l 20 20 20 -e 500 -c 2 -lr 0.001
```
Same model but trained for 500 epochs 
![alt text](https://raw.githubusercontent.com/JoseMLopezSanz/EasyNeuralNetwork/master/Figure_2.png)


Example 3
```
python Neural_Network.py --dataset blobs  -l 20 20 20  -e 500 -c 5 -lr 0.001
```
In this example we use a different dataset with 5 classes. 
![alt text](https://raw.githubusercontent.com/JoseMLopezSanz/EasyNeuralNetwork/master/Figure_3.png)



## Authors

* **Jose M Lopez** - *Initial work* - [Me](https://github.com/JoseMLopezSanz)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
