import math
import numpy as np
from Data_Wrapper import Data
from NeuralNetwork import NeuralNetwork
from matplotlib import pyplot as plt
import pickle
# WARNING: It's better to download the ./assets/train.txt from the assignment page since it seem to have problems with crlf thingy on git

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def sigmoidDeriv(x):
    '''

    :param x: value that has already been sigmoided
    :return: derivative of sigmoid function has it been applied on x before sigmoiding it
    '''
    return x * (1 - x)

if __name__ == '__main__':
    '''
    test from lab (OK)
        myW = [np.array([0]), np.array([[0.3, -0.9, 1], [-1.2, 1, 1]]), np.array([[1, 0.8]])]
        nn = NeuralNetwork([3, 2, 1], sigmoid, sigmoidDeriv, myW)
        x = np.array([1, 0, 1])
        y = np.array([0])
        nn.forward(x)
        # print(nn.a)
        print(nn.backward(y, 0.3))
        print()
        print()
        print(nn.w)
    '''
    data = Data("./assets/train.txt")
    nn = NeuralNetwork(data.structure, sigmoid, sigmoidDeriv)
    epochs = int(input("Epochs: "))
    alpha = float(input("Learning rate: "))
    nn.learn(data.data, epochs, alpha)
    plt.plot(nn.errors)
    plt.show()
    with open('./assets/weights.txt', 'wb') as outfile:
        pickle.dump(nn.w, outfile, pickle.HIGHEST_PROTOCOL)