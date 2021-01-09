import math
import numpy as np
import conf
from Data_Wrapper import Data
from NeuralNetwork import NeuralNetwork
from matplotlib import pyplot as plt
import pickle

# WARNING: It's better to download the ./assets/train.txt from the assignment page since it seem to have problems with crlf thingy on git
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
    data = Data(conf.TRAIN_PATH)
    nn = NeuralNetwork(data.structure, conf.sigmoid, conf.sigmoidDeriv)
    epochs = int(input("Epochs: "))
    alpha = float(input("Learning rate: "))
    nn.learn(data.data, epochs, alpha)
    plt.plot(nn.errors)
    plt.show()

    params = [nn.w, [data.data.normFeatures, data.data.normLabels]]

    with open(conf.PARAMS_PATH, 'wb') as outfile:
        pickle.dump(params, outfile, pickle.HIGHEST_PROTOCOL)

    print(nn.w)