import math

from Data_Wrapper import Data
from NeuralNetwork import NeuralNetwork

# WARNING: It's better to download the ./assets/train.txt from the assignment page since it seem to have problems with crlf thingy on git

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def sigmoidDeriv(x):
    return sigmoid(x) * (1 - sigmoid(x))

if __name__ == '__main__':
    data = Data("./assets/train.txt")

    nn = NeuralNetwork(data.structure, sigmoid, sigmoidDeriv)
    nn.learn(data.data, 1000)