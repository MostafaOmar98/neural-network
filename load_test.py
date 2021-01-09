import pickle
import numpy as np
import conf
from Data_Wrapper import Data
from DataSet import DataSet
from NeuralNetwork import NeuralNetwork

if __name__ == '__main__':
    with open(conf.PARAMS_PATH, 'rb') as infile:
        params = pickle.load(infile)
        w = params[0]
        norm = params[1]
        normFeatures = norm[0]
        normLabels = norm[1]

    test_path = conf.TEST_PATH
    s = str(input("Enter test file path or enter x to use \"" + test_path + "\": "))
    if (s != "x"):
        test_path = s
    print("Using test path: " + test_path)
    data = Data(test_path, normFeatures, normLabels)
    nn = NeuralNetwork(data.structure, conf.sigmoid, conf.sigmoidDeriv, w)
    print("MSE On test data = " + str(nn.MSE(data.data)))
    while(True):
        s = str(input("Do you want to test one more example [y/n]: "))
        if (s == "n"):
            break
        inp = [1]
        for i in range(data.inSize - 1):
            inp.append(float(input("Enter iput for feature #" + str(i) + ": ")))

        x = np.array(inp)
        x -= normFeatures[0]
        x /= normFeatures[1]

        y = nn.forward(x)
        y *= normLabels[1]
        y += normLabels[0]

        print("denormalized output: " + str(y))