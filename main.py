from Data_Wrapper import Data
from NeuralNetwork import NeuralNetwork

# WARNING: It's better to download the ./assets/train.txt from the assignment page since it seem to have problems with crlf thingy on git
if __name__ == '__main__':
    data = Data("./assets/train.txt")

    nn = NeuralNetwork(data.structure)