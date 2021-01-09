import numpy as np

class NeuralNetwork:
    def __init__(self, structure, w=None):
        '''
        constructor
        :param structure: size of each layer, structure[0] is input layer size (including bias), and structure[-1] is output layer size
        '''
        self.n = structure
        self.L = len(structure)
        self.a = [np.zeros(shape=length, dtype=np.float64) for length in self.n]
        if (w == None):
            self.w = [np.zeros(1)] + [np.random.rand(self.n[l], self.n[l - 1]) for l in range(1, self.L)]
        else:
            self.w = w