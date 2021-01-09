import numpy as np

from DataSet import DataSet


class NeuralNetwork:
    def __init__(self, structure, f, fDeriv, w=None):
        '''
        constructor
        :param structure: size of each layer, structure[0] is input layer size (including bias), and structure[-1] is output layer size
        '''
        self.n = structure
        self.L = len(structure)
        self.a = [np.zeros(shape=length, dtype=np.float64) for length in self.n]
        self.f_v = np.vectorize(f)
        if (w == None):
            self.randInitWeights()
        else:
            self.w = w

    def randInitWeights(self):
        self.w = [np.zeros(1)] + [np.random.rand(self.n[l], self.n[l - 1]) for l in range(1, self.L)]

    def learn(self, ds: DataSet, EPOCHS: int):
        self.randInitWeights()
        for i in range(EPOCHS):
            for [x, y] in ds:
                self.forward(x)
                self.backward(y)

    def forward(self, x: np.ndarray):
        '''

        :param x: 1D input feature vector of size self.n[0]
        :return: 1D output vector of size self.n[-1]
        '''
        self.a[0] = np.copy(x)
        for i in range (1, self.L):
            self.a[i] = self.f_v(np.matmul(self.w[i], self.a[i - 1]))
        return self.a[-1]

    def backward(self, y):
        pass