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
        self.a = self.networkShapedMatrix()
        self.f_v = np.vectorize(f)
        self.fDeriv = fDeriv
        self.fDeriv_v = np.vectorize(fDeriv)
        if (w == None):
            self.randInitWeights()
        else:
            self.w = w

    def networkShapedMatrix(self):
        return [np.zeros(shape=length, dtype=np.float64) for length in self.n]

    def randInitWeights(self):
        self.w = [np.zeros(1)] + [np.random.rand(self.n[l], self.n[l - 1]) for l in range(1, self.L)]
        # self.w = [np.zeros(1)] + [np.ones(shape=(self.n[l], self.n[l - 1])) for l in range(1, self.L)]

    def learn(self, ds: DataSet, EPOCHS: int):
        self.randInitWeights()
        for i in range(1):
            for [x, y] in ds:
                self.forward(x)
                self.backward(y)
                break

    def forward(self, x: np.ndarray):
        '''
        updates self.a to hold value of forward propagating x
        :param x: 1D input feature vector of size self.n[0]
        :return: 1D output vector of size self.n[-1]
        '''
        self.a[0] = np.copy(x)
        for i in range (1, self.L):
            self.a[i] = self.f_v(np.matmul(self.w[i], self.a[i - 1]))
        return self.a[-1]

    def backward(self, y: np.ndarray):
        '''
        updated weights on w
        :param y: 1D output vector of size self.n[-1]
        :return: void
        '''
        d = self.networkShapedMatrix()
        d[-1] = (self.a[-1] - y) * self.fDeriv_v(self.a[-1])
        for h in range (self.L - 2, -1, -1):
            d[h] = np.matmul(self.w[h+1].transpose(), d[h + 1]) * self.fDeriv_v(self.a[h])
            '''
            explanation of above line:
            np.matmul(...) will return an array of same size as d[h]. each element of the array is resultant from the dot product d[h + 1] array with each column of self.w[h + 1]
            then this array is element-wise multiplied with an array containing the derivative of each element of a[h]
            '''
            # for j in range(self.n[h]):
            #     d[h][j] = np.sum(d[h + 1] * self.w[h + 1][:, j]) * self.fDeriv(self.a[h][j])

        for h in range(1, self.L):
            pass