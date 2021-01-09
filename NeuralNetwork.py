import numpy as np

from DataSet import DataSet


class NeuralNetwork:
    def __init__(self, structure, f, fDeriv, w=None):
        '''
        constructor
        :param structure: size of each layer, structure[0] is input layer size (including bias), and structure[-1] is output layer size
        :param f: activation function
        :param fDeriv: activation function derivative. it should take f(x) as an input and returns derivative of f(x) as output
        :param w: weight matrix
        '''
        self.n = structure
        self.L = len(structure) # Number of layers
        self.a = self.networkShapedMatrix() # a[i][j] is the value of jth node in the ith layer
        self.f_v = np.vectorize(f) # vectorizing f to apply it to a vector instead of just a scalar
        self.fDeriv = fDeriv
        self.fDeriv_v = np.vectorize(fDeriv) # vectorizing fDeriv
        if (w == None):
            self.randInitWeights()
        else:
            self.w = w
        self.errors = [] # for logging of cost

    def networkShapedMatrix(self):
        '''

        :return: 2D Matrix. mat[i][j] is the jth node in the ith layer.
        '''
        return [np.zeros(shape=length, dtype=np.float64) for length in self.n]

    def randInitWeights(self):
        '''
        initializes all weight matrices [1, L) randomly with values in range [0, 1)
        :return: void
        '''
        self.w = [np.zeros(1)] + [np.random.rand(self.n[l], self.n[l - 1]) for l in range(1, self.L)]
        # self.w = [np.zeros(1)] + [np.ones(shape=(self.n[l], self.n[l - 1])) for l in range(1, self.L)]

    def learn(self, ds: DataSet, EPOCHS: int, alpha: float):
        '''
        learns weight w on dataset ds
        :param ds: training set
        :param EPOCHS: number of epochs to train on
        :param alpha: learning rate
        :return:
        '''
        self.errors.append(self.MSE(ds))
        for i in range(EPOCHS):
            for [x, y] in ds:
                self.forward(x)
                self.backward(y, alpha)
            self.errors.append(self.MSE(ds))
            print("Error at iteration " + str(i) + " = " + str(self.errors[-1]))

    def SE(self, y_pred, y_actual):
        '''

        :param y_pred: predicted output vector
        :param y_actual: actual output vector
        :return: summation of squared pairwise difference of elements of both vectors
        '''
        return np.sum((y_pred - y_actual)**2)

    def MSE(self, ds):
        '''
        Mean square error
        :param ds:
        :return:
        '''
        total = 0
        for [x, y] in ds:
            total += self.SE(self.forward(x), y)
        total /= 2 * ds.m
        return total

    def forward(self, x: np.ndarray):
        '''
        updates self.a to hold value of forward propagating x
        :param x: 1D input feature vector of size self.n[0]
        :return: 1D output vector of size self.n[-1]
        '''
        self.a[0] = x
        for i in range (1, self.L):
            self.a[i] = self.f_v(np.matmul(self.w[i], self.a[i - 1]))
        return self.a[-1]

    def backward(self, y: np.ndarray, alpha):
        '''
        updated weights on w
        :param y: 1D output vector of size self.n[-1]
        :return: delta matrix
        '''
        d = self.networkShapedMatrix()
        d[-1] = (self.a[-1] - y) * self.fDeriv_v(self.a[-1])
        for h in range (self.L - 2, 0, -1):
            d[h] = np.matmul(self.w[h + 1].transpose(), d[h + 1]) * self.fDeriv_v(self.a[h])
            '''
            explanation of above line:
            np.matmul(...) will return an array of same size as d[h]. each element of the array is resultant from the dot product d[h + 1] array with each column of self.w[h + 1]
            then this array is element-wise multiplied with an array containing the derivative of each element of a[h]
            '''
            # for j in range(self.n[h]):
            #     d[h][j] = np.sum(d[h + 1] * self.w[h + 1][:, j]) * self.fDeriv(self.a[h][j])

        for h in range(1, self.L):
            self.w[h] -= alpha * self.makeMat(d[h], self.a[h - 1])
        return d

    def makeMat(self, v1: np.ndarray, v2: np.ndarray):
        '''
        generates matrix out of product of v1, v2
        :param v1: 1D np array
        :param v2: 1D np array
        :return: matrix where mat[i][j] is v1[i] * v2[j]
        '''
        return np.matmul(v1.reshape((v1.shape[0], 1)), v2.reshape((1, v2.shape[0])))