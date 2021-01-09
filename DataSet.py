import numpy as np


class DataSet:
    def __init__(self, features: np.ndarray, labels: np.ndarray, normFeatures: np.ndarray, normLabels: np.ndarray):
        self.features = features
        self.labels = labels

        self.addBias()

        self.n = self.features.shape[1]  # Number of features with bias included
        self.m = self.features.shape[0]

        '''
        norm will be list of size 2 [mean, std] for each of features and labels
        '''
        self.normFeatures = None
        self.normLabels = None

        if (normFeatures == None):
            assert(normLabels == None)
            self.calcNorm()
        else:
            self.normFeatures = normFeatures
            self.normLabels = normLabels

        self.applyNormalization(self.normFeatures, self.normLabels)

    def addBias(self):
        self.features = np.insert(self.features, 0, 1, 1)  # Adding bias

    def __getitem__(self, i: int):
        '''
        returns list [x, y]
        x is nparray, features of example i with bias included
        y is float value, the label of example i
        '''
        return [self.features[i], self.labels[i]]

    def calcNorm(self):
        '''
        calculates normalization parameters (Including bias)
        handles bias so applyNormalization doesn't have to worry about it
        '''
        self.normFeatures = self.calcNormSingular(self.features)
        self.normFeatures[0][0] = 0 # for bias
        self.normFeatures[1][0] = 1 # for bias
        self.normLabels = self.calcNormSingular(self.labels)

    def calcNormSingular(self, v):
        mean = np.mean(v, axis=0)
        std = np.std(v, axis=0)
        return [mean, std]

    def applyNormalization(self, normFeatures, normLabels):
        self.applyNormalizationSingular(self.features, normFeatures)
        self.applyNormalizationSingular(self.labels, normLabels)

    def applyNormalizationSingular(self, v, norm):
        # TODO: Howwa bel debugger shaklo 3amalaha sa7, bas make sure bardo from el documentation
        v[:] -= norm[0]
        v[:] /= norm[1]