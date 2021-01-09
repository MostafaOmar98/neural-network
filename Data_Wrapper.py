import pandas as pd
import numpy as np

from DataSet import DataSet


class Data:
    data = None

    def __init__(self, path, normFeatures=None, normLabels=None):
        '''
        path is absolute
        path is a tsv file
        structure has the network structure. structure[0] is the size of the input layer (With bias), and structure[-1] is the size of output layer
        '''
        self.structure = pd.read_csv(filepath_or_buffer=path, delim_whitespace=True, header=None, nrows=1).to_numpy(dtype=int).flatten()
        self.structure[0] += 1
        self.data = pd.read_csv(filepath_or_buffer=path, delim_whitespace=True, header=None, skiprows=[0, 1]).to_numpy(dtype=np.float64)

        self.features = self.data[:, :self.structure[0]]
        self.labels = self.data[:, self.structure[0]:]
        self.inSize = self.features.shape[1] + 1  # Number of features with bias included
        self.outSize = self.labels.shape[1]
        self.data_size = self.features.shape[0]
        self.data = DataSet(self.features, self.labels, normFeatures, normLabels)