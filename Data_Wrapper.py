import pandas as pd
import numpy as np

from DataSet import DataSet


class Data:
    data = None

    def __init__(self, path, norm=None):
        '''
        path is absolute
        path is a tsv file
        '''
        self.structure = pd.read_csv(filepath_or_buffer=path, delim_whitespace=True, header=None, nrows=1).to_numpy(dtype=int).flatten()
        self.data = pd.read_csv(filepath_or_buffer=path, delim_whitespace=True, header=None, skiprows=[0, 1]).to_numpy(dtype=np.float64)

        self.features = self.data[:, :self.structure[0]]
        self.labels = self.data[:, self.structure[0]:]
        self.inSize = self.features.shape[1] + 1  # Number of features with bias included
        self.outSize = self.labels.shape[1]
        self.data_size = self.features.shape[0]
        self.data = DataSet(self.features, self.labels, norm)