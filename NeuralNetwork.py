import numpy as np

class NeuralNetwork:
    def __init__(self, structure):
        '''
        constructor
        :param structure: size of each layer, structure[0] is input layer size (including bias),
        '''
        self.n = structure


