import math

TRAIN_PATH = './assets/train.txt'
PARAMS_PATH = './assets/params.pkl'
TEST_PATH = './assets/test.txt'

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def sigmoidDeriv(x):
    '''

    :param x: value that has already been sigmoided
    :return: derivative of sigmoid function has it been applied on x before sigmoiding it
    '''
    return x * (1 - x)