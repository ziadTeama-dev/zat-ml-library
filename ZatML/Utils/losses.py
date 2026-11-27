import numpy as np
from ZatML.Utils.distances import eucudian_distance

def MSRE(Y_predicted , Y_target):
    N , _ = Y_predicted.shape
    return 1/N * np.sum(eucudian_distance(Y_predicted , Y_target))


def binary_crossentropy(y_predicted , y_target):
    N , _ = y_predicted.shape
    return -1/N * np.sum(y_target * np.log1p(y_predicted) + (1-y_target) * np.log1p(1-y_predicted))