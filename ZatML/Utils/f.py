import numpy as np
def sigmoid(x : np.array):
    return 1 /  (1 + np.e ** -x) 