import numpy as np
def eucudian_distance(x1:np.array , x2:np.array):
    """
    This function used to get distance between 2 numpy list\n
    x1 : first point\n
    x2 : second point\n
    """
    return np.sqrt(np.sum((x1 - x2) **2,
                  axis=-1,))

def manhattendistance(x1:np.array , x2:np.array):
    return np.sum(abs(x1 - x2) , axis = -1 )