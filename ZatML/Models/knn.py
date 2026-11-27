import numpy as np
from ZatML.Utils.distances import manhattendistance


class Knn:
    def __init__(self , k):
        self.x = None
        self.y = None
        self.k = k

    def fit(self, x : np.array , y : np.array) -> None:
        # initialize the data 
        self.x = x
        self.y = y

    def predict(self , x : np.array) -> np.array:
        """
        """
        # first i will get the distance between the input and all data
        distances = manhattendistance(x , self.x)
        # then get the k-indices that represent the minimum  of each distance
        # print(distances)
        k_nearest = np.argsort(distances)[:self.k]

        return self.y[k_nearest].mode()