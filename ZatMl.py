import numpy as np
import math

def eucudian_distance(x1 , x2):
    """
    This function used to get distance between 2 numpy list\n
    x1 : first point\n
    x2 : second point\n
    """
    return np.sum(math.sqrt((x1 - x2) **2),
                  axis=-1,
                  keepdims=True)

def manhattendistance(x1 , x2):
    return abs(x1 - x2)

class Knn:
    def __init__(self , k):
        self.x = None
        self.y = None
        self.k = k

    def fit(self, x : np.array , y : np.array):
        # initialize the data 
        self.x = x
        self.y = y

    def predict(self , x : np.array):
        """
        """
        # first i will get the distance between the input and all data
        distances = manhattendistance(x , self.x)
        
        

knn = Knn()

x = np.random.randn(5,10)

y = np.random.randint(0,10,(5,1))

x1 = np.random.randn(1,10)


knn.fit(x,y)

knn.predict(x1)



        


    
