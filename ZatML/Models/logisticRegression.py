import numpy as np
from ZatML.Utils.f import sigmoid
from ZatML.Utils.losses import binary_crossentropy


class LogisticRegression:
    def __init__(self , num_iter:int , show_logs : bool = True , lr : float = .1):

        self.num_iter = num_iter
        self.lr = lr
        self.show_logs = show_logs

        self.x = None
        self.y = None
        self.w = None
        self.b = None

    def fit(self , x:np.array , y:np.array) -> None:
        N , D = x.shape
        # initialize the params
        self.x = x
        self.y = y
        self.w = np.random.randn(D,1) * 1/D
        self.b = 0

        for iter in range(1,self.num_iter+1):
            # forward
            # output = sigmoid(x*w + b)
            y_prediction = sigmoid(self.x.dot(self.w) + self.b)
            # calculate the loss
            Loss = binary_crossentropy(y_prediction , self.y)
            # update the weight
            self.w -= self.lr * 1/N * self.x.T.dot(y_prediction - self.y)
            # update the bias
            self.b -= self.lr * 1/N * np.sum(y_prediction - self.y)
            # showing the loss
            if self.show_logs:
               print(f"[{self.num_iter}|{iter}]{"="*30}{">"} {Loss}")

    def predict(self, x : np.array) -> np.array:
        
        output =  sigmoid(x.dot(self.w) + self.b)
        labels = np.where(output >= .5 , 1 , 0)
        return output , labels