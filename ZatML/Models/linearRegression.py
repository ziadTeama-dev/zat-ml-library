import numpy as np
from ZatML.Utils.losses import MSRE       
class LinearRegression:
    def __init__(self , num_iter:int , show_logs : bool = True , lr : float = .1):

        self.num_iter = num_iter
        self.lr = lr
        self.show_logs = show_logs
        self.x = None
        self.y = None
        self.w = None   

    def fit( self , x : np.array , y : np.array) -> None:
        N , D = x.shape
        self.x = x
        self.y = y
        self.w = np.random.randn(D,1) * 1/D
        self.b = 0

        for iter in range(1,self.num_iter+1):
            # 1 forward => output = x * self.w + b
            y_predicted = self.x.dot(self.w) + self.b
            # print(self.w.shape)
            # print(y_predicted)
            # print(y_predicted)
            # calculate the loss
            Loss = MSRE(y_predicted , self.y)
            # update the the w of the line w = w + lr * (Y - Y_predicted)*x
            # print(y_predicted.shape)
            # print(self.y.shape)
            self.w -=  self.lr * 2/N * self.x.T.dot( y_predicted - self.y)
            # print(self.w.shape)
            self.b -=  self.lr * 2/N * sum((y_predicted-self.y))
            # print out the information of the loss
            if self.show_logs:
               print(f"[{self.num_iter}|{iter}]{"="*30}{">"} {Loss}")
    def predict(self, x : np.array) -> np.array:
        
        output =  x.dot(self.w) + self.b
        
        return output