import numpy as np


def eucudian_distance(x1 , x2):
    """
    This function used to get distance between 2 numpy list\n
    x1 : first point\n
    x2 : second point\n
    """
    return np.sqrt(np.sum((x1 - x2) **2,
                  axis=-1,))

def manhattendistance(x1 , x2):
    return np.sum(abs(x1 - x2) , axis = -1 )

def MSRE(Y_predicted , Y_target):
    N , _ = Y_predicted.shape
    return 1/N * np.sum(eucudian_distance(Y_predicted , Y_target))

def sigmoid(x : np.array):
    return 1 /  (1 + np.e ** -x) 

def binary_crossentropy(y_predicted , y_target):
    N , _ = y_predicted.shape
    return -1/N * np.sum(y_target * np.log(y_predicted) + (1-y_target) * np.log(1-y_predicted))

class Metrices:
    def __init__(self,):
        # incase of needing to calculate f1 score
        # i will made the values saved once calculated
        self.precesion_val = None
        self.recall_val = None
        self.f1_score_val = None
        self.accuracy_val = None


    def accuracy(self , prediction:np.array , target:np.array ) -> float:
        """
        this function used to mesure the ac of the predicted label takes two argument\n
        prediction \n
        target
        """
        if self.accuracy_val is not None:
            return self.accuracy_val
        
        self.accuracy_val = np.mean(prediction.reshape(-1) == target)

        return self.accuracy_val

    def recall(self, prediction:np.array , target : np.array) -> float :
        """
        This function used to mesure the the positive label out  all acual positive label \n
        "equation" : tp / tp + fn 
        """
        # return the recall if it's already calculated
        if self.recall_val is not None:
            return self.recall_val
        
        tp_mask = target == 1
        # print(tp_mask)
        tp = (prediction.reshape(-1) == target)[tp_mask].sum()
        # print(tp)
        fn_mask = target == 1
        # print(fn_mask)
        fn = (prediction.reshape(-1) !=target)[fn_mask].sum()
        # print(fn)


        self.recall_val = tp / (tp + fn)
        return self.recall_val

    def precesion(self, prediction:np.array , target : np.array) -> float :
        """
        This function used to mesure the model predicted positive label out  all predicted positive label \n
        "equation" : tp / tp + fp
        """
        if self.precesion_val is not None:
            return self.precesion_val
        
        tp_mask = target == 1
        tp = (prediction.reshape(-1) == target)[tp_mask].sum()
        fp_mask = target == 0
        fp = (prediction.reshape(-1) !=target)[fp_mask].sum()

        self.precesion_val  = tp / (tp + fp) 
        return self.precesion_val

    def f1_score(self, prediction:np.array , target : np.array) -> float :

        # return the f1score incase it's precalculated
        if self.f1_score_val is not None:
            return self.f1_score
        # calculate the precesion incase it's not calculated
        if self.precesion_val is None:
          self.precesion_val = self.precesion(prediction , target)
        # calculate the recall incase it's not calculated
        if self.recall_val is None:
          self.recall_val = self.recall(prediction , target)

        self.f1_score_val = 2*((self.precesion_val * self.recall_val) / (self.precesion_val + self.recall_val))
        return self.f1_score_val
        

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

        return self.y[k_nearest]
        
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
            y_prediction = sigmoid(self.x.dot(self.w))
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
            









# Testing
# logisticRegression = LogisticRegression(10 , show_logs=True)
# metric = Metrices()
# x = np.random.randn(5,10)


# y = np.random.randint(0,2,(5,1))

# x1 = np.random.randn(5,10)
# y1 = np.random.randint(0,2,(5,))



# logisticRegression.fit(x,y)
# print("the labels:\n")
# print(y1)
# print(f"predicted : ")
# prediction , predicted_labels = logisticRegression.predict(x1)
# print(predicted_labels)
# print(f"the recall : {metric.recall(predicted_labels , y1)}")
# print(f"the precision : {metric.precesion(predicted_labels , y1)}")
# print(f"the f1score : {metric.f1_score(predicted_labels , y1)}")
# print(f"the accuracy : {metric.accuracy(predicted_labels , y1)}")


