import numpy as np

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
            return self.f1_score_val
        # calculate the precesion incase it's not calculated
        if self.precesion_val is None:
          self.precesion_val = self.precesion(prediction , target)
        # calculate the recall incase it's not calculated
        if self.recall_val is None:
          self.recall_val = self.recall(prediction , target)

        self.f1_score_val = 2*((self.precesion_val * self.recall_val) / (self.precesion_val + self.recall_val))
        return self.f1_score_val