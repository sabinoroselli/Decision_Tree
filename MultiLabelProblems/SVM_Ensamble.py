from sklearn.svm import SVC
import numpy as np

class MultiLabel_SVM:
    def __init__(self,random_state,C):
        self.RS = random_state
        self.regularization_coeff = C
        self.model = {}
        self.failed = False

    def fit(self,X,Y):
        for i in range(Y.shape[1]):
            self.model.update({
                i:SVC(
                    kernel='linear',
                    random_state=self.RS,
                    C=self.regularization_coeff
                )
                               })
        for i in range(Y.shape[1]):
            if len(np.unique(Y[:,i])) > 1:
                self.model[i].fit(X,Y[:,i])
            else:
                self.failed = True
        return None

    def predict(self,X):
        predicted = []
        for x in X:
            # print(x)
            prediction = [int(i.predict([x])) for i in self.model.values()]
            # print(prediction)
            predicted.append(prediction)
        return predicted
