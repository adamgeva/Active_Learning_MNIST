import time
import numpy as np
from sklearn import metrics


class Trainer:

    def __init__(self, model_object):
        self.accuracies = []
        self.model_object = model_object()

    def print_model_type(self):
        print (self.model_object.model_type)

    # we train normally and get probabilities for the validation set.
    # i.e., we use the probabilities to select the most uncertain samples
    def train(self, X_train, y_train, X_test, c_weight):
        print('Train set:', X_train.shape, 'y:', y_train.shape)
        print('Test  set:', X_test.shape)
        t0 = time.time()

        # fit model to trainig data
        self.model_object.fit(X_train, y_train, c_weight)

        self.test_y_predicted = self.model_object.predict(X_test)
        self.run_time = time.time() - t0

    def get_probabilities(self, X):
        return self.model_object.predict_probabilities(X)

    # we want accuracy only for the test set
    def get_test_accuracy(self, i, y_test):
        classif_rate = np.mean(self.test_y_predicted.ravel() == y_test.ravel()) * 100
        self.accuracies.append(classif_rate)
        print('--------------------------------')
        print('Iteration:', i)
        print('--------------------------------')