from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K


class BaseModel(object):

    def __init__(self):
        pass

    def fit(self):
        pass

    def predict(self):
        pass

    def predict_probabilities(self):
        pass


class LinearSVC(BaseModel):
    model_type = 'Linear SCV'

    def fit(self, X_train, y_train, c_weight):
        print('training svm...')
        self.model = SVC(C=1.0, kernel='linear', probability=True, class_weight=c_weight)
        self.model.fit(X_train, y_train)

    def predict(self, X):
        y_predicted = self.model.predict(X)
        return y_predicted

    def predict_probabilities(self, X):
        y_prob = self.model.predict_proba(X)
        return y_prob


class LogModel(BaseModel):
    model_type = 'Multinominal Logistic Regression'

    def fit(self, X_train, y_train, c_weight):
        print('training multinomial logistic regression')
        train_samples = X_train.shape[0]
        self.model = LogisticRegression(
            C=50. / train_samples,
            multi_class='multinomial',
            penalty='l1',
            solver='saga',
            tol=0.1,
            class_weight=c_weight,
        )
        self.model.fit(X_train, y_train)

    def predict(self, X):
        y_predicted = self.model.predict(X)
        return y_predicted

    def predict_probabilities(self, X):
        y_prob = self.model.predict_proba(X)
        return y_prob


class RFModel(BaseModel):
    model_type = 'Random Forest'

    def fit(self, X_train, y_train, c_weight):
        print('training random forest...')
        self.model = RandomForestClassifier(n_estimators=500, class_weight=c_weight)
        self.model.fit(X_train, y_train)

    def predict(self, X):
        y_predicted = self.model.predict(X)
        return y_predicted

    def predict_probabilities(self, X):
        y_prob = self.model.predict_proba(X)
        return y_prob


class AdaBoostModel(BaseModel):
    model_type = 'AdaBoost Model'

    def fit(self, X_train, y_train, c_weight):
        print('training AdaBoost...')
        self.model = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2), n_estimators=600, learning_rate=1)
        self.model.fit(X_train, y_train)

    def predict(self, X):
        y_predicted = self.model.predict(X)
        return y_predicted

    def predict_probabilities(self, X):
        y_prob = self.model.predict_proba(X)
        return y_prob


class SimpleCNN(BaseModel):

    def __init__(self):

        self.batch_size = 24
        self.num_classes = 10
        self.epochs = 10

        # input image dimensions
        self.img_rows, self.img_cols = 28, 28

        if K.image_data_format() == 'channels_first':
            input_shape = (1, self.img_rows, self.img_cols)
        else:
            input_shape = (self.img_rows, self.img_cols, 1)

        self.model = Sequential()
        self.model.add(Conv2D(32, kernel_size=(3, 3),
                         activation='relu',
                         input_shape=input_shape))
        self.model.add(Conv2D(64, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))
        self.model.add(Flatten())
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(self.num_classes, activation='softmax'))

        self.model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adadelta(),
                      metrics=['accuracy'])

    def fit(self, X_train, y_train, c_weight):
        x_train = self.reshape_to_map(X_train)

        # convert class vectors to binary class matrices
        y_train = keras.utils.to_categorical(y_train, self.num_classes)
        self.model.fit(x_train, y_train, batch_size=self.batch_size, epochs=self.epochs, verbose=1,
                       class_weight=c_weight)

    def predict(self, X):
        x = self.reshape_to_map(X)

        y_predicted = self.model.predict_classes(x)
        y_predicted_str = np.array([str(i) for i in y_predicted])
        return y_predicted_str

    def predict_probabilities(self, X):
        x = self.reshape_to_map(X)

        y_prob = self.model.predict_proba(x)
        return y_prob

    def reshape_to_map(self, X):
        x = np.reshape(X, (-1, self.img_rows, self.img_cols))

        if K.image_data_format() == 'channels_first':
            x = x.reshape(x.shape[0], 1, self.img_rows, self.img_cols)
        else:
            x = x.reshape(x.shape[0], self.img_rows, self.img_cols, 1)
        return x

 #   print('Test loss:', score[0])
 #   print('Test accuracy:', score[1])