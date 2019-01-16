from sklearn.preprocessing import MinMaxScaler


class Normalize(object):

    def normalize(self, X_train, X_test, X_train_full):
        self.scaler = MinMaxScaler()
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        X_train_full_norm = self.scaler.transform(X_train_full)
        return (X_train, X_test, X_train_full_norm)

    def inverse(self, X_train, X_test):
        X_train = self.scaler.inverse_transform(X_train)
        X_test = self.scaler.inverse_transform(X_test)
        return (X_train, X_test)