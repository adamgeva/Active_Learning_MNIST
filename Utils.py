import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
import numpy as np


# download MNIST dataset
def download():
    # Load data from https://www.openml.org/d/554
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
    print ('MNIST:', X.shape, y.shape)
    return (X, y)


# split into train and test
def split(train_size, X, y):
    X_train = X[:train_size]
    y_train = y[:train_size]
    X_test = X[train_size:]
    y_test = y[train_size:]
    return (X_train, y_train, X_test, y_test)


# random selection of initial samples form entire set
def get_k_random_samples(initial_labeled_samples, X_train_full, y_train_full, trainset_size):
    np.random.seed(100)
    selection = np.random.choice(trainset_size,
                                   initial_labeled_samples,
                                   replace=False)

    print ('initial random chosen samples', selection.shape),

    X_train = X_train_full[selection]
    y_train = y_train_full[selection]
    X_train = X_train.reshape((X_train.shape[0], -1))
    return (selection, X_train, y_train)


# print results
def performance_plot(fully_supervised_accuracy, dic, models, selection_functions, Ks):
    fig, ax = plt.subplots()
    ax.plot([0, 500], [fully_supervised_accuracy, fully_supervised_accuracy], label='Fully Supervised')
    for model_object in models:
        for selection_function in selection_functions:
            for idx, k in enumerate(Ks):
                x = np.arange(float(Ks[idx]), 500 + float(Ks[idx]), float(Ks[idx]))
                accs = np.array(dic[model_object][selection_function][k][0])
                ax.plot(x, accs, label=model_object + '-' + selection_function + '-' + str(k))
    ax.legend()
    ax.set_xlim([50, 500])
    ax.set_ylim([40, 100])
    ax.grid(True)
    plt.xlabel('Labeled Examples')
    plt.ylabel('Accuracy')
    plt.show()
