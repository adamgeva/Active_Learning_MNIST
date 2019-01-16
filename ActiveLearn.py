from Utils import *
from Normalize import Normalize
from Trainer import Trainer


#  for a given model, number of samples per iteration k, and a selection function, this class is
#  in charge of performing the Active learning algorithm
class ActiveLearn(object):
    accuracies = []

    def __init__(self, k_samples, model_object, selection_function):
        self.k_samples = k_samples
        self.model_object = model_object
        self.sample_selection_function = selection_function

    def run(self, X_train_full, y_train_full, X_test, y_test, max_queried, trainset_size):

        # train initial classifier on randomly sampled examples
        (selected_so_far, X_train, y_train) = \
            get_k_random_samples(self.k_samples, X_train_full, y_train_full, trainset_size)
        self.queried = self.k_samples
        self.samplecount = [self.k_samples]

        # normalize data
        normalizer = Normalize()
        X_train, X_test, X_train_full_norm = normalizer.normalize(X_train, X_test, X_train_full)

        trainer = Trainer(self.model_object)
        trainer.train(X_train, y_train, X_test, 'balanced')
        active_iteration = 1
        trainer.get_test_accuracy(1, y_test)

        while self.queried < max_queried:
            active_iteration += 1

            # get validation probabilities
            probas_full = trainer.get_probabilities(X_train_full_norm)

            # select samples using a selection function
            uncertain_samples = self.sample_selection_function.select(probas_full, selected_so_far, self.k_samples)

            # add uncertain samples to selected so far
            selected_so_far = np.concatenate((selected_so_far, uncertain_samples))

            # normalization needs to be inversed and recalculated based on the new train and test set.
            X_train, X_test = normalizer.inverse(X_train, X_test)

            # get the uncertain samples from the unlabeled set
            print('trainset before', X_train.shape, y_train.shape)
            X_train = np.concatenate((X_train, X_train_full[uncertain_samples]))
            y_train = np.concatenate((y_train, y_train_full[uncertain_samples]))
            print('trainset after', X_train.shape, y_train.shape)
            self.samplecount.append(X_train.shape[0])

            # normalize again after creating the 'new' train/test sets
            normalizer = Normalize()
            X_train, X_test, X_train_full_norm = normalizer.normalize(X_train, X_test, X_train_full)

            self.queried += self.k_samples
            trainer.train(X_train, y_train, X_test, 'balanced')
            trainer.get_test_accuracy(active_iteration, y_test)

        print('final active learning accuracies',
              trainer.accuracies)

        return trainer.accuracies
