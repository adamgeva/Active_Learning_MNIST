import numpy as np
from sklearn.utils import check_random_state
from scipy.stats import entropy

from sklearn.cluster import MiniBatchKMeans

import Config


class BaseSelectionFunction(object):

    def __init__(self, X_train_full):
        self.X_train_full = X_train_full
        self.num_of_clusters = Config.num_of_clusters

    def select(self):
        pass

# recieves: X_train_full - the entire training set (some is still unlabeled)
#           probas_val   - the probilities for classes of the samples in he unlabeled training set
#           k            - the number of samples to query


class RandomSelection(BaseSelectionFunction):

    def select(self, probas_full, selected_so_far, k):
        random_state = check_random_state(0)
        all = np.arange(probas_full.shape[0])
        np.random.shuffle(all)
        not_selected = [i for i in all if i not in selected_so_far]
        selection = not_selected[:k]
        return selection


class EntropySelectionClustering(BaseSelectionFunction):

    def __init__(self, X_train_full):
        super().__init__(X_train_full)
        # cluster the entire training data
        self.cluster_model = MiniBatchKMeans(n_clusters=self.num_of_clusters)
        self.cluster_model.fit(self.X_train_full)
        unique, counts = np.unique(self.cluster_model.labels_, return_counts=True)
        self.cluster_prob = counts / sum(counts)
        self.cluster_labels = self.cluster_model.labels_

    def select(self, probas_full, selected_so_far, k):

        ent = entropy(np.transpose(probas_full))
        ent_sorted = (np.argsort(ent)[::-1])

        rank_ind = [i for i in ent_sorted if i not in selected_so_far]

        new_selection_cluster_counts = [0 for _ in range(self.num_of_clusters)]
        new_selection = []
        for i in rank_ind:
            if len(new_selection) == k:
                break
            label = self.cluster_labels[i]
            if new_selection_cluster_counts[label] / k < self.cluster_prob[label]:
                new_selection.append(i)
                new_selection_cluster_counts[label] += 1
        n_slot_remaining = k - len(new_selection)
        batch_filler = list(set(rank_ind) - set(selected_so_far) - set(new_selection))
        new_selection.extend(batch_filler[0:n_slot_remaining])

        return new_selection


class EntropySelection(BaseSelectionFunction):

    def select(self, probas_full, selected_so_far, k):

        ent = entropy(np.transpose(probas_full))
        ent_sorted = (np.argsort(ent)[::-1])

        rank_ind = [i for i in ent_sorted if i not in selected_so_far]

        selection = rank_ind[:k]

        return selection


class MarginSamplingSelection(BaseSelectionFunction):

    def select(self, probas_full, selected_so_far, k):
        rev = np.sort(probas_full, axis=1)[:, ::-1]
        values = rev[:, 0] - rev[:, 1]
        values_sorted_ind = np.argsort(values)
        rank_ind = [i for i in values_sorted_ind if i not in selected_so_far]
        selection = rank_ind[:k]
        return selection


class LeastConfidence(BaseSelectionFunction):

    def select(self, probas_full, selected_so_far, k):
        max_prob = np.max(probas_full, axis=1)
        values_sorted_ind = np.argsort(max_prob)
        rank_ind = [i for i in values_sorted_ind if i not in selected_so_far]
        selection = rank_ind[:k]
        return selection
