from Models import *
from SelectionFunc import *


experiment_name = 'exp15'

# set parameters
max_to_query = 500
train_size = 60000  # out of 70k examples
num_of_classes = 10

# used in Entropy cluster sampling method
num_of_clusters = 100

# models to test
# models = [LinearSVC, RFModel, LogModel, SimpleCNN]
models = [RFModel]

# query functions to test
# selection_functions = [RandomSelection, MarginSamplingSelection, EntropySelection, LeastConfidence, EntropySelectionClustering]
selection_functions = [RandomSelection, MarginSamplingSelection]

# frac of uncertain samples for entropy selection
# frac = 0.4

# number of queries per iteration
Ks = [25]

# calculated by training on all training data - 6000 examples
cnn_supervised = 99.
random_supervised = 97.
svc_supervised = 94.
