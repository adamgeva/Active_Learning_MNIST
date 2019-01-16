import json
import Utils
import Config

# read the results and plot graphs
with open('experiments/exp2.json') as json_file:
    results = json.load(json_file)

Ks = ['10', '50', '100', '150']
models = ['RFModel', 'LinearSVC', 'SimpleCNN']
selections = ['EntropySelection', 'MarginSamplingSelection', 'RandomSelection']

### comparison of different K values for each model:

# RF Model:
for curr_k in Ks:
    Utils.performance_plot(Config.random_supervised, results, ['RFModel'],
                           ['EntropySelection', 'MarginSamplingSelection', 'RandomSelection'], [curr_k])

# SVM Model:
for curr_k in Ks:
    Utils.performance_plot(Config.svc_supervised, results, ['LinearSVC'],
                           ['EntropySelection', 'MarginSamplingSelection', 'RandomSelection'], [curr_k])

# SimpleCNN Model:
for curr_k in Ks:
    Utils.performance_plot(Config.cnn_supervised, results, ['SimpleCNN'],
                           ['EntropySelection', 'MarginSamplingSelection', 'RandomSelection'], [curr_k])


### comparison of different models for a fixed k and selection method:
for selection in selections:
    Utils.performance_plot(100, results, models,
                           [selection], ['10'])

### comparison of different models for a fixed k and selection method:
for selection in selections:
    Utils.performance_plot(100, results, models,
                           [selection], ['50'])


### Clustering method
# read the results and plot graphs
with open('experiments/exp9.json') as json_file:
    results = json.load(json_file)

Utils.performance_plot(Config.svc_supervised, results, ['LinearSVC'],
                           ['EntropySelection', 'EntropySelectionClustering'], ['50'])

with open('experiments/exp10.json') as json_file:
    results = json.load(json_file)

Utils.performance_plot(Config.random_supervised, results, ['RFModel'],
                           ['EntropySelection', 'EntropySelectionClustering'], ['50'])


with open('experiments/exp11.json') as json_file:
    results = json.load(json_file)

Utils.performance_plot(Config.random_supervised, results, ['RFModel'],
                           ['EntropySelection', 'EntropySelectionClustering'], ['10'])