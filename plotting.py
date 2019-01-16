import json
import Utils
import Config

# read the results and plot graphs
with open('exp15.json') as json_file:
    results = json.load(json_file)
print(results)


models_str = [model.__name__ for model in Config.models]
selection_functions_str = [sel_func.__name__ for sel_func in Config.selection_functions]
Ks_str = [str(i) for i in Config.Ks]


### comparison of different values of K for each model:

Utils.performance_plot(Config.random_supervised, results, ['RFModel'],
                       selection_functions_str, Ks_str)


