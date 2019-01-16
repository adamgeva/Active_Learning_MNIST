import json
import numpy as np

from ActiveLearn import ActiveLearn
import Utils
import Config


# download the mnist dataset
(X, y) = Utils.download()

# split the total data into test and train
(X_train_full, y_train_full, X_test, y_test) = Utils.split(Config.train_size, X, y)


print('train:', X_train_full.shape, y_train_full.shape)
print('test :', X_test.shape, y_test.shape)
classes = len(np.unique(y))
print('unique classes', classes)


# res collects all the resulting accuracies of all [model, selection, k] combination
res = {}

# counts the number of runs
run_num = 0

# iterate over all combinations of [k ,model, selection_func] and perform active learning
for model_object in Config.models:
    if model_object.__name__ not in res:
        res[model_object.__name__] = {}

    for selection_function in Config.selection_functions:
        if selection_function.__name__ not in res[model_object.__name__]:
            res[model_object.__name__][selection_function.__name__] = {}
            # initialize the selection function with the training data:
            curr_selection_func = selection_function(X_train_full)

        for k in Config.Ks:
            res[model_object.__name__][selection_function.__name__][str(k)] = []

            print('Run = %s, using model = %s, selection_function = %s, k = %s.' % (
                run_num, model_object.__name__, selection_function.__name__, k))

            alg = ActiveLearn(k, model_object, curr_selection_func)
            accs = alg.run(X_train_full, y_train_full, X_test, y_test, Config.max_to_query, Config.train_size)

            res[model_object.__name__][selection_function.__name__][str(k)].append(accs)
            run_num += 1

            print('***Run Complete***')


# dump the experiments results to json file
print(res)
with open(Config.experiment_name + '.json', 'w') as outfile:
    json.dump(res, outfile, indent=2, sort_keys=True)

# read the results and plot graphs
with open(Config.experiment_name + '.json') as json_file:
    results = json.load(json_file)
print(results)
