# Active_Learning_MNIST

This code performs sequential active learning on the MNIST dataset

# The following packages were used:

Keras                         2.2.4                 
matplotlib                    2.2.2                 
numpy                         1.14.3                
scikit-learn                  0.20.2                
scipy                         1.2.0                 
sklearn                       0.0                   
tensorflow                    1.12.0   

# To run:
Modify the file Config.py according to the desired run:
A selection of model, sampling method, and Ks

Run main.py

The results will be saved in a dict and dumped to a json file (as defined in Config.py)
In order to visualize the results afterwards - run plotting.py

# Previous results:
To plot results of previous experiments, run plot_exp.py. This will collect the results from the folder experiments/ and visualize them.


             
