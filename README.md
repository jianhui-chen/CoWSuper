# Weakly-Supervised-Learning
This is a package that produces labels using weakly supervised learning with constraint-based methods.

The package contains 2 algorithms, Data Consistent Weak Supervision (DCWS) and Constrained Label Learning (CLL), 
that are contains code for the following papers

    * Constrained Labeling for Weakly Supervised Learning
    * Data Consistency for Weakly Supervised Learning

If you use this work in an academic study, please cite our paper

# Requirements

The library is tested in Python 3.6 and 3.7. 

Its main requirements are Tensorflow and numpy. 

Scikit-learn is required to run the experiments.

# Examples

We have provided a run_experiment file as an example on both algorithms, along the real datasets. They can all be found under the examples folder.

# Logging

Logging is done via TensorBoard. The suggested storage format for each run is by the date/time the expirment was started, and then by dataset, and then by algorithm. Use:

tensorboard --logdir=logs/data_and_time/data_set/algorithm

Example: 

tensorboard --logdir=logs/2021_07_28-05:50:52_PM/breast-cancer/CLL

Enjoy!
