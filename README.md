# Weakly-Supervised-Learning
This package contains code for the following papers

    * Constrained Labeling for Weakly Supervised Learning
    * Data Consistency for Weakly Supervised Learning


If you use this work in an academic study, please cite our paper

```
@inproceedings{arachie2019adversarial,
  title={Adversarial label learning},
  author={Arachie, Chidubem and Huang, Bert},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={33},
  pages={3183--3190},
  year={2019}
}
```

# Requirements

The library is tested in Python 3.6 and 3.7. 

Its main requirements are Tensorflow and numpy. 

Scikit-learn is required to run the experiments.

# Algorithms

CLL:
Is built off of the Constraint Estimator abstract class in ConstraintEstimators.py. The most important script is the ConstraintEstimators.py script that contains implementation of the algorithm inside of the CLL class.

Data Consistancy:
Is built off of the Label Estimator abstract class in LabelEstimators.py. The most important script is the LabelEstimators.py script that contains implementation of the algorithm inside of the DataConsistency class.

# Usage

# Examples

We have provided a run_experiment that runs experiments on the real datasets provided on both algorithms. Running experiment on other user datasets is fairly easy to implement.

# Logging

Logging is done via TensorBoard. The suggested storage format for each run is by the date/time the expirment was started, and then by dataset, and then by algorithm. Use:

tensorboard --logdir=logs/data_and_time/data_set/algorithm

Example: 

tensorboard --logdir=logs/2021_07_28-05:50:52_PM/breast-cancer/CLL


# Models

CLL:

Data Consistancy:

# Bounds

NOTE: FIX THIS
ALL:
The model is set to use the True bounds of the data. When this bounds is unknown, the user can provide an upper bounds for the weak signals or use constant bounds in the experiments scripts

MultiALL:

CLL:

Data Consistancy:


# Limitations

The ALL algorithm only supports binary classification and weak signals that do not abstain, code for MultiALL fixes these limitations and returns similar results to that of ALL. For now, we skip over running these data sets on 
