import json
import numpy as np 

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dropout, Dense


def read_text_data(datapath):
    """ Read text datasets """

    train_data = np.load(datapath + 'data_features.npy', allow_pickle=True)[()]
    weak_signals = np.load(datapath + 'weak_signals.npy', allow_pickle=True)[()]
    train_labels = np.load(datapath + 'data_labels.npy', allow_pickle=True)[()]
    test_data = np.load(datapath + 'test_features.npy', allow_pickle=True)[()]
    test_labels = np.load(datapath + 'test_labels.npy', allow_pickle=True)[()]

    if len(weak_signals.shape) == 2:
        weak_signals = np.expand_dims(weak_signals.T, axis=-1)

    # if weak_signals.shape[2] == 1:
    #     train_labels = np.squeeze(train_labels)
    #     test_labels = np.squeeze(test_labels)

    data = {}
    data['train'] = train_data, train_labels
    data['test'] = test_data, test_labels
    data['weak_signals'] = weak_signals
    return data


def writeToFile(data, filename):
    """ Save data in json format """

    json.dump(data,
              codecs.open(filename, 'w', encoding='utf-8'),
              separators=(',', ':'),
              sort_keys=True,
              indent=4)


def majority_vote_signal(weak_signals):
    """
    Calculate majority vote labels for the weak_signals

    Parameters
    ----------
    weak_signals: ndarray of shape (num_weak, num_examples, num _class)
        weak signal probabilites containing -1 for abstaining signals, and between
        0 to 1 for non-abstaining

    Returns
    -------
    mv_weak_labels: ndarray of shape (num_examples, num_class)
        fitted  Data consistancy algorithm
    """

    baseline_weak_labels = np.rint(weak_signals)
    mv_weak_labels = np.ones(baseline_weak_labels.shape)

    mv_weak_labels[baseline_weak_labels == -1] = 0
    mv_weak_labels[baseline_weak_labels == 0] = -1

    mv_weak_labels = np.sign(np.sum(mv_weak_labels, axis=0))
    break_ties = np.random.randint(2, size=int(np.sum(mv_weak_labels == 0)))
    mv_weak_labels[mv_weak_labels == 0] = break_ties
    mv_weak_labels[mv_weak_labels == -1] = 0
    return mv_weak_labels


def convert_to_ovr_signals(weak_signals):
    """
        Convert different numbers of weak signals per class to same size

        Parameters
        ----------
        weak_signals: ndarray of shape (num_weak, num_examples, num _class)
            weak signal probabilites containing -1 for abstaining signals, and between
            0 to 1 for non-abstaining

        Returns
        -------
        weak_signals
    """
    if weak_signals.ndim == 2: #check if the input array is 2D
        m, n  = weak_signals.shape #find the number of examples and weak signa;s
        flatten = np.ndarray.flatten(weak_signals) #flatten out the array
        k = np.max(flatten) + 1 #find the number of classes
        final = [] #making an empty list

        for i in range(0, k): #the number of classes
            for j in range(m*n):  #the flatten array
                if flatten[j] == i: 
                    final.append(1)
                elif flatten[j] == -1: 
                    final.append(-1)
                else: 
                    final.append(0)

        weak_signals = np.reshape(final, (k , m , n))
        return weak_signals

    elif weak_signals.ndim == 3: #check if the input array is 3D
        flatten = np.ndarray.flatten(weak_signals) #flatten out the array
        if all(p == -1 or 0 <= p <= 1 for p in np.ndarray.flatten(flatten)): #check if the format is correct, i.e. -1 or between 0 and 1
            return weak_signals
        else:
            raise ValueError("incorrect format for weak signal inputs, the value should be -1 or between 0 and 1 ")

    else: #if the input array is not 2D or 3D
        raise ValueError("incorrect dimension for weak signal inputs")


def mlp_model(dimension, output):
    """
        Builds Simple MLP model

        Parameters
        ----------
        :param dimension: amount of input
        :type  dimension: int
        :param output: amount of final states
        :type  output: int

        Returns
        -------
        :returns: Simple MLP
        :return type: Sequential tensor model
    """
    actv = 'sigmoid' if output == 1 else 'softmax'
    loss = 'binary_crossentropy' if output == 1 else 'categorical_crossentropy'
    model = Sequential()
    model.add(Dense(512, activation='relu', input_shape=(dimension,)))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(output, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                optimizer='adagrad', metrics=['accuracy'])

    return model