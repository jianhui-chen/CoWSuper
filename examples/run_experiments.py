from utilities import *

# Import model
from CLL import CLL
from DCWS import DataConsistency

"""
    Binary Datasets:
        1. SST-2
        2. IMDB
        3. Cardio
        4. Breast Cancer
        5. Yelp
    
    Multi-Class Datasets:
        1. Fashion Mnist
        2. SVHN

    Algorithms:
        1. CLL     
        2. DCWS

    Weak Signals:
        -1 for abstain, 1 for positive labels and 0 for negative labels
        Supports 2D weak signals for both binary and multi-class data: num_examples vs num_signals
            For multi-class the signals are given as class labels [0, 1, 2, 3]
            -1 is abstain
            
        Supports One-vs-rest weak signals: num_signals vs num_examples vs num_classes
            Weak signals are 3D array
            See run experiments

"""

def run_experiments(dataset):
    """ 
        sets up and runs experiments on various algorithms

        Parameters
        ----------
        dataset : dictionary of ndarrays
            contains training set, testing set, and weak signals 
            of read in data
        
        Returns
        -------
        nothing
    """

    train_data, train_labels = dataset['train']
    test_data, test_labels = dataset['test']
    weak_signals = dataset['weak_signals']
    m, n, num_classes = weak_signals.shape
    
    ###################################################################
    ### CLL
    
    cll = CLL()
    cll.fit(weak_signals)
    predicted_proba = cll.predict_proba()
    predicted_labels = cll.predict()
     
    print(f"The train accuracy of CLL is: {cll.get_score(train_labels, predicted_proba, metric='accuracy')}")
    print()
    
    
    ###################################################################
    ### DCWS
    
    dcws = DataConsistency()
    dcws.fit(train_data, weak_signals)
    predicted_proba = np.squeeze(dcws.predict_proba(train_data))
    predicted_labels = dcws.predict(train_data)

    print(f"The train accuracy of DCWS is: {dcws.get_score(train_labels, predicted_proba, metric='accuracy')}")
    test_pred = np.squeeze(dcws.predict_proba(test_data))
    print(f"The test F-score of DCWS is: {dcws.get_score(test_labels, test_pred, metric='f1')}")
    print()
    
    # ###################################################################
    #### Train an end model
    
    model = mlp_model(train_data.shape[1], num_classes)
    model.fit(train_data, train_labels, batch_size=32, epochs=20, verbose=1)
    test_predictions = np.squeeze(model.predict(test_data))
    print(f"The test accuracy is: {dcws.get_score(test_labels, test_predictions, metric='accuracy')}")
    

if __name__ == '__main__':
    # print("Running synthetic experiments...")
    print("Running real experiments...")
    
    # text and tabular experiments:
    # run_experiments(read_text_data('../datasets/imbd/'))
    # run_experiments(read_text_data('../datasets/yelp/'))
    # run_experiments(read_text_data('../datasets/sst-2/'))
    # run_experiments(load_svhn(),'svhn')
    # run_experiments(load_fashion_mnist(),'fmnist')


    # experiments for datasets used in ALL
    # run_experiments(read_text_data('../datasets/breast-cancer/'))
    # run_experiments(read_text_data('../datasets/obs-network/'))
    # run_experiments(read_text_data('../datasets/cardiotocography/'))
    # run_experiments(read_text_data('../datasets/phishing/'))
