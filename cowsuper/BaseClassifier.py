import numpy as np

from abc import ABC, abstractmethod
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score


class BaseClassifier(ABC):
    """
    Abstract Base Class for learning classifiers

    Constructors are all defined in subclasses

    """

    @abstractmethod
    def predict(self, X):
        """
        Computes predicted classes for the weak signals.

        Parameters
        ----------
        weak_signals : ndarray of either weak signals or data depending on the algorithm

        Returns
        -------
        predicted classes : ndarray array of predicted classes
        """
        pass


    def get_score(self, true_labels, predicted_labels, metric):
        """
        Computes metrics of predicted labels based on the true labels.

        Parameters
        ----------
        true_labels : ndarray

        predicted_labels : ndarray

        metric : supports accuracy and F1 score

        Returns
        -------
        score : float
            Value between 0 to 1.0

        """
        assert true_labels.shape == predicted_labels.shape, "True labels and predicted labels shape do not match"
        if len(predicted_labels.shape) == 2:
            true_labels = np.argmax(true_labels, axis=-1)
            predicted_labels = np.argmax(predicted_labels, axis=-1)

        if metric == 'f1':
            return f1_score(true_labels, np.round(predicted_labels), average='micro')
        return accuracy_score(true_labels, np.round(predicted_labels))


    @abstractmethod
    def predict_proba(self, X):
        """
        Computes probability estimates for given class
        Should be able to be extendable for multi-class implementation

        Parameters
        ----------
        X : ndarray of either weak signals or data depending on the algorithm

        Returns
        -------
        probas : ndarray of label probabilities

        """
        pass


    @abstractmethod 
    def fit(self, weak_signals, weak_signal_error_bounds):
        """
        Abstract method to fit models

        Parameters
        ----------
        weak_signals : ndarray of weak signals

        weak_signal_error_bounds : ndarray of weak signals error bounds
        """
        pass
