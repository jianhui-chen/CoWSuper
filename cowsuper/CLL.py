import sys
import numpy as np
import tensorflow as tf

from .BaseClassifier import BaseClassifier
from .ConstraintEstimator import ConstraintEstimator
from .utilities import convert_to_ovr_signals
from .log import Logger

class CLL(BaseClassifier):
    """
    Constrained Label Learning

    This class implements CLL training on a set of data

    Parameters
    ----------
    max_iter : int, default=300
        Maximum number of iterations taken for solvers to converge.
    
    log_name : string, default=None
        Specifies directory name for a logger object.
    """

    def __init__(self, max_iter=300, log_name=None,):
        
        #get rid of trials

        """
        Logging is done via TensorBoard. 
        The suggested storage format for each run is by the date/time the expirment was started, 
        and then by dataset, and then by algorithm.
        """

        self.max_iter = max_iter

        if log_name is None:
            self.logger = None
        elif type(log_name) is str:
            self.logger = Logger("logs/" + log_name) 
        else:
            sys.exit("Not of string type")
        self.constraints = None
        
        
    def _run_constraints(self, y, error_constraints):
        """
        Run constraints from CLL

        :param y: Random starting values for labels
        :type  y: ndarray 
        :param error_constraints: error constraints (a_matrix and bounds) of the weak signals 
        :type  error_constraints: dictionary

        :return: estimated learned labels
        :rtype: ndarray
        """
        a_matrix = error_constraints['A']
        bounds = error_constraints['b']
        
        adam_optimizer = tf.keras.optimizers.Adam()
        
        for iter in range(self.max_iter):
            
            with tf.GradientTape() as tape:
                
                constraint = tf.Variable(tf.zeros(bounds.shape), tf.int32)
                for i, current_a in enumerate(a_matrix):
                    constraint = tf.add(constraint, tf.reduce_sum(tf.multiply(current_a, y), axis=0))
                    loss = tf.subtract(constraint, bounds)
                    
                grads = tape.gradient(loss, y)
        
            adam_optimizer.apply_gradients(zip([grads], [y]))

            # log current data 
            if self.logger is not None and iter % 10 == 0:
                with self.logger.writer.as_default():
                    self.logger.log_scalar("y", np.average(y), iter)
                    self.logger.log_scalar("loss", np.average(loss), iter)
        return y
    

    def fit(self, weak_signals, weak_signals_error_bounds=None):
        """
        Finds estimated labels

        Parameters
        ----------
        :param weak_signals: weak signals for the data
        :type  weak_signals_probas: ndarray 
        :param weak_signals_error_bounds: error bounds of the weak signals
        :type  error_bounds: ndarray

        """
        cons = ConstraintEstimator(error_threshold=0)
        self.constraints = cons.error_constraint(weak_signals, weak_signals_error_bounds)

        weak_signals = convert_to_ovr_signals(weak_signals)
        m, n, num_classes = weak_signals.shape

        # initialize y and lists
      
        self.ys = []

        y = np.random.rand(n, num_classes)
        y = tf.Variable(y.astype(np.float32), trainable=True, constraint=lambda x: tf.clip_by_value(x, 0, 1))  
        self.ys.append(self._run_constraints(y, self.constraints))
            
        self.ys = np.array(self.ys)

    def predict_proba(self, indices=None):
        """
        Computes probability estimates for given class
        Should be able to be extendable for multi-class implementation

        Parameters
        ----------
        indices : the indices of desired label

        Returns
        -------
        probas : ndarray of label probabilities

        """        
        if indices == None:
            return np.squeeze(np.mean(self.ys, axis = 0))
        else:
            return np.squeeze(np.mean(self.ys[indices, :], axis = 0))


    def predict(self, indices=None):
        """
        Computes predicted classes for the weak signals.

        Parameters
        ----------
        indices : the indices of desired label

        Returns
        -------
        predicted classes : ndarray array of predicted classes
        """
        return np.argmax(self.predict_proba(indices), axis=-1)