import sys
import numpy as np
import tensorflow as tf

from tensorflow.keras import layers
from .BaseClassifier import BaseClassifier
from .ConstraintEstimator import ConstraintEstimator
from .utilities import convert_to_ovr_signals, majority_vote_signal
from .log import Logger


class DataConsistency(BaseClassifier):
    """
    Data Consistent Weak Supervision

    This class implements Data Consistency training for weakly supervised learning

    Parameters
    ----------
    max_iter : int, default=300
       max number of iterations to train model

    max_stagnation : int, default=100
        number of epochs without improvement to tolerate
    
    log_name : string, default=None
        Specifies directory name for a logger object.
    """

    def __init__(self, max_iter=1000, max_stagnation=100, log_name=None,):

        """
        Logging is done via TensorBoard. 
        The suggested storage format for each run is by the date/time the expirment was started, 
        and then by dataset, and then by algorithm.
        """
        
        self.max_iter = max_iter 
        self.max_stagnation = max_stagnation

        if log_name is None:
            self.logger = None
        elif type(log_name) is str:
            self.logger = Logger("logs/" + log_name)
        else:
            sys.exit("Not of string type")        

        self.model = None
        self.constraints = None


    def _simple_nn(self, dimension, output):
        """ 
        Data consistent model

        Parameters
        ----------
        dimension: int
            num of features for the data

        output: int
            number of classes

        Returns
        -------
        model: data consistent weak supervision model
        """

        actv = 'softmax' if output > 1 else 'sigmoid'
        model = tf.keras.Sequential()
        model.add(layers.Dense(512, activation='relu', input_shape=(dimension,)))
        model.add(layers.Dropout(0.2))
        model.add(layers.Dense(output, activation=actv))
        return model


    def _consistency_loss(self, model, X, reg_labels, a_matrix, bounds, slack, gamma, C):

        """
        Consistency loss function

        Parameters
        ----------
        model: Sequential model
         
        X: tensor of shape (num_examples, num_features)
            training data
         
        reg_labels: (num_examples, num_classes)
            regularization labels
         
        a_matrix: ndarray of shape (num_weak_signals, num_examples,  num_class)
            left bounds on the data
        
        bounds: ndarray of shape (num_weak_signals,  num_class)
            right bounds on the data

        slack: tensor of shape (num_weak_signals, num_class)
            linear slack to adaptively relax the constraints

        gamma: tensor of shape (num_weak_signals, num_class)
            gamma constant
        
        C: tf.Tensor(10.0, shape=(), dtype=float32)
            tensor constant

        Returns
        -------
        lagragian_objective: a tensor float32
            the loss value

        constraint_violation: a tensor float32 of shape (1,)
            How much the constraints are currently being violated 
        """

        m, n, k = a_matrix.shape
        lagragian_objective = tf.zeros(k)
        constraint_violation = tf.zeros(k)
        i = 0
        Y = model(X, training=True)

        primal = tf.divide(tf.nn.l2_loss(Y - reg_labels), n)
        primal = tf.add(primal, tf.multiply(C, tf.reduce_mean(slack)))
        for A in a_matrix:
            AY = tf.reduce_sum(tf.multiply(A, Y), axis=0)
            violation = tf.add(bounds[i], slack[i])
            violation = tf.subtract(AY, violation)
            value = tf.multiply(gamma[i], violation)
            lagragian_objective = tf.add(lagragian_objective, value)
            constraint_violation = tf.add(constraint_violation, violation)
            i += 1
        lagragian_objective = tf.add(primal, tf.reduce_sum(lagragian_objective))

        return lagragian_objective, constraint_violation


    def fit(self, X, weak_signals, weak_signals_error_bounds=None, reg_labels=None):
        """
        Estimate parameters for the label model

        Parameters
        ----------
        :param X: training data
        :type X: ndarray
        :param weak_signals: weak signals for the data
        :type  weak_signals_probas: ndarray
        :param weak_signals_error_bounds: error bounds of the weak signals
        :type  error_bounds: ndarray
        :param reg_labels: regularization labels
        :type  reg_labels: ndarray

        """
        weak_signals = convert_to_ovr_signals(weak_signals)
        m, n, k = weak_signals.shape

        cons = ConstraintEstimator()
        self.constraints = cons.error_constraint(weak_signals, weak_signals_error_bounds)

        if reg_labels is None:
            reg_labels = majority_vote_signal(weak_signals)

        a_matrix = tf.constant(self.constraints['A'], dtype=tf.float32)
        b = tf.constant(self.constraints['b'], dtype=tf.float32)
        X = tf.cast(X, dtype=tf.float32)
        model = self._simple_nn(X.shape[1], k)

        gamma = np.random.rand(m, k)
        gamma = tf.Variable(gamma.astype(np.float32), trainable=True,
                            constraint=lambda x: tf.clip_by_value(x, 0, np.infty))
        slack = np.zeros(b.shape, dtype="float32")
        slack = tf.Variable(slack, trainable=True,
                            constraint=lambda x: tf.clip_by_value(x, 0, np.infty))
        reg_labels = tf.constant(reg_labels, dtype=tf.float32)
        C = tf.constant(10, dtype=tf.float32)


        train_loss_results = []
        train_accuracy_results = []
        adam_optimizer = tf.keras.optimizers.Adam()
        sgd_optimizer = tf.keras.optimizers.SGD()

        best_viol, best_iter = np.inf, self.max_iter
        early_stop = False

        for iters in range(self.max_iter):
            if early_stop:
                break

            with tf.GradientTape() as tape:
                loss_value, constraint_viol = self._consistency_loss(
                    model, X, reg_labels, a_matrix, b, slack, gamma, C)
                model_grad, gamma_grad, slack_grad = tape.gradient(
                    loss_value, [model.trainable_variables, gamma, slack])

            adam_optimizer.apply_gradients(zip(model_grad, model.trainable_variables))
            sgd_optimizer.apply_gradients(zip([slack_grad], [slack]))
            sgd_optimizer.apply_gradients(zip([-1 * gamma_grad], [gamma]))

            # check primal feasibility
            constraint_viol = tf.reduce_sum(
                constraint_viol[constraint_viol > 0]).numpy()

            #log values
            if self.logger is not None and iters % 50 == 0:
                with self.logger.writer.as_default():
                    self.logger.log_scalar("Loss", loss_value, iters)
                    self.logger.log_scalar("Violation", constraint_viol, iters)

            # if iters % 50 == 0:
            #     print("Iter {:03d}:, Loss: {:.3}, Violation: {:.3}".format(
            #     iters, loss_value, constraint_viol))

            # check if nothing is improving for a while, or save last improvment
            if best_iter < iters - self.max_stagnation and best_viol < 1e-8:
                early_stop = True
            if constraint_viol < best_viol:
                best_viol, best_iter = constraint_viol, iters

        self.model = model


    def predict_proba(self, X):
        """
        Computes probabilistic labels for the training data

        Parameters
        ----------
        X : ndarray of training data


        Returns
        -------
        probas : ndarray of label probabilities

        """

        return self.model(X)


    def predict(self, X):
        """
        Computes predicted classes for the training data.

        Parameters
        ----------
        X : ndarray of training data

        Returns
        -------
        predicted classes : ndarray array of predicted classes
        """
        proba = self.predict_proba(X)
        proba = np.squeeze(proba)
        if len(proba.shape)==1:
            return np.round(proba)
        return np.argmax(proba, axis=-1)
