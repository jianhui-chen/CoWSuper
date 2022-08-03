import numpy as np

from .utilities import convert_to_ovr_signals


class ConstraintEstimator():
    """
    Estimates constraints for constraint based weak supervision algorithms

    This class implements error and precision constraints

    Parameters
    ----------
    error_threshold : float, default=0.1
        Constant threshold for error bounds.

    precision_threshold : float, default=0.9
        Constant threshold for precision bounds
    """

    def __init__(self, error_threshold=0.1, precision_threshold=0.9):
        self.error_threshold = error_threshold
        self.precision_threshold = precision_threshold


    def build_constraints(self, a_matrix, bounds):
        """
        Builds the constraints as a dictionary

        Parameters
        ----------
        :param a_matrix: transformation of the weak signals q (1-2q)
        :type a_matrix: ndarray
        :param bounds: transformation of the bounds of the weak signals b (nb - q.T 1v)
        :type bounds: ndarray

        Returns
        -------
        :return: constraint set of the weak signals
        :rtype: dict
        """

        m, n, k = a_matrix.shape
        assert (m,k) == bounds.shape, \
        "The constraint matrix shapes don't match"

        constraints = dict()
        constraints['A'] = a_matrix
        constraints['b'] = bounds

        return constraints


    def error_constraint(self, weak_signals, weak_signals_error_bounds):
        """
        Calculates error constraints (Ay = b) for the weak signals

        Parameters
        ----------
        :param weak signals: weak signal of predicted labels
        :type true_labels: ndarray
        :param weak_signals_error_bounds: bounds for the weak signals error
        :type bounds: ndarray

        Returns
        -------
        :return: a matrix and bounds for error constraints
        :rtype: dict
        """
        weak_signals = convert_to_ovr_signals(weak_signals)
        m, n, k = weak_signals.shape
        if weak_signals_error_bounds is None:
            weak_signals_error_bounds = np.ones((m,k)) * self.error_threshold

        constraint_set = dict()
        error_amatrix = np.zeros((m, n, k))
        constants = []

        for i, weak_signal in enumerate(weak_signals):
            active_signal = weak_signal >= 0
            error_amatrix[i] = (1 - 2 * weak_signal) * active_signal

            # error denom to check abstain signals
            error_denom = np.sum(active_signal, axis=0) + 1e-8
            error_amatrix[i] /= error_denom

            # constants for error constraints
            constant = (weak_signal*active_signal) / error_denom
            constants.append(constant)

        # set up error upper bounds constraints
        constants = np.sum(constants, axis=1)
        assert len(constants.shape) == len(weak_signals_error_bounds.shape)
        bounds = weak_signals_error_bounds - constants
        error_constraint = self.build_constraints(error_amatrix, bounds)

        return error_constraint


    def precision_constraint(self, weak_signals, precision_bounds):
        """
        Calculates precision constraints for the weak signals

        Parameters
        ----------
        :param weak signals: weak signal of predicted labels
        :type true_labels: ndarray
        :param precision_bounds: bounds for the weak signals
        :type bounds: ndarray

        Returns
        -------
        :return: a matrix and bounds for precision constraints
        :rtype: dict
        """
        weak_signals = convert_to_ovr_signals(weak_signals)
        m, n, k = weak_signals.shape
        if precision_bounds is None:
            precision_bounds = np.ones(m,k) * self.precision_threshold

        constraint_set = dict()
        precision_amatrix = np.zeros((m, n, k))

        for i, weak_signal in enumerate(weak_signals):
            active_signal = weak_signal >= 0
            precision = precision * active_signal
            precision_amatrix[i] = precision / (np.sum(precision, axis=0)+ 1e-8)

        # set up precision upper bounds constraints
        precision_constraint = self.build_constraints(precision_amatrix, precision_bounds)

        return precision_constraint


    def linear_constraint(self, weak_signals, weak_signals_error_bounds=None, precision_bounds=None):
        """
        Calculates linear constraints for the weak signals

        Parameters
        ----------
        :param weak signals: weak signal of predicted labels
        :type true_labels: ndarray
        :param weak_signals_error_bounds: bounds for the weak signals' error
        :type bounds: ndarray
        :param precision_bounds: bounds for the weak signals
        :type bounds: ndarray

        Returns
        -------
        :return: a dictionary of error and precision constraints
        :rtype: dict
        """

        constraint_set = dict()
        error_constraints = self.error_constraint(weak_signals, weak_signals_error_bounds)
        precision_constraints = self.precision_constraint(weak_signals, weak_signals_error_bounds)
        constraint_set['error'] = error_constraints
        constraint_set['precision'] = precision_constraints
        return constraint_set


    def calculate_true_bounds(self, true_labels, predicted_labels, mask=None):
        """
        Computes bounds for the predicted labels

        Parameters
        ----------
        :param true_labels: true labels of the data
        :type true_labels: ndarray
        :param predicted_labels: label predictions
        :type bounds: ndarray

        Returns
        -------
        :return: error and precision bounds
        :rtype: 1D vector, 1D vector
        """

        if mask is None:
            mask = np.ones(true_labels.shape)

        if len(true_labels.shape) == 1:
            predicted_labels = predicted_labels.ravel()

        assert predicted_labels.shape == true_labels.shape, "The true labels and predicted labels must have the same shape"

        error_rate = true_labels*(1-predicted_labels) + predicted_labels*(1-true_labels)
        with np.errstate(divide='ignore', invalid='ignore'):
            error_rate = np.sum(error_rate*mask, axis=0) / np.sum(mask, axis=0)
            error_rate = np.nan_to_num(error_rate)

        precision = true_labels * predicted_labels
        precision = np.sum(precision*mask, axis=0) / (np.sum(predicted_labels*mask, axis=0)+ 1e-8)

        # check results are scalars
        if np.isscalar(error_rate):
            error_rate = np.asarray([error_rate])
            precision = np.asarray([precision])
        return error_rate, precision


    def get_true_weak_signal_bounds(self, true_labels, weak_signals):

        """
            Computes error and precision bounds for the weak signals

            Parameters
            ----------
            :param true_labels: true labels of the data
            :type true_labels: ndarray
            :param weak_signals: weak signals
            :type bounds: ndarray

            Returns
            -------
            :return: error and precision bounds
            :rtype: list, list
        """

        error_rates = []
        precisions = []
        weak_signals = convert_to_ovr_signals(weak_signals)
        mask = weak_signals >= 0

        for i, weak_label in enumerate(weak_signals):
            active_mask = mask[i]
            error_rate, precision = self.calculate_true_bounds(true_labels, weak_label, active_mask)
            error_rates.append(error_rate)
            precisions.append(precision)

        return error_rates, precisions
    
