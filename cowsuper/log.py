""" 
file for logging calculations
    Currently focusing on Tensorboard logs, with add for normal logs
Borrowing code from Jeasine Ma
"""

import tensorflow as tf
# is it possible to do from tensorflow import Summary, summary
#from tensorflow.compat.v1 import summary
#from tensorflow.compat.v1 import Summary


class Logger(object):
    """Logging in tensorboard without tensorflow ops."""

    def __init__(self, log_dir):
        """Creates a summary writer logging to log_dir."""
        self.writer = tf.summary.create_file_writer(log_dir)

    def log_scalar(self, tag, value, step):
        """Log a scalar variable.
        Parameter
        ----------
        tag : basestring
            Name of the scalar
        value
        step : int
            training iteration
        """
        tf.summary.scalar(tag, value, step=step)

        self.writer.flush()



