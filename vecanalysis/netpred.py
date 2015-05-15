from nn.base import NNBase
from misc import random_weight_matrix
import numpy as np

class DenseEncodeDecode(NNBase):

    def __init__(self, dims=[300, 300, 300],
                 reg=0.1, alpha=0.001,
                 rseed=10):
        """
        Set up classifier: parameters, hyperparameters
        """
        ##
        # Store hyperparameters
        self.lreg = reg # regularization
        self.alpha = alpha # default learning rate

        ##
        # NNBase stores parameters in a special format
        # for efficiency reasons, and to allow the code
        # to automatically implement gradient checks
        # and training algorithms, independent of the
        # specific model architecture
        # To initialize, give shapes as if to np.array((m,n))
        param_dims = dict(Win = (dims[1], dims[0]), # 5x100 matrix
                          b = (dims[1]), Wout = (dims[1], dims[2]), bout=dims[2]) # column vector
        NNBase.__init__(self, param_dims, None)

        self.params.Win = random_weight_matrix(*self.params.Win.shape)
        self.params.Wout = random_weight_matrix(*self.params.Wout.shape)

    def _acc_grads(self, x, y):
        """
        Accumulate gradients from a training example.
        """
        ##
        # Forward propagation
        ##
        h = np.tanh(self.Win.dot(x) + self.bin)
        pred_y = np.tanh(self.Wout(h) + self.bout)

        ##
        # Backwards propogation
        ##
        delta = np.abs(x-y) * (1 - pred_y**2)
        self.grads.Wout = np.outer(delta, pred_y)
        self.grads.bout = delta
        delta2 = np.Wout.T.dot(delta) * (1 - h ** 2)
        self.grads.Win = np.outer(delta2, h) 
        self.grads.bin = delta2

    def compute_loss(self, x, y):
        """
        Compute the cost function for a single example.
        """
        pred_y = self.predict(x)
        J = np.norm(pred_y - y) / 2
        Jreg = (self.lreg / 2.0) * (sum(self.params.Win**2.0) + sum(self.params.Wout**2))
        return J + Jreg

    def predict(self, x):
        """Predict output vector."""
        h = np.tanh(self.Win.dot(x) + self.bin)
        pred_y = np.tanh(self.Wout(h) + self.bout)
        return pred_y
