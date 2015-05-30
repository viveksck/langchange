from vecanalysis.nn.base import NNBase
from vecanalysis.nn.misc import random_weight_matrix
import numpy as np
import numexpr as ne

class SimpleDenseEncodeDecode(NNBase):

    def __init__(self, dims=[300, 300, 300],
                 reg=0.0, alpha=0.1,
                 rseed=10):
        """
        Set up classifier: parameters, hyperparameters
        """
        ##
        # Store hyperparameters
        self.lreg = reg # regularization
        # To initialize, give shapes as if to np.array((m,n))
        param_dims = dict(Win = (dims[1], dims[0]), # 5x100 matrix
                          bin = dims[1], 
                          U=(dims[2], dims[1]), bu=(dims[2])) # column vector
        NNBase.__init__(self, param_dims, {})

        self.params.Win = random_weight_matrix(*self.params.Win.shape)
        self.params.U = random_weight_matrix(*self.params.U.shape)

    def _acc_grads(self, x, y):
        """
        Accumulate gradients from a training example.
        """
        ##
        # Forward propagation
        ##
        hout = np.tanh(self.params.Win.dot(x) + self.params.bin)
        pred_y = self.params.U.dot(hout) + self.params.bu

        ##
        # Backwards propogation
        ##
        delta = (pred_y-y)
        self.grads.U += np.outer(delta, hout)
        self.grads.bu += delta
        delta  = self.params.U.T.dot(delta) * (1 - hout**2)
        self.grads.Win += np.outer(delta, x) + self.params.Win * self.lreg
        self.grads.bin += delta

    def _compute_loss(self, x, y):
        """
        Compute the cost function for a single example.
        """
        pred_y = self.predict(x)
        J = ((pred_y - y) ** 2.0).sum() / 2
        J += (self.lreg / 2.0) * ((self.params.Win**2.0).sum() + (self.params.U**2).sum())
        return J 

    def compute_loss(self, X, Y):
        if len(X.shape) == 1:
            return self._compute_loss(X, Y)
        pred_ys = self.predict(X)
        J = ((pred_ys - Y) ** 2.0).sum() / 2
        J += (self.lreg / 2.0) * ((self.params.Win**2.0).sum() + (self.params.U**2).sum())
        return J

    def _one_predict(self, x):
        """Predict output vector."""
        h = np.tanh(self.params.Win.dot(x) + self.params.bin)
        pred_y = self.params.U.dot(h) + self.bu
        return pred_y

    def predict(self, X):
        if len(X.shape) == 1:
            return self._one_predict(X)
        else:
            hs = self.params.Win.dot(X.T).T + self.params.bin
            hs = ne.evaluate("tanh(hs)")
            return self.params.U.dot(hs.T).T + self.params.bu
