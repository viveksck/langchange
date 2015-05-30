##
# Miscellaneous helper functions
##

from numpy import *

def random_weight_matrix(m, n):
    #### YOUR CODE HERE ####
    eps = sqrt(6.0 / (m + n))
    A0 = random.rand(m , n) * 2 * eps - ones((m ,n)) * eps

    #### END YOUR CODE ####
    assert(A0.shape == (m,n))
    return A0