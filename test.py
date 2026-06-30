import numpy as np
import zanglib as zl
from timeit import default_timer as timer
from matplotlib import pyplot as plt

DIM = 6
np.set_printoptions(precision=8, suppress=False)

def soldiag_test():
    A = np.diag(np.random.rand(DIM))
    x = np.ones(DIM)
    b = np.matmul(A, x)

    x_star = zl.soldiag(A, b)
    print(f"x = \n {x} \n")
    print(f"x* = \n {x} \n")

    return

soldiag_test()