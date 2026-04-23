import numpy as np
import zanglib as zl
from timeit import default_timer as timer
from matplotlib import pyplot as plt


DIM = 5

A = np.array([[1.0, 2.0, 3.0], [0.0, 1.0, 4.0], [5.0, 6.0, 0.0]], dtype=np.float64)
print(f"A:\n{A}\n")
zl.gaussjordan(A)
print(f"A inversa:\n{A}\n")
