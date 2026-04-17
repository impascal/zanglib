import numpy as np
from zanglib import gaussDiag
from timeit import default_timer as timer
from matplotlib import pyplot as plt

def main():
    DIM = 800

    A = np.random.randn(DIM, DIM)
    x = np.ones(DIM, dtype=np.float64)
    b = A @ x

    start = timer()
    det = gaussDiag(A, b)
    stop = timer()

    print(f"Solution expected:\n{x} \nSolution found:\n{b}\nTime needed: {stop - start:.3f} s")

if __name__ == "__main__":
    main()