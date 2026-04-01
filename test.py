import numpy as np
from zanglib import gaussDiag
from timeit import default_timer as timer
from matplotlib import pyplot as plt

def main():
    rnd = np.random.default_rng()
    
    n_max = 300
    X = np.empty(n_max)
    y = np.empty(n_max)

    for i in range(n_max):
        dim = i+1
        X[i] = dim

        # construction of test problem
        A = rnd.uniform(size=(dim, dim))
        x = np.ones(dim)
        b = np.matmul(A, x)

        start = timer()
        res = gaussDiag(A, b)
        stop = timer()
        
        y[i] = stop - start

    # plot data
    plt.title("Gaussian eliminination algorithm time series vs Matrix dimension")
    plt.plot(X, y, "b")
    plt.xlabel("Dimension (nxn)")
    plt.ylabel("Time (s)")
    plt.show()

if __name__ == "__main__":
    main()