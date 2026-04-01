# Collection of "toy" functions for numerical methods

import importlib.util
import shutil
import sys

import numpy as np
from numpy.linalg import norm

def check_and_test():
    # 1. Python Version Check
    print("--- 1. System Version Check ---")
    v = sys.version_info
    print(f"Python version: {sys.version.split()[0]}")
    if v.major == 3 and v.minor >= 10:
        print("Status: Python version meets the requirement (>= 3.10).")
    else:
        print("Status: Python version is below 3.10.")
    print("Ref: https://devguide.python.org/versions/")
    print("-" * 50)

    # 2. Package and Functional Tests
    test_results = {}

    # --- NumPy & SciPy: Numerical Linear Algebra Test ---
    print("--- 2. Testing Numerical Stack (NumPy/SciPy) ---")
    if importlib.util.find_spec("numpy") and importlib.util.find_spec("scipy"):
        try:
            import scipy.linalg as la

            # Test: Solving a small system Ax = b
            A = np.array([[4, 3], [3, 2]], dtype=float)
            b = np.array([1, 1], dtype=float)
            x = la.solve(A, b)
            print(f"[SUCCESS] NumPy/SciPy: Linear system solved. x = {x}")
            test_results["Numerical"] = True
        except Exception as e:
            print(f"[FAILURE] NumPy/SciPy: Test failed with error: {e}")
            test_results["Numerical"] = False
    else:
        print("[MISSING] NumPy or SciPy not found.")

    # --- Matplotlib: Visualization Test ---
    print("\n--- 3. Testing Visualization (Matplotlib) ---")
    if importlib.util.find_spec("matplotlib"):
        try:
            import matplotlib.pyplot as plt

            # Test: Generate a figure to verify backend stability
            plt.figure(figsize=(4, 2))
            plt.plot([0, 1], [0, 1])
            plt.title("Backend Test")
            plt.close()  # Close to avoid hanging the script
            print("[SUCCESS] Matplotlib: Figure generated and backend initialized.")
            test_results["Graphics"] = True
        except Exception as e:
            print(f"[FAILURE] Matplotlib: Test failed with error: {e}")
            test_results["Graphics"] = False
    else:
        print("[MISSING] Matplotlib not found.")

    # --- IPython: Interactive Shell Check ---
    print("\n--- 4. Checking Interactive Shell (IPython) ---")
    ipy_path = shutil.which("ipython")
    if ipy_path:
        print(f"[SUCCESS] IPython found at: {ipy_path}")
        test_results["Interactive"] = True
    else:
        ipy_path = shutil.which("ipython3")
        if ipy_path:
            print(f"[SUCCESS] IPython found at: {ipy_path}")
            test_results["Interactive"] = True
        else:
            print("[MISSING] IPython executable not found in PATH.")
            test_results["Interactive"] = False

    # Summary
    print("\n" + "=" * 20)
    print("FINAL ENVIRONMENT SUMMARY")
    print("=" * 20)
    for test, status in test_results.items():
        print(f"{test:12}: {'PASS' if status else 'FAIL'}")

def invupper(R):
    """
    invupper - Sovrascrive una matrice triangolare superiore invertibile con la propria inversa
    SYNOPSIS: R = invupper(R)
    INPUT: R (float array) - Matrice triangolare superiore da invertire
    OUTPUT: R (float array) - La matrice in input sovrascritta con la propria inversa
    """
    # Aggiungere opportuni controlli sull’input
    [m, n] = R.shape
    if not np.diag(R).all(): # R ha almeno un elemento diagonale nullo
        raise ValueError("elementi nulli sulla diagonale: matrice R non invertibile")
    if R.dtype != np.float64: 
        R = np.float64( R ) # per avere massima accuratezza nei calcoli
        
    R[n-1, n-1] = 1.0 / R[n-1, n-1]
    for i in range( n-2, -1, -1 ):
        R[i, i] = 1.0 / R[i, i]
        for j in range( n-1, i, -1 ):
            R[i, j] = -np.dot( R[i, i+1 : j+1 ], R[ i+1 : j+1, j] ) * R[i, i]
    return R

def utrisol(R: np.ndarray, b: np.ndarray):
    """
    Solve linear system Rx = b using backward subtitution
    PARAMETERS:
        R: Numpy upper triangular matrix
        b: Numpy 1D array
    """

    if not isinstance(R, np.ndarray):
        raise TypeError("R must be a Numpy ndarray...")

    if not isinstance(b, np.ndarray):
        raise TypeError("b must be a NumPy array...") 

    if len(R.shape) > 2 or R.shape[0] != R.shape[1]:
        raise ValueError("R must be a n x n upper triangular matrix...")

    if len(b.shape) > 1:
        raise ValueError("b must be a one dimensional array...")

    if b.shape[0] != R.shape[0]:
        raise ValueError("b must be and R must have the same number of rows...")

    # Convert everything to float64 just to be sure
    np.float64(R)
    np.float64(b)

    eps = np.finfo(np.float64).eps
    if any(np.abs(np.diag(R)) < eps):
        raise ValueError("Some value of R are numerically too little...")

    for i in range(R.shape[0] - 1, -1, -1):
        b[i] = b[i] / R[i, i]
        R[0:i, i] *= b[i]
        b[0:i] -= R[0:i, i]


def ltrisol(L: np.ndarray, b: np.ndarray):
    """
    Solve linear system Lx = b in which R is a lower triangular matrix
    and b is the vector of constant terms using forward substution algorithm.
    Both R and b are not preserved, b can be used both as return value
    and output parameter.
    """

    [n, m] = L.shape
    if (n != m) or len(L.shape) > 2:
        raise ValueError("R must be a n x n upper triangular matrix...")

    if len(b.shape) > 1:
        raise ValueError("b must be a one dimensional array...")

    if len(b) != n:
        raise ValueError("b must be and R must have the same number of rows...")

    # Convert everything to float64 just to be sure
    np.float64(L)
    np.float64(b)

    eps = np.finfo(np.float64).eps
    if any(np.diag(L) < eps):
        raise ValueError("Some value of R are numerically too little...")

    for i in range(n):
        b[i] = b[i] / L[i, i]
        L[i + 1 : n, i] *= b[i]
        b[i + 1 : n] -= L[i + 1 : n, i]

def gaussDiag(A: np.ndarray, b: np.ndarray) -> np.float64:
    """
    In-place LR factorization of A using Gauss elimination algorithm
    with diagonal strategy. Solution (if A non-singular) will be placed in b

    Returns det(A), if 0 then A is singular and his pseudorank(A) != rank([A|b])
    """
    n = A.shape[0]
    
    np.float64(A)
    np.float64(b)

    tol = np.finfo(np.float64).eps * norm(A, np.inf)

    for i in range(n - 1):
        # check on pivot
        if np.abs(A[i,i]) < tol:
            return 0

        # multiplicator arrays
        m = A[i+1:, i] / A[i,i]

        # elementar gauss transformation of dimension n - i
        L = np.eye(n - i, dtype=np.float64)
        L[1:, 0] = - m

        # apply elementar transformartion on submatrix of A  
        np.matmul(L, A[i:, i:], out=A[i:, i:])
        # apply elementar transformation on subvector of b
        np.matmul(L, b[i:], out=b[i:])

        # nella colonna i-esima inserisco la rispettiva colonna della fattorizzazione
        # calcolata come l'inverso della trasformazione elementare
        A[i+1: , i] = m

    if np.abs(A[n-1,n-1]) < tol:
        return 0

    # solve Rx = y
    utrisol(np.triu(A), b)

    # compute determinant
    det = 1
    for i in range(n):
        det *= A[i,i]

    return det