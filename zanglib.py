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


def utrisol(R: np._ArrayFloat64_co, b: np._ArrayFloat64_co) -> np._ArrayFloat64_co:
    """
    Solve linear system Rx = b in which R is an upper triangular matrix
    and b is the vector of constant terms using backward substution algorithm.
    Both R and b are not preserved, b can be used both as return value
    and output parameter.
    """
    [n, m] = R.shape
    if (n != m) or len(R.shape) > 2:
        raise ValueError("R must be a n x n upper triangular matrix...")

    if len(b.shape) > 1:
        raise ValueError("b must be a one dimensional array...") 

    if len(b) != n:
        raise ValueError("b must be and R must have the same number of rows...")

    # Convert everything to float64 just to be sure
    np.float64(R)
    np.float64(b)

    eps = np.finfo(np.float64).eps * norm(R, np.inf)
    if any(np.diag(R) < eps):
        raise ValueError("Some value of R are numerically too little...")

    for i in range(n - 1, -1, -1):
        b[i] = b[i] / R[i, i]
        R[0:i, i] *= b[i]
        b[0:i] -= R[0:i, i]

    return b

def ltrisol(L: np._ArrayFloat64_co, b: np._ArrayFloat64_co) -> np._ArrayFloat64_co:
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

    eps = np.finfo(np.float64).eps * norm(L, np.inf)
    if any(np.diag(L) < eps):
        raise ValueError("Some value of R are numerically too little...")

    for i in range(n):
        b[i] = b[i] / L[i, i]
        L[i+1:n, i] *= b[i]
        b[i+1:n] -= L[i+1:n, i]

    return b