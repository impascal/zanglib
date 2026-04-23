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
    if not np.diag(R).all():  # R ha almeno un elemento diagonale nullo
        raise ValueError("elementi nulli sulla diagonale: matrice R non invertibile")
    if R.dtype != np.float64:
        R = np.float64(R)  # per avere massima accuratezza nei calcoli

    R[n - 1, n - 1] = 1.0 / R[n - 1, n - 1]
    for i in range(n - 2, -1, -1):
        R[i, i] = 1.0 / R[i, i]
        for j in range(n - 1, i, -1):
            R[i, j] = -np.dot(R[i, i + 1 : j + 1], R[i + 1 : j + 1, j]) * R[i, i]
    return R


def utrisol(R, b):
    """
    Solve linear system Rx = b using backward subtitution
    R is maintained and solution is put in b
    PARAMETERS:
        R: Numpy upper triangular matrix
        b: Numpy 1D array
    """

    n, m = R.shape
    if n != m:
        raise ValueError("R must be a quadrat matrix..")

    if b.shape[0] != n:
        raise ValueError("b must have the same number of rows as R...")

    eps = np.finfo(np.float64).eps * norm(R, np.inf)
    if any(np.abs(np.diag(R)) < eps):
        raise ValueError("Some value of R are numerically too small...")

    for i in range(R.shape[0] - 1, -1, -1):
        b[i] /= R[i, i]
        b[0:i] -= R[0:i, i] * b[i]


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


def gaussDiag(A: np.ndarray) -> np.float64:
    """
    In-place LR factorization of A using Gauss elimination algorithm
    with diagonal strategy.

    Returns det(A), if 0 then A is singular.
    """
    n, m = A.shape

    np.float64(A)
    tol = np.finfo(np.float64).eps * norm(A, np.inf)

    for i in range(0, min(n, m) - 1):
        # check for too small pivot
        if np.abs(A[i, i]) < tol:
            raise ValueError(f"Pivot {i} under tolerance...")

        # in-place multiplicator vector
        A[i + 1 :, i] /= A[i, i]
        A[(i + 1) :, (i + 1) :] -= np.outer(A[(i + 1) :, i], A[i, (i + 1) :])

    nn = min(n, m)
    L = np.tril(A[:, :nn], -1)
    L[range(nn), range(nn)] = 1.0

    return L, np.triu(A)


def gaussPivPar(A: np.ndarray):
    """
    LR factorization of A using Gauss with partial pivoting on rows

    Returns L, R, p, det(A) if A is non-singular, raise value error otherwise
    """

    np.float64(A)  # make sure we are using double precisoin
    m, n = A.shape
    tol = np.finfo(np.float64).eps * norm(A, np.inf)  # tau = e_mach * norm_inf(A)

    # permutation vector
    p = np.array(range(m), dtype=np.int64)

    for k in range(min(m - 1, n)):
        piv_i = np.fabs(A[k:, k]).argmax() + k
        if piv_i != k:  # found a better pivot
            if np.abs(A[piv_i, k]) < tol:  # it is too small
                raise ValueError("Pivot too small...")

            # swap rows in active matrix
            A[[k, piv_i], :] = A[[piv_i, k], :]
            # swap indexes in permutation vector
            p[[k, piv_i]] = p[[piv_i, k]]
            # in place mult vector
            A[k + 1 :, k] /= A[k, k]
            # trasformation on active matrix
            A[k + 1 :, k + 1 :] -= np.outer(A[k + 1 :, k], A[k, k + 1 :])

    nn = min(n, m)
    if np.fabs(A[nn - 1, nn - 1]) < tol:  # non singular matrix
        print(f"WARNING: last pivot under tolerance (< {tol})...")

    L = np.tril(A[:, :nn], -1)
    L[range(nn), range(nn)] = 1.0
    return L, np.triu(A[:nn, :]), p


import numpy as np


def chol_in_place(A):
    """
    chol_in_place - Fattorizzazione "in place" di Cholesky di A (sovrascrive A)
    Calcola il fattore di Cholesy L di A, ossia la matrice triangolare inferiore non singolare L ad
    elementi diagonali positivi tale che A = L @ L.T
    ATTENZIONE: sovrascrive la parte strettamente sottodiagonale di A con l’omologa parte di L.
    SYNOPSIS: p, detA = chol_in_place(A)
    INPUT: A (float array) - Matrice simmetrica definita positiva
    OUTPUT: p (float array) - Diagonale del fattore di Cholesky L di A: p = np.diag( L )

    detA (float) - Determinante di A

    """
    [m, n] = A.shape
    p = np.zeros(n, dtype=np.float64)
    detA = 1.0
    if m != n:
        raise ValueError("matrice dei coefficienti non quadrata")
    for j in range(n):
        for i in range(j, n):
            s = A[j, i] - np.dot(A[i, 0:j], A[j, 0:j])
            if i == j:  # elemento della diagonale principale di L
                if s <= 0:
                    raise ValueError("matrice non definita positiva")
                else:
                    detA *= s

                p[j] = np.sqrt(s)
            else:  # elemento strettamente sottodiagonale di L
                A[i, j] = s / p[j]

    return p, detA


def givensrot(x1, x2):
    """
    givensrot - Rotazione elementare di Givens
    Si determinano c ed s tali da annullare l’elemento y2
    SINOPSYS: c, s = givensrot(x1, x2)
    """
    tol = np.finfo(np.float64).eps * max(abs(x1), abs(x2))
    if abs(x2) < tol:  # se abs(x2) e’ gia’ sotto soglia, non si esegue la rotazione
        c = 1.0
        s = 0.0
        return c, s

    # si utilizzano le formule numericamente piu’ stabili
    if abs(x2) >= abs(x1):
        t = np.float64(x1) / x2
        s = np.sign(x2) / np.sqrt(1 + t**2)
        c = s * t
    else:
        t = np.float64(x2) / x1
        c = np.sign(x1) / np.sqrt(1 + t**2)
        s = c * t

    return c, s


def qrfact(A):
    """
    qrfact - Fattorizzazione QR con rotazioni di Givens (non sovrascrive A)
    Implementazione applicabile anche al caso di matrice A non quadrata.
    SINOPSYS: Q, R = qrfact( A )
    OUTPUT: Q matrice ortogonale, R matrice triangolare (o trapezoidale) superiore
    """
    m, n = A.shape
    tol = np.finfo(np.float64).eps * norm(A, np.inf)
    r = min(m - 1, n)
    Q = np.eye(m)
    for i in range(r):
        for j in range(i + 1, m):
            if abs(A[j, i]) > tol:
                c, s = givensrot(A[i, i], A[j, i])
                # trasformazione di Givens sulle righe i-esima e j-esima
                Gij = np.array([[c, s], [-s, c]], dtype=np.float64)
                A[[i, j], i:n] = Gij @ A[[i, j], i:n]
                Q[:, [i, j]] @= Gij.T

    # Si possono rendere non negativi gli elementi diagonali di R:
    for i in range(min(m, n)):
        if A[i, i] < 0:
            A[i, i:n] = -A[i, i:n]
            Q[:, i] = -Q[:, i]

    return Q, np.triu(A)


def gaussjordan( A ):
# gaussjordan - Algoritmo di Gauss-Jordan per il calcolo dell’inversa (non sovrascrive A)
# SINOPSYS: Ainv = gaussjordan( A )
# OUTPUT: Ainv, matrice inversa di A (se e’ calcolabile)
    m, n = A.shape; tol = np.finfo( np.float64 ).eps * norm(A, np.inf)
    if ( m != n ): 
        raise ValueError("matrice non quadrata")
        # si effettua una copia per non sovrascrivere A e si assembla direttamente [ A, I ]
    A = np.c_[ np.float64( A.copy() ), np.eye( m ) ]
    for k in range( m ):
    # indice del PRIMO elemento sottodiagonale di modulo massimo nella k-esima colonna
        i = abs( A[k:, k] ).argmax() + k
        if ( i != k ): # scambio delle righe k-esima e i-esima
            A[ [k, i], : ] = A[ [i, k], : ]
            # calcolo e memorizzazione dei moltiplicatori
        if ( abs( A[k, k] ) > tol ):
            ind = np.r_[ np.arange(k), np.arange(k+1, m) ]
            A[ind, k] /= A[k, k]
            # operazione di base di livello 2: aggiornamento mediante diade
            A[ ind, (k+1): ] -= np.outer( A[ ind, k ], A[ k, (k+1): ] )
        else: # se il pivot e’ troppo piccolo, la matrice e’ NON invertibile numericamente
            raise ValueError(f"elemento pivot di modulo troppo piccolo (< tol = {tol})")
    
    return np.array( [ A[k, range(n, 2*n)] / A[k, k] for k in range( m ) ] )
