from scipy import linalg as la
import numpy as np

def Drazin(A, tol=1e-4):
    """Accepts a numpy nd-array and computes the Drazin inverse
    
    Parameters:
        A ((n,n) ndarray): An nxn matrix.
        tol (float): A float close to zero to classify 0 eigenvalues

    Returns:
        AD ((n,n) ndarray): The nxn Drazin Inverse of A
    """
    n, n = A.shape
    # This is some algorithm to compute the Drazin Inverse I found in an archived ACME Lab
    T1, Q1, k1 = la.schur(A, sort=lambda x: abs(x) > tol)
    T2, Q2, k2 = la.schur(A, sort=lambda x: abs(x) <= tol)
    U = np.hstack((Q1[:, :k1], Q2[:, :n-k1]))
    U_inv = la.inv(U)
    V = U_inv @ A @ U
    Z = np.zeros((n, n))
    if k1 != 0:
        Z[:k1, :k1] = la.inv(V[:k1, :k1])
    return U @ Z @ U_inv


def verify_if_drazin(A, k, AD):
    """Verify that a matrix AD is the Drazin inverse of A
    
    Parameters:
        A ((n,n) ndarray): An nxn matrix.
        AD ((n,n) ndarray): An nxn candidate for Drazin inverse of A
        k (int): the index of A

    Returns:
        (bool) True of AD is the Drazin inverse of A, False otherwise
    """
    is_drazin = True
    # Test all cases that uniquely determine the drazin inverse of A
    if not np.allclose(A@AD, AD@A, rtol=1e-3):  
        return False, 1
    B = np.linalg.matrix_power(A, k)  # only compute this once, A^k
    if not np.allclose(B@A@AD, B, rtol=1e-3):
        return False, 2
    if not np.allclose(AD @ A @ AD, AD, rtol=1e-3):
        return False, 3
    
    return is_drazin  # otherwise everything passed, so return true

def index(A, tol=1e-5):
    """Compute the index of the matrix A 
    
    Parameters:
        A ((n,n) ndarray): An nxn matrix

    Returns:
        k (int): The index of A
    """

    # test for non-singularity
    if not np.allclose(la.det(A), 0):
        return 0

    n = len(A)
    k = 1
    Ak = A.copy()
    while k <= n:
        r1 = np.linalg.matrix_rank(Ak)
        r2 = np.linalg.matrix_rank(np.dot(A, Ak))
        if r1 == r2:  # equivalent to checking dimension of nullspaces
            return k
        Ak = np.dot(A, Ak)
        k += 1
    
    return k
