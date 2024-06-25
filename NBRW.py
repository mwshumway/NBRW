"""NBRW.py: A Python package for Non-Backtracking Random Walks on networks.
Matthew Shumway, 2024.

This package is a Python implementation of the Non-Backtracking Random Walks (NBRW) on networks. It is designed
to be used with SageMath, a Python-based open-source mathematics software system. The package is designed to be used in
both research and applications of NBRW. It contains a NBRW class, which is designed to compute and store various attributes
associated to NBRW and Kemeny's constant.
"""

# Importing necessary packages
import numpy as np
from SageMath import *
from matplotlib import pyplot as plt

# Defining the NBRW class
class NBRW_test():
    """A class for Non-Backtracking Random Walks on networks. Accepts as input a SageMath graph object"""

    def __init__(self, G: Graph) -> None:
        """Initializes the NBRW class with a Sage"Math graph object. Stores all relevant attributes of the NBRW."""

        self.G = G
        self.m, self.n = len(G.edges()), len(G.vertices())
        self.A = G.adjacency_matrix()
        self.edges_list = list(G.edges())

        self.S = self.S_matrix()
        self.T = self.T_matrix()
        self.tau = self.tau_matrix()

        self.C = self.S @ self.T
        self.B = self.C - self.tau
        self.D = np.diag(G.degree())
        self.D_inv = np.diag(1 / np.array(G.degree()))
        self.De = self.De_matrix()
        self.De_inv = np.diag(1 / np.diag(self.De))

        

    def show(self) -> None:
        """Displays the graph G"""
        self.G.show()

    # Attribute computation methods
    # ==============================================================================================================================

    def S_matrix(self) -> np.ndarray:
        """Computes the endpoint incidence operator, S, of the graph G. S is a (2m x n) matrix.
        Computational Complexity - O(m).
        Spatial Complexity - O(mn)."""
        S = np.zeros((2*self.m, self.n), dtype=int)  # initialize empty S matrix

        # Iterate through all edges of the graph, shorten loop to m instead of 2m
        for j in range(self.m):
            u, v, _ = self.edges_list[j]      # edges[j] of format (u, v, weight)
            S[j, v] = 1             # j is the unique index of (u,v). So S[j, v] = S((u,v), v) := 1
            S[j + self.m, u] = 1    # j+m is the unique index of (v,u). So S[j+m, u] = S((v,u), u) := 1 -- assuming G underirected    

        return S
    
    def T_matrix(self) -> np.ndarray:
        """Computes the starting point incidence operator, T, of the graph G. T is a (n x 2m) matrix.
        Computational Complexity - O(m).
        Spatial Complexity - O(mn)."""
        T = np.zeros((self.n, 2*self.m), dtype=int)  # initialize empty T matrix

        # Similar implementation to S_matrix, see self.S_matrix() for more details
        for j in range(self.m):
            u, v, _ = self.edges_list[j]
            T[u, j] = 1             # T[u, j] = T(u, (u,v)) := 1
            T[v, j + self.m] = 1
        
        return T
    
    def tau_matrix(self) -> np.ndarray:
        """Computes the edge reversal operator, tau. tau is a (2m x 2m) matrix.
        Computational Complexity - determined by numpy -- update later from documentation.
        Spatial Complexity - O(m^2)."""
        zero = np.zeros((self.m, self.m), dtype=int)
        I = np.eye(self.m, dtype=int)
        return np.block([[zero, I], [I, zero]])
    
    def De_matrix(self) -> np.ndarray:
        """Computes the diagonal matrix of edge degrees, De. De is a (2m x 2m) matrix.
        """
        De = np.zeros((2*self.m, 2*self.m))
        deg = self.G.degree()
        
        # Extracting the degree of the endpoints of each edge -- assuming G is undirected
        deg_array = np.array([deg[j] for _, j, _ in self.G.edges()])
        deg_array_shifted = np.array([deg[i] for i, _, _ in self.G.edges()])  # gets edge (j, i) if edge (i, j) exists
        
        # Fancy indexing to fill in De array
        De[np.arange(self.m), np.arange(self.m)] = deg_array
        De[np.arange(self.m, 2*self.m), np.arange(self.m, 2*self.m)] = deg_array_shifted
        
        return De
    
