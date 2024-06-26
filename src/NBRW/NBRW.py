"""NBRW.py: A Python package for Non-Backtracking Random Walks on networks.
Matthew Shumway, 2024.

Work in conjunction with Adam Knudson, Dr. Mark Kempton, and Dr. Jane Breen.

This package is a Python implementation of the Non-Backtracking Random Walks (NBRW) on networks. It is designed
to be used with SageMath, a Python-based open-source mathematics software system. The package is designed to be used in
both research and applications of NBRW. It contains a NBRW class, which is designed to compute and store various attributes
associated to NBRW and Kemeny's constant.
"""

# Importing necessary packages
import numpy as np
from sage.all import *
from matplotlib import pyplot as plt

# Defining the NBRW class
class NBRW():
    """A class for Non-Backtracking Random Walks on networks. Accepts as input a SageMath graph object.
            
        Attributes:
        ----------------BASIC ATTRIBUTES OF GRAPH----------------
        G (Graph) :                                 SageMath graph object
        m (int) :                                   number of edges in G
        n (int) :                                   number of vertices in G
        A (np.ndarray) :                            adjacency matrix of G
        edges_list (list) :                         list of edges in G
        S (np.ndarray) :                            endpoint incidence operator
        T (np.ndarray) :                            starting point incidence operator
        tau (np.ndarray) :                          edge reversal operator
        C (np.ndarray) :                            S @ T
        B (np.ndarray) :                            C - tau
        ---------------DEGREE MATRICES-------------------------
        D (np.ndarray) :                            diagonal matrix of vertex degrees
        D_inv (np.ndarray) :                        inverse of D
        De (np.ndarray) :                           diagonal matrix of edge degrees
        De_inv (np.ndarray) :                       inverse of De
        -----------------STATIONARY DISTRIBUTIONS----------------
        pi (np.ndarray) :                           stationary distribution in vertex space
        pi_e (np.ndarray) :                         stationary distribution in edge space
        Wnb (np.ndarray) :                          1/(2m) J matrix in edge space - each row is NB edge space stationary distribution
        Wv (np.ndarray) :                           matrix whose rows are SRW vertex space stationary distribution
        We (np.ndarray) :                           matrix whose rows are SRW edge space stationary distribution
        -----------------TRANSITION MATRICES---------------------
        Pnb (np.ndarray) :                          non-backtracking transition matrix
        P (np.ndarray) :                            SRW transition matrix in vertex space
        Pe (np.ndarray) :                           SRW transition matrix in edge space 
        -----------------FUNDAMENTAL MATRICES---------------------
        Znb (np.ndarray) :                          NBW analogue to the fundamental matrix in SRW
        Znb_e (np.ndarray) :                        NBW analogue to the fundamental matrix in SRW in edge space
        Z (np.ndarray) :                            fundamental matrix in vertex space
        Z_e (np.ndarray) :                          fundamental matrix in edge space
        -----------------HITTING AND RETURN TIME MATRICES---------------------
        M (np.ndarray) :                            helper matrix used to compute Mnb -- from Dario paper
        Mev (np.ndarray) :                          matrix of hitting times from edge to vertex
        Mnb (np.ndarray) :                          matrix of hitting times in vertex space
        Mnb_e (np.ndarray) :                        matrix of hitting times in edge space
        Mv (np.ndarray) :                           matrix of mean first passage times in vertex space
        M_e (np.ndarray) :                          matrix of mean first passage times in edge space
        R_e (np.ndarray) :                          diagonal mean return time matrix in edge space
        R (np.ndarray) :                            diagonal mean return time matrix in vertex space
        -----------------KEMENY'S CONSTANTS---------------------
        Kv (float) :                                Kemeny's constant in vertex space
        Ke (float) :                                Kemeny's constant in edge space
        Knb_e (float) :                             NB Kemeny's constant in edge space
        Knb_v_trace (float) :                       NB Kemeny's constant in vertex space using trace of Znb
        Knb_v_mfpt (float) :                        NB Kemeny's constant in vertex space using pi @ Mnb @ pi
        Knb_v_sub (float) :                         NB Kemeny's constant in vertex space using edge space Kemeny's constant"""

    def __init__(self, G, pinwheel: bool = False) -> None:
        """Initializes the NBRW class with a Sage"Math graph object. Stores all relevant attributes of the NBRW."""

        self.G = G
        self.m, self.n = len(G.edges()), len(G.vertices())
        self.A = np.array(G.adjacency_matrix())
        self.edges_list = list(G.edges())
        self.S = self.S_matrix()
        self.T = self.T_matrix()
        self.tau = self.tau_matrix()
        self.C = self.S @ self.T
        self.B = self.C - self.tau

        self.D = np.diag(G.degree())
        self.D_inv = np.diag([1 / d for d in sorted(self.G.degree(), reverse=True)]) if pinwheel else np.diag([1 / d for d in self.G.degree()])
        self.De = self.De_matrix()
        self.De_inv = np.diag(1 / np.diag(self.De))

        self.pi = np.array([d/(2*self.m) for d in sorted(self.G.degree(), reverse=True)]) if pinwheel else np.array([d/(2*self.m) for d in self.G.degree()])
        self.pi_e = np.ones(2*self.m) / (2*self.m)
        self.Wnb = np.ones((2 * self.m, 2 * self.m)) / (2 * self.m)
        self.Wv = np.outer(np.ones(self.n), self.pi)
        self.We = np.outer(np.ones(2*self.m), self.pi_e)
 
        self.Pnb = self.nb_trans_matrix()
        self.P = self.D_inv @ self.A
        self.Pe = self.De_inv @ self.C

        self.Znb = self.znb_matrix()
        self.Znb_e = self.fund_matrix(P=self.Pnb, W=self.Wnb)
        self.Z = self.fund_matrix(P=self.P, W=self.Wv)
        self.Z_e = self.fund_matrix(P=self.Pe, W=self.We)
        
        self.M = self.M_matrix()
        self.Mev = self.M_ev_matrix()
        self.Mnb = self.Mnb_matrix()
        self.Mnb_e = self.Mnb_e_matrix()
        self.Mv = self.mfpt_matrix(size=self.n, Z=self.Z, W=self.Wv)
        self.M_e = self.mfpt_matrix(size=2*self.m, Z=self.Z_e, W=self.We)

        self.R_e = 2*self.m * np.eye(2*self.m)
        self.R = self.mrt_nbrw_matrix()

        self.Kv = np.trace(self.Z) - 1          # Kemeny's constant in vertex space. Always agrees with mfpt definition.
        self.Ke = self.Kv + 2*self.m - self.n   # Kemeny's constant in edge space. Always agrees with mfpt definition.
        self.Knb_e = np.trace(self.Znb_e) - 1   # NB Kemeny's constant in edge space. Always agrees with mfpt definition.
        self.Knb_v_trace = np.trace(self.Znb) - 1
        self.Knb_v_mfpt = self.pi @ self.Mnb @ self.pi
        self.Knb_v_sub = self.Knb_e - 2*self.m + self.n

    # Display methods
    # ==============================================================================================================================

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
            S[j + self.m, u] = 1    # j+m is the unique index of (v,u). So S[j+m, u] = S((v,u), u) := 1 -- assuming G undirected    

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
        Computational Complexity - O(m).
        Spatial Complexity - O(m^2).
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
    
    def nb_trans_matrix(self) -> np.ndarray:
        """Computes the non-backtracking transition matrix, Pnb, of the graph G. Pnb is a (2m x 2m) matrix.
        Computational Complexity - dependent on numpy implementation.
        Spatial Complexity - O(m^2)."""
        return np.linalg.inv(self.De - np.eye(2*self.m)) @ self.B
    
    def znb_matrix(self) -> np.ndarray:
        """Computes what we call the Znb matrix, which is a (n x n) matrix. This is the NBW analogue to the 
        fundamental matrix in the SRW case. 
        Computational Complexity - O(nm^2), or larger as determined by numpy implementation.
        Spatial Complexity - O(n^2)."""
        return np.eye(self.n) + self.D_inv @ self.T @ np.linalg.inv(np.eye(2*self.m) - self.Pnb + self.Wnb) @ self.S  - self.Wv
    
    def M_matrix(self) -> np.ndarray:
        """Found in (4.4) of the Dario hitting times paper. M is a (n x n) matrix. Startpoint incident operator T divided 
        by row sums of T. Used to calculate Mnb_v.
        Computational Complexity - O(nm).
        Spatial Complexity - O(nm)."""
        return self.T / self.T.sum(axis=1, keepdims=1)
    
    def M_ev_matrix(self) -> np.ndarray:
        """Computes the matrix M_{ev}, which is a (2m x n) matrix. This is the matrix of hitting times from edge e
        to vertex v.
        Computational Complexity - O(nm).
        Spatial Complexity - O(mn)."""
        Mev = np.zeros((2*self.m, self.n))
        for i in range(self.n):
            M_col_i = self.nb_hitting_times(i)
            Mev[:, i] = M_col_i
        return Mev
    
    def Mnb_matrix(self) -> np.ndarray:
        """Computes the matrix Mv_nb, which is a (n x n) matrix. This is the matrix of hitting times in the vertex space.
        Computational Complexity - O(nm).
        Spatial Complexity - O(n^2)."""
        Mnb_v = np.zeros((self.n, self.n))
        for i in range(self.n):
            M_col_i = self.M @ self.nb_hitting_times(i)
            Mnb_v[:, i] = M_col_i
        return Mnb_v
    
    def Mnb_e_matrix(self) -> np.ndarray:
        """Computes the matrix Mnb_e, which is a (2m x 2m) matrix. This is the matrix of hitting times in the edge space.
        Computational Complexity - O(nm).
        Spatial Complexity - O(m^2)."""

        M = np.zeros((2*self.m, 2*self.m))
        for i in range(2*self.m):
            N_i = self.nb_fund_mat(i)
            row_sums = np.sum(N_i, axis=1)
            if i > 0:
                M[:i, i] = row_sums[:i]  # Elements before the current row
            M[i+1:, i] = row_sums[i:]  # Elements after the current row

        return M
    
    def mfpt_matrix(self, size, Z, W) -> np.ndarray:
        """
        Computes the square matrix of mean first passage times in the SRW. Can handle both vertex and edge spaces.
        This procedure is adapted from well known results, but is made explicit in
        Hunter, Jeffrey J. "The computation of the mean first passage times for Markov chains." Linear Algebra and its Applications 549 (2018): 100-122.
        Computational Complexity - dependent on numpy functions.
        Spatial Complexity - either O(m^3) or O(n^3).
        """
        I, J = np.eye(size), np.ones((size, size))
        M = (I - Z + J @ np.diag(np.diag(Z))) @ np.linalg.inv(np.diag(np.diag(W)))
        return M - np.diag(np.diag(M))
    
    def fund_matrix(self, P, W) -> np.ndarray:
        """Computes the fundamental matrix. Works for either the edge space or the vertex space.
        Computational Complexity - depends on np.linalg.inv.
        Spatial Complexity - O(m^2) or O(n^2)."""
        size = P.shape[0]
        return np.linalg.inv(np.eye(size) - P + W)
    
    def mrt_nbrw_matrix(self) -> np.ndarray:
        """Computes the diagonal mean return time matrix, which is size (n x n). Should agree with the SRW.
        Equations taken from (4.8) in the Dario paper.
        Computational Complexity - O(nm^2).
        Spatial Complexity - O(n^2)."""
        return np.eye(self.n) + np.diag(np.diag(self.D_inv @ self.T @ self.Pnb @ self.Mev))
    

    # Helper methods
    # ==============================================================================================================================
    def nb_hitting_times(self, j: int) -> np.ndarray:
        """
        Compute the NB hitting times from edge e to vertex j. This implementation is based on Theorem 4.3 in the 
        Dario paper.
        Resulting vector is a column of M_{ev}.
        Computational Complexity - depends on the np.linalg.solve() function.
        Spatial Complexity - O(m).
        """
        Gedges = [(g[0], g[1]) for g in self.G.edges()]
        Gedges.extend([(g[1], g[0]) for g in self.G.edges()])
        
        entries_to_delete = []

        # Find the rows/columns to delete
        entries_to_delete = [i for i in range(len(Gedges)) if Gedges[i][0] == j]

        # Mask instead of actually deleting rows and columns
        mask = np.ones((2 * self.m), dtype=bool)
        mask[entries_to_delete] = False

        P_new = self.Pnb[mask, :][:, mask]

        # Compute the tau vector -- coming from paper
        tau_vec = np.linalg.solve(np.eye(len(P_new)) - P_new, np.ones(len(P_new)))

        # Construct the full sized tau vector
        full_tau = np.zeros(2 * self.m)
        full_tau[mask] = tau_vec

        return full_tau

    
    def nb_fund_mat(self, i: int) -> np.ndarray:
        """Compute (I-Q)^-1 where Q = Pnb_{i,i} is Pnb with deleted cols/rows i.
        Computational Complexity - depends on np.linalg.inv.
        Spatial Complexity - O(m^2)."""
        Pnb = self.Pnb.copy()

        # use mask instead of actually deleting rows/cols
        mask = np.ones(Pnb.shape[0], dtype=bool)
        mask[i] = False

        Pnb_sub = Pnb[mask][:, mask]  # submatrix with deleted row/col i

        return np.linalg.inv(np.identity(len(Pnb_sub)) - Pnb_sub)
    
    # Methods that aren't deemed necessary (yet)
    # ==============================================================================================================================
    def alpha_matrix(self) -> np.ndarray:
        """Computes the Diag(Alpha) matrix, which is a (n x n) matrix. This is the matrix from the Dario paper."""
        return (np.eye(2*self.m) + np.diag(np.diag(self.Pnb @ self.Mev @ self.T))) / (2*self.m)
    
    def beta_vector(self, mat) -> np.ndarray:
        """Computes the Beta vector, which is a (n x 1) vector. This is the vector from the Dario paper."""
        return np.diag(self.D_inv @ self.T @ self.Mnb_e @ mat @ self.T.T)
    
    def italian_mfpt(self, alpha, beta):
        """Computes the NB vertex mean first passage times (Mv_nb) from methods in the Italian paper."""
        return self.D_inv @ self.T @ self.Mnb_e @ alpha @ self.T.T - np.outer(np.ones(self.n), beta)
