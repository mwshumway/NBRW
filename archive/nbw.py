import numpy as np
from sage.all import *

#######################################################################################
#######################################################################################
# Functions that were used in the edge space project.
#  Some may still be useful in this project

def S_mat_sage(G):
    """return matrix S such that ST = C (TS=A)"""
    edge_list = list(G.edges())
    m = len(edge_list)
    n = len(G.vertices())

    S = matrix.zero(2 * m, n)  # np.zeros((2*m,n))
    for x in range(n):
        # iterate through nodes
        for j in range(m):
            # iterate through edges
            if edge_list[j][1] == x:
                # if edge points to x
                S[j, x] = 1
            if edge_list[j][0] == x:
                # if edge comes from x
                # well there must be an edge leaving too
                S[j + m, x] = 1

    return S


def T_mat_sage(G):
    """return matrix T such that ST = C (TS=A)"""
    edge_list = list(G.edges())
    m = len(edge_list)
    n = len(G.vertices())

    T = matrix.zero(n, 2 * m)  # np.zeros((n,2*m))
    for x in range(n):
        # iterate through nodes
        for j in range(m):
            # iterate through edges
            if edge_list[j][0] == x:
                # if edge comes from x
                T[x, j] = 1
            if edge_list[j][1] == x:
                # must be edge from other side
                T[x, j + m] = 1

    return T


def tau_mat_sage(G):
    """return matrix tau such that ST-tau = B"""
    edge_list = list(G.edges())
    m = len(edge_list)
    zero_mat = matrix.zero(m)
    id_mat = identity_matrix(m)
    t = block_matrix([[zero_mat, id_mat], [id_mat, zero_mat]])
    return t


def B_mat_sage(G):
    """return matrix B = ST-tau from a sage graph"""
    edge_list = list(G.edges())
    m = len(edge_list)
    n = len(G.vertices())

    # get S mat
    S = matrix.zero(2 * m, n)  # np.zeros((2*m,n))
    for x in range(n):
        # iterate through nodes
        for j in range(m):
            # iterate through edges
            if edge_list[j][1] == x:
                # if edge points to x
                S[j, x] = 1
            if edge_list[j][0] == x:
                # if edge comes from x
                # well there must be an edge leaving too
                S[j + m, x] = 1

    T = matrix.zero(n, 2 * m)  # np.zeros((n,2*m))
    for x in range(n):
        # iterate through nodes
        for j in range(m):
            # iterate through edges
            if edge_list[j][0] == x:
                # if edge comes from x
                T[x, j] = 1
            if edge_list[j][1] == x:
                # must be edge from other side
                T[x, j + m] = 1

    zero_mat = matrix.zero(m)
    id_mat = identity_matrix(m)
    t = block_matrix([[zero_mat, id_mat], [id_mat, zero_mat]])

    return S * T - t


def nb_trans_mat_sage(G):
    """return NBRW edge space transition probability matrix"""
    b = B_mat_sage(G)
    the_row_sums = sum(b)
    row_sums = [i ^ -1 for i in the_row_sums]
    return b * diagonal_matrix(row_sums)  # *b #wasn't stochastic like this...


def Dmat_inv(G):
    """
    D^{-1}
    """
    return np.diag([1 / d for d in G.degree()])


def trans_mat_sage(G):
    """vertex space transition matrix of a simple random walk"""
    D = np.diag([i ^ -1 for i in G.degree()])
    return D @ G.adjacency_matrix()


#######################################################################################
#######################################################################################
# Functions to calculate the NBRW in the vertex space

def nb_hitting_times(G, j):
    """
    return list (np.array()) m_{i, j} where m is
    NB hitting time from i to j for all i, with j fixed.

    THIS IS ACTUALLY I THINK nb hitting time m_{e, j}. 2m vector of hitting times from
    the edge e to the vertex j. (SEE Theorem 4.3 I believe.)

    i.e. the columns of M_{ev}
    """
    m = len(G.edges())
    n = len(G)
    P = nb_trans_mat_sage(G)
    Gedges = [(g[0], g[1]) for g in G.edges()]
    Gedges.extend([(g[1], g[0]) for g in G.edges()])
    entries_to_delete = []

    # Find the rows/columns to delete
    for i in range(len(Gedges)):
        if Gedges[i][0] == j:
            entries_to_delete.append(i)
    P_new = np.delete(P, entries_to_delete, axis=0)
    P_new = np.delete(P_new, entries_to_delete, axis=1)

    # Keep a list of what was NOT deleted, to construct the full vector later
    remaining_entries = [i for i in range(2 * m)]
    for k in entries_to_delete:
        remaining_entries.remove(k)

    # Solve (I - P)x = 1
    ones = np.array([1] * len(P_new))
    tau_vec = np.linalg.solve(np.identity(len(P_new)) - P_new, ones)

    # recreate full tau_vec, putting the 0's back in where necessary
    full_tau = np.array([0.0] * (2 * m))
    for i in range(len(remaining_entries)):
        full_tau[remaining_entries[i]] = tau_vec[i]

    return full_tau


def M_mat(G):
    """
    Matrix M as in (4.4) in the Hitting Times Paper
    (Basically the start point operator divided by the row sums)
    """
    M = T_mat_sage(G).numpy()
    # Divide by row sums
    return M / M.sum(axis=1, keepdims=1)


def nb_vertex_mfpt(G):
    """
    Get the matrix (np.array probably) where M_{i,j} is the mfpt i -> j.
    That is, $M_v^{nb}$

    For now, assumes vertices are {0, 1, ..., n-1}
    """
    n = len(G)
    MFPT = np.zeros((n, n))

    for i in range(n):
        MFPT_col_i = M_mat(G) @ nb_hitting_times(G, i)
        for j in range(n):
            MFPT[j][i] = MFPT_col_i[j]

    return MFPT


def nb_vertex_kemeny_mfpt(G):
    """
    Use these hitting times to get Kemeny's constant
    """
    volG = 2 * len(G.edges())
    steady = np.array([d / volG for d in G.degree()])
    return steady @ nb_vertex_mfpt(G) @ steady


def M_ev_NBRW(G):
    """
    return the matrix $M_{ev}$, that is, the hitting times (mean first passage times)
    of the nonbacktracking random walk
    """
    m = len(G.edges())
    n = len(G)
    Mev = np.zeros((2 * m, n))
    for i in range(n):
        Mvec = nb_hitting_times(G, i)
        for j in range(2 * m):
            Mev[j][i] = Mvec[j]

    return Mev


#######################################################################################
#######################################################################################
# Other useful NBRW stuff. Particularly helpful when working with Z_{nb}

def P_hat(G):
    """
    (D-I)^{-1}A.
    I should probably not take an inverse since its so easy but whatever. change as needed
    """
    return np.diag([1 / (d - 1) for d in G.degree()]) @ G.adjacency_matrix()


def P_mat(G):
    """
    D^{-1}A
    """
    return np.diag([1 / d for d in G.degree()]) @ G.adjacency_matrix()


def clean_matrix_display(M, tol=1e-10):
    """
    make zeros look like 0
    """
    m = len(M[0][:])
    n = len(M[:][0])
    for i in range(n):
        for j in range(m):
            if np.abs(M[i][j]) < tol:
                M[i][j] = 0
    return M


# October 17, 2023: Potential candidate for fundamental matrix of NBRW. The trace of which should be Kemeny + 1
def Znb(G):
    """
    A guess at what Znb should be. Seems to work pretty close to what we want numerically. Matches exactly with
    edge transitive graphs. Usually close with all other graphs we've tried.
    """
    n = len(G)
    m = len(G.edges())
    Dinv = np.diag([1 / d for d in G.degree()])
    T = T_mat_sage(G)
    S = S_mat_sage(G)
    Wnb = np.ones((2 * m, 2 * m)) / (2 * m)
    Wv = np.outer(np.ones(n), [d / (2 * m) for d in G.degree()])

    return np.eye(n) + Dinv @ T @ (np.linalg.inv(np.eye(2 * m) - nb_trans_mat_sage(G) + Wnb)) @ S - Wv


def mrt_NBRW_vertex(G):
    """
    Diagonal matrix of mean return times of a NBRW.
    See Equations below (4.8) in hitting times paper.

    Note: This should be the same Mean Return Times as SRW I believe which might just be 1/\pi_j
          if I remember correctly as I'm typing this note
    """
    Dinv = np.diag([1 / d for d in G.degree()])
    T = T_mat_sage(G)
    Pnb = nb_trans_mat_sage(G)
    Mev = M_ev_NBRW(G)
    return np.eye(len(G)) + np.diag(np.diag(Dinv @ T @ Pnb @ Mev))


def Pv_NBRW(G, k):
    """
    Return the vertex space transition probability matrix for the nonbacktracking random walk for the kth step
    """
    if k == 0:
        return np.eye(len(G))

    if k == 1:
        return trans_mat_sage(G)

    # If k > 1
    Dinv = np.diag([1 / d for d in G.degree()])
    T = T_mat_sage(G)
    S = S_mat_sage(G)
    Pnb = nb_trans_mat_sage(G)
    return Dinv @ T @ (np.linalg.matrix_power(Pnb, k - 1)) @ S


if __name__ == '__main__':
    G = sm.graphs.GemGraph()
    S_mat_sage(G)

