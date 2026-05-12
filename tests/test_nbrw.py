"""
Comprehensive tests for the NBRW package.

Run from the repo root with the SageMath Python interpreter:

    sage --python -m pytest tests/ -v

or, if pytest is on the SageMath PATH:

    sage -sh -c "pytest tests/ -v"

Mathematical conventions used throughout:
  n   = number of vertices
  m   = number of undirected edges  (so 2m directed edges)
  e   = directed edge index in [0, 2m)
  v   = vertex index in [0, n)
"""

import numpy as np
import pytest
from sage.all import graphs, Graph
from NBRW import NBRW
from NBRW.extra_graphs import cycle_barbell, necklace, pinwheel


# ===========================================================================
# 1. Basic attributes and dimensions
# ===========================================================================

class TestBasicAttributes:

    def test_n_matches_graph_order(self, G):
        assert G.n == G.G.order()

    def test_m_matches_graph_size(self, G):
        assert G.m == G.G.size()

    def test_A_is_adjacency_matrix(self, G):
        expected = np.array(G.G.adjacency_matrix())
        assert np.array_equal(G.A, expected)

    def test_A_is_symmetric(self, G):
        assert np.array_equal(G.A, G.A.T)

    def test_A_diagonal_is_zero(self, G):
        assert np.all(np.diag(G.A) == 0)

    def test_S_shape(self, G):
        assert G.S.shape == (2 * G.m, G.n)

    def test_T_shape(self, G):
        assert G.T.shape == (G.n, 2 * G.m)

    def test_tau_shape(self, G):
        assert G.tau.shape == (2 * G.m, 2 * G.m)

    def test_B_shape(self, G):
        assert G.B.shape == (2 * G.m, 2 * G.m)

    def test_S_is_binary(self, G):
        assert set(np.unique(G.S)).issubset({0, 1})

    def test_T_is_binary(self, G):
        assert set(np.unique(G.T)).issubset({0, 1})

    def test_tau_is_binary(self, G):
        assert set(np.unique(G.tau)).issubset({0, 1})


# ===========================================================================
# 2. Incidence operator structure and identities
# ===========================================================================

class TestIncidenceOperators:

    def test_C_equals_S_at_T(self, G):
        assert np.allclose(G.C, G.S @ G.T)

    def test_B_equals_C_minus_tau(self, G):
        assert np.allclose(G.B, G.C - G.tau)

    def test_tau_is_involution(self, G):
        """tau applied twice is the identity."""
        assert np.allclose(G.tau @ G.tau, np.eye(2 * G.m))

    def test_tau_is_symmetric(self, G):
        assert np.array_equal(G.tau, G.tau.T)

    def test_T_at_S_equals_adjacency(self, G):
        """T @ S reconstructs the adjacency matrix."""
        assert np.allclose(G.T @ G.S, G.A)

    def test_S_colsums_equal_degrees(self, G):
        """Each column of S sums to deg(v): the number of directed edges ending at v."""
        expected = np.array(G.G.degree())
        assert np.allclose(G.S.sum(axis=0), expected)

    def test_T_rowsums_equal_degrees(self, G):
        """Each row of T sums to deg(v): the number of directed edges starting at v."""
        expected = np.array(G.G.degree())
        assert np.allclose(G.T.sum(axis=1), expected)

    def test_each_directed_edge_has_exactly_one_startpoint(self, G):
        assert np.all(G.T.sum(axis=0) == 1)

    def test_each_directed_edge_has_exactly_one_endpoint(self, G):
        assert np.all(G.S.sum(axis=1) == 1)

    def test_B_diagonal_is_zero(self, G):
        """An edge cannot be its own non-backtracking successor."""
        assert np.all(np.diag(G.B) == 0)

    def test_B_and_tau_are_disjoint(self, G):
        """B[e,f]=1 and tau[e,f]=1 cannot both hold (B excludes reversed edges)."""
        assert np.all((G.B * G.tau) == 0)


# ===========================================================================
# 3. Degree matrices
# ===========================================================================

class TestDegreeMatrices:

    def test_D_is_diagonal(self, G):
        assert np.allclose(G.D, np.diag(np.diag(G.D)))

    def test_D_inv_is_inverse_of_D(self, G):
        assert np.allclose(G.D_inv @ G.D, np.eye(G.n))

    def test_De_is_diagonal(self, G):
        assert np.allclose(G.De, np.diag(np.diag(G.De)))

    def test_De_inv_is_inverse_of_De(self, G):
        assert np.allclose(G.De_inv @ G.De, np.eye(2 * G.m))

    def test_De_entries_are_at_least_2(self, G):
        """NBRW requires min degree ≥ 2 so De - I is invertible."""
        assert np.all(np.diag(G.De) >= 2)

    def test_D_diagonal_entries_match_degrees(self, G):
        deg = np.array(G.G.degree())
        assert np.allclose(np.sort(np.diag(G.D)), np.sort(deg))


# ===========================================================================
# 4. Stationary distributions
# ===========================================================================

class TestStationaryDistributions:

    def test_pi_sums_to_one(self, G):
        assert np.isclose(G.pi.sum(), 1.0)

    def test_pi_e_sums_to_one(self, G):
        assert np.isclose(G.pi_e.sum(), 1.0)

    def test_pi_e_is_uniform(self, G):
        assert np.allclose(G.pi_e, np.ones(2 * G.m) / (2 * G.m))

    def test_pi_proportional_to_degrees(self, G):
        """pi[v] = deg(v) / (2m) for the SRW on an undirected graph."""
        deg = np.array(G.G.degree())
        expected = deg / (2 * G.m)
        assert np.allclose(np.sort(G.pi), np.sort(expected))

    def test_pi_is_stationary_for_SRW(self, G):
        assert np.allclose(G.pi @ G.P, G.pi)

    def test_pi_e_is_stationary_for_Pnb(self, G):
        assert np.allclose(G.pi_e @ G.Pnb, G.pi_e)

    def test_pi_e_is_stationary_for_Pe(self, G):
        assert np.allclose(G.pi_e @ G.Pe, G.pi_e)

    def test_pi_nonnegative(self, G):
        assert np.all(G.pi >= 0)

    def test_pi_e_nonnegative(self, G):
        assert np.all(G.pi_e >= 0)


# ===========================================================================
# 5. Transition matrices
# ===========================================================================

class TestTransitionMatrices:

    def test_P_rows_sum_to_one(self, G):
        assert np.allclose(G.P.sum(axis=1), np.ones(G.n))

    def test_Pnb_rows_sum_to_one(self, G):
        assert np.allclose(G.Pnb.sum(axis=1), np.ones(2 * G.m))

    def test_Pe_rows_sum_to_one(self, G):
        assert np.allclose(G.Pe.sum(axis=1), np.ones(2 * G.m))

    def test_P_is_nonnegative(self, G):
        assert np.all(G.P >= -1e-12)

    def test_Pnb_is_nonnegative(self, G):
        assert np.all(G.Pnb >= -1e-12)

    def test_Pe_is_nonnegative(self, G):
        assert np.all(G.Pe >= -1e-12)

    def test_P_equals_Dinv_A(self, G):
        assert np.allclose(G.P, G.D_inv @ G.A)

    def test_Pe_equals_DeInv_C(self, G):
        assert np.allclose(G.Pe, G.De_inv @ G.C)

    def test_Pnb_diagonal_is_zero(self, G):
        """Cannot stay on the same directed edge."""
        assert np.allclose(np.diag(G.Pnb), 0)

    def test_Pnb_and_tau_are_disjoint(self, G):
        """The non-backtracking walk cannot traverse the reverse edge."""
        assert np.all((G.Pnb > 1e-12) & (G.tau > 0.5) == False)


# ===========================================================================
# 6. Fundamental matrices
# ===========================================================================

class TestFundamentalMatrices:

    def test_Z_satisfies_fundamental_equation(self, G):
        """(I - P + Wv) @ Z = I by definition of Z."""
        size = G.n
        lhs = (np.eye(size) - G.P + G.Wv) @ G.Z
        assert np.allclose(lhs, np.eye(size))

    def test_Z_e_satisfies_fundamental_equation(self, G):
        size = 2 * G.m
        lhs = (np.eye(size) - G.Pe + G.We) @ G.Z_e
        assert np.allclose(lhs, np.eye(size))

    def test_Znb_e_satisfies_fundamental_equation(self, G):
        size = 2 * G.m
        lhs = (np.eye(size) - G.Pnb + G.Wnb) @ G.Znb_e
        assert np.allclose(lhs, np.eye(size))

    def test_Znb_satisfies_fundamental_equation(self, G):
        """Znb = I + D^{-1} T (I-Pnb+Wnb)^{-1} S - Wv, tested by reconstructing via definition."""
        reconstructed = (
            np.eye(G.n)
            + G.D_inv @ G.T @ G.Znb_e @ G.S
            - G.Wv
        )
        assert np.allclose(G.Znb, reconstructed)

    def test_Wv_rows_are_pi(self, G):
        expected = np.outer(np.ones(G.n), G.pi)
        assert np.allclose(G.Wv, expected)

    def test_We_rows_are_pi_e(self, G):
        expected = np.outer(np.ones(2 * G.m), G.pi_e)
        assert np.allclose(G.We, expected)

    def test_Wnb_is_uniform(self, G):
        expected = np.ones((2 * G.m, 2 * G.m)) / (2 * G.m)
        assert np.allclose(G.Wnb, expected)


# ===========================================================================
# 7. Mean first-passage time matrices
# ===========================================================================

class TestMFPTMatrices:

    def test_Mv_diagonal_is_zero(self, G):
        assert np.allclose(np.diag(G.Mv), 0.0)

    def test_Me_diagonal_is_zero(self, G):
        assert np.allclose(np.diag(G.M_e), 0.0)

    def test_Mnb_diagonal_is_zero(self, G):
        assert np.allclose(np.diag(G.Mnb), 0.0)

    def test_Mv_is_nonnegative(self, G):
        assert np.all(G.Mv >= -1e-10)

    def test_Me_is_nonnegative(self, G):
        assert np.all(G.M_e >= -1e-10)

    def test_Mv_kemeny_row_consistency(self, G):
        """For SRW, sum_j pi[j]*Mv[i,j] = Kv for every starting vertex i."""
        row_sums = G.Mv @ G.pi
        assert np.allclose(row_sums, G.Kv * np.ones(G.n), rtol=1e-6)

    def test_Me_kemeny_row_consistency(self, G):
        """For SRW in edge space, sum_f pi_e[f]*M_e[e,f] = Ke for every starting edge e."""
        row_sums = G.M_e @ G.pi_e
        assert np.allclose(row_sums, G.Ke * np.ones(2 * G.m), rtol=1e-6)

    def test_Mev_shape(self, G):
        assert G.Mev.shape == (2 * G.m, G.n)

    def test_Mev_nonnegative(self, G):
        assert np.all(G.Mev >= -1e-10)

    def test_Mnb_e_shape(self, G):
        assert G.Mnb_e.shape == (2 * G.m, 2 * G.m)

    def test_Mnb_e_diagonal_is_zero(self, G):
        assert np.allclose(np.diag(G.Mnb_e), 0.0)

    def test_Mnb_e_is_nonnegative(self, G):
        assert np.all(G.Mnb_e >= -1e-10)


# ===========================================================================
# 8. Kemeny's constants – formulas and cross-method agreement
# ===========================================================================

class TestKemenyConstants:

    def test_Kv_equals_trace_Z_minus_one(self, G):
        assert np.isclose(G.Kv, np.trace(G.Z) - 1, rtol=1e-6)

    def test_Ke_equals_Kv_plus_2m_minus_n(self, G):
        assert np.isclose(G.Ke, G.Kv + 2 * G.m - G.n, rtol=1e-6)

    def test_Ke_equals_trace_Ze_minus_one(self, G):
        assert np.isclose(G.Ke, np.trace(G.Z_e) - 1, rtol=1e-6)

    def test_Knb_e_equals_trace_Znb_e_minus_one(self, G):
        assert np.isclose(G.Knb_e, np.trace(G.Znb_e) - 1, rtol=1e-6)

    def test_Knb_v_trace_equals_trace_Znb_minus_one(self, G):
        assert np.isclose(G.Knb_v_trace, np.trace(G.Znb) - 1, rtol=1e-6)

    def test_Knb_v_trace_and_mfpt_agree(self, G):
        """Knb_v_trace and Knb_v_mfpt agree for edge-transitive graphs; open question otherwise."""
        if not G.G.is_edge_transitive():
            pytest.skip("agreement only guaranteed for edge-transitive graphs")
        assert np.isclose(G.Knb_v_trace, G.Knb_v_mfpt, rtol=1e-5)

    def test_Kv_is_positive(self, G):
        assert G.Kv > 0

    def test_Ke_is_positive(self, G):
        assert G.Ke > 0

    def test_Knb_e_is_positive(self, G):
        assert G.Knb_e > 0

    def test_Knb_v_trace_is_positive(self, G):
        assert G.Knb_v_trace > 0

    def test_Ke_geq_Kv(self, G):
        """Edge-space Kemeny ≥ vertex-space Kemeny since 2m ≥ n for connected graphs."""
        assert G.Ke >= G.Kv - 1e-10


# ===========================================================================
# 9. Mean return time matrices
# ===========================================================================

class TestMeanReturnTimes:

    def test_R_e_equals_2m_identity(self, G):
        """SRW mean return time in edge space = 2m (since pi_e is uniform 1/2m)."""
        expected = 2 * G.m * np.eye(2 * G.m)
        assert np.allclose(G.R_e, expected)

    def test_R_is_diagonal(self, G):
        assert np.allclose(G.R, np.diag(np.diag(G.R)))

    def test_R_diagonal_equals_reciprocal_pi(self, G):
        """SRW mean return time to vertex v = 1/pi[v] by Kac's theorem."""
        expected_diagonal = 1.0 / G.pi
        assert np.allclose(np.diag(G.R), expected_diagonal, rtol=1e-6)

    def test_R_entries_are_positive(self, G):
        assert np.all(np.diag(G.R) > 0)


# ===========================================================================
# 10. Regular graphs – additional structural checks
# ===========================================================================

class TestRegularGraphs:
    """Extra invariants that hold specifically for regular graphs."""

    @pytest.mark.parametrize("fixture_name", ["G_k4", "G_k5", "G_petersen"])
    def test_pi_is_uniform_for_regular(self, fixture_name, request):
        G = request.getfixturevalue(fixture_name)
        assert np.allclose(G.pi, np.ones(G.n) / G.n)

    @pytest.mark.parametrize("fixture_name", ["G_k4", "G_k5", "G_petersen"])
    def test_P_is_doubly_stochastic_for_regular(self, fixture_name, request):
        G = request.getfixturevalue(fixture_name)
        assert np.allclose(G.P.sum(axis=0), np.ones(G.n))

    def test_Kv_for_complete_k4(self, G_k4):
        """For K_n, Kv = (n-1)^2/n from the eigenvalue formula (eigenvalues of P are 1 and -1/(n-1))."""
        n = G_k4.n
        assert np.isclose(G_k4.Kv, (n - 1) ** 2 / n, rtol=1e-6)

    def test_Kv_for_complete_k5(self, G_k5):
        n = G_k5.n
        assert np.isclose(G_k5.Kv, (n - 1) ** 2 / n, rtol=1e-6)


# ===========================================================================
# 11. Bipartite graph checks
# ===========================================================================

class TestBipartiteGraph:

    def test_A_is_bipartite_block_structure(self, G_k33):
        """K_{3,3} adjacency has zero 3×3 diagonal blocks."""
        A = G_k33.A
        assert np.all(A[:3, :3] == 0)
        assert np.all(A[3:, 3:] == 0)

    def test_pi_uniform_for_regular_bipartite(self, G_k33):
        expected = np.ones(G_k33.n) / G_k33.n
        assert np.allclose(G_k33.pi, expected)


# ===========================================================================
# 12. cycle=True flag – partial attribute set
# ===========================================================================

class TestCycleFlag:

    def test_cycle5_basic_attributes_exist(self, G_cycle5):
        G = G_cycle5
        assert G.n == 5
        assert G.m == 5

    def test_cycle5_P_rows_sum_to_one(self, G_cycle5):
        assert np.allclose(G_cycle5.P.sum(axis=1), np.ones(G_cycle5.n))

    def test_cycle5_Pnb_rows_sum_to_one(self, G_cycle5):
        assert np.allclose(G_cycle5.Pnb.sum(axis=1), np.ones(2 * G_cycle5.m))

    def test_cycle5_pi_sums_to_one(self, G_cycle5):
        assert np.isclose(G_cycle5.pi.sum(), 1.0)

    def test_cycle5_pi_stationary_for_P(self, G_cycle5):
        assert np.allclose(G_cycle5.pi @ G_cycle5.P, G_cycle5.pi)

    def test_cycle5_pi_e_stationary_for_Pnb(self, G_cycle5):
        assert np.allclose(G_cycle5.pi_e @ G_cycle5.Pnb, G_cycle5.pi_e)

    def test_cycle5_Mv_diagonal_is_zero(self, G_cycle5):
        assert np.allclose(np.diag(G_cycle5.Mv), 0.0)

    def test_cycle5_Kv_positive(self, G_cycle5):
        assert G_cycle5.Kv > 0

    def test_cycle5_Ke_relationship(self, G_cycle5):
        G = G_cycle5
        assert np.isclose(G.Ke, G.Kv + 2 * G.m - G.n, rtol=1e-6)

    def test_cycle5_nb_attributes_not_set(self, G_cycle5):
        """cycle=True skips attributes that would require inverting a singular matrix."""
        for attr in ("Znb", "Znb_e", "Mnb_e", "Knb_e", "Knb_v_trace", "Knb_v_sub"):
            assert not hasattr(G_cycle5, attr), f"{attr} should not be set when cycle=True"


# ===========================================================================
# 13. Extra graph constructors
# ===========================================================================

class TestExtraGraphConstructors:

    def test_cycle_barbell_constructs_without_error(self):
        G = NBRW(cycle_barbell(2, 4, 5))
        assert G.n > 0 and G.m > 0

    def test_cycle_barbell_vertex_count(self):
        k, a, b = 2, 4, 5
        G = NBRW(cycle_barbell(k, a, b))
        assert G.n == k + a + b - 2

    def test_necklace_constructs_without_error(self):
        G = NBRW(necklace(2))
        assert G.n > 0 and G.m > 0

    def test_necklace_raises_for_k_less_than_2(self):
        with pytest.raises(ValueError):
            necklace(1)

    def test_necklace_vertex_count(self):
        k = 3
        G = NBRW(necklace(k))
        assert G.n == 4 * k + 2

    def test_pinwheel_constructs_without_error(self):
        G = NBRW(pinwheel([3, 4, 5]), pinwheel=True)
        assert G.n > 0 and G.m > 0

    def test_pinwheel_has_central_vertex(self):
        """Central vertex (vertex 0) should have degree = 2 × number of blades."""
        cycle_sizes = [3, 4, 5]
        G = NBRW(pinwheel(cycle_sizes), pinwheel=True)
        central_degree = G.G.degree(0)
        assert central_degree == 2 * len(cycle_sizes)

    def test_pinwheel_basic_invariants(self, G_pinwheel):
        G = G_pinwheel
        assert np.isclose(G.pi.sum(), 1.0)
        assert np.allclose(G.P.sum(axis=1), np.ones(G.n))
        assert np.allclose(G.Pnb.sum(axis=1), np.ones(2 * G.m))

    def test_pinwheel_kemeny_consistency(self, G_pinwheel):
        G = G_pinwheel
        assert np.isclose(G.Ke, G.Kv + 2 * G.m - G.n, rtol=1e-6)


# ===========================================================================
# 14. Petersen graph – known analytic values
# ===========================================================================

class TestPetersenKnownValues:
    """
    The Petersen graph is 3-regular with n=10, m=15.
    For a k-regular graph: Kv = n/(n-1) * (n-1) ... actually for
    a k-regular graph Kv can be computed from eigenvalues.
    The Petersen graph has eigenvalues 3 (×1), 1 (×5), -2 (×4).
    Kv = sum_{i≥2} 1/(1 - lambda_i/k) = 5*(1/(1-1/3)) + 4*(1/(1+2/3))
       = 5*(3/2) + 4*(3/5) = 7.5 + 2.4 = 9.9
    """

    def test_petersen_n_and_m(self, G_petersen):
        assert G_petersen.n == 10
        assert G_petersen.m == 15

    def test_petersen_Kv(self, G_petersen):
        assert np.isclose(G_petersen.Kv, 9.9, rtol=1e-6)

    def test_petersen_Ke(self, G_petersen):
        # Ke = Kv + 2m - n = 9.9 + 30 - 10 = 29.9
        assert np.isclose(G_petersen.Ke, 29.9, rtol=1e-6)

    def test_petersen_pi_is_uniform(self, G_petersen):
        assert np.allclose(G_petersen.pi, np.ones(10) / 10)

    def test_petersen_Pnb_is_uniform_over_successors(self, G_petersen):
        """Each directed edge has exactly 2 non-backtracking successors (degree 3 - 1).
        So Pnb should have entries 0 or 1/2."""
        G = G_petersen
        nonzero = G.Pnb[G.Pnb > 1e-12]
        assert np.allclose(nonzero, 0.5)


# ===========================================================================
# 15. K4 – small exact checks
# ===========================================================================

class TestK4KnownValues:
    """
    K_4: n=4, m=6, 3-regular.
    Eigenvalues of adjacency matrix: 3 (×1), -1 (×3).
    Eigenvalues of P = A/3: 1 (×1), -1/3 (×3).
    Kv = 3 * 1/(1-(-1/3)) = 3 * 3/4 = 9/4 = 2.25
    Ke = Kv + 2m - n = 2.25 + 8 = 10.25
    """

    def test_k4_n_m(self, G_k4):
        assert G_k4.n == 4
        assert G_k4.m == 6

    def test_k4_Kv(self, G_k4):
        assert np.isclose(G_k4.Kv, 9 / 4, rtol=1e-6)

    def test_k4_Ke(self, G_k4):
        assert np.isclose(G_k4.Ke, 9 / 4 + 8, rtol=1e-6)

    def test_k4_Pnb_entries_are_half(self, G_k4):
        """K_4 is 3-regular so each directed edge has exactly 2 NB successors → weight 1/2."""
        G = G_k4
        nonzero = G.Pnb[G.Pnb > 1e-12]
        assert np.allclose(nonzero, 0.5)

    def test_k5_Kv(self, G_k5):
        """K_5: eigenvalues of P are 1 and -1/4 (×4). Kv = 4/(1+1/4) = 16/5 = 3.2."""
        assert np.isclose(G_k5.Kv, 16 / 5, rtol=1e-6)
