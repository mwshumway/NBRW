from NBRW import NBRW_new
from nbrw_old import NBRW_old
import numpy as np
from sage.all import *


def test_nbrw():
    for g in graphs.nauty_geng("20 -c -d3"):
        if not g.is_regular() and not g.is_edge_transitive():
            G = g
            break
    
    Go = NBRW_old(G)
    Gn = NBRW_new(G)

    assert Gn.n == Go.n, 'n is not the same'
    assert Gn.m == Go.m, 'm is not the same'
    assert np.allclose(Gn.S, Go.S), 'S is not the same'
    assert np.allclose(Gn.T, Go.T), 'T is not the same'
    assert np.allclose(Gn.tau, Go.tau), 'tau is not the same'
    assert np.allclose(Gn.B, Go.B), 'B is not the same'
    assert np.allclose(Gn.C, Go.S * Go.T), 'C is not the same'
    assert np.allclose(Gn.D_inv, Go.Dinv), 'D_inv is not the same'
    assert np.allclose(Gn.De, Go.De), 'De is not the same'
    assert np.allclose(Gn.De_inv, Go.De_inv), 'De_inv is not the same'
    assert np.allclose(Gn.pi, Go.pi), 'pi is not the same'
    assert np.allclose(Gn.pi_e, Go.pi_e), 'pi_e is not the same'
    assert np.allclose(Gn.Pnb, Go.Pnb), 'Pnb is not the same'
    assert np.allclose(Gn.P, Go.P), 'P is not the same'
    assert np.allclose(Gn.Pe, Go.Pe), 'Pe is not the same'
    assert np.allclose(Gn.Wnb, Go.Wnb), 'Wnb is not the same'
    assert np.allclose(Gn.Wv, Go.Wv), 'Wv is not the same'
    assert np.allclose(Gn.We, Go.We), 'We is not the same'
    assert np.allclose(Gn.Znb, Go.Znb), 'Znb is not the same'
    assert np.allclose(Gn.Z, Go.Z), 'Z is not the same'
    assert np.allclose(Gn.Znb_e, Go.Znb_e), 'Znb_e is not the same'
    assert np.allclose(Gn.Z_e, Go.Z_e), 'Z_e is not the same'
    assert np.allclose(Gn.M, Go.M), 'M is not the same'
    assert np.allclose(Gn.Mev, Go.Mev), 'Mev is not the same'
    assert np.allclose(Gn.Mnb, Go.Mv_nb), 'Mnb is not the same'
    assert np.allclose(Gn.Mnb_e, Go.Mnb_e), 'Mnb_e is not the same'
    assert np.allclose(Gn.Mv, Go.Mv), 'Mv is not the same'
    assert np.allclose(Gn.M_e, Go.M_e), 'M_e is not the same'
    assert np.allclose(Gn.R_e, Go.Re), 'R_e is not the same'
    assert np.allclose(Gn.R, Go.Rv), 'R is not the same'
    assert np.allclose(Gn.Kv, Go.K_vertex), 'Kv is not the same'
    assert np.allclose(Gn.Ke, Go.K_edge), 'Ke is not the same'
    assert np.allclose(Gn.Knb_e, Go.Knb_e), 'Knb_e is not the same'
    assert np.allclose(Gn.Knb_v_trace, Go.Knb_v2), 'Knb_v_trace is not the same'
    assert np.allclose(Gn.Knb_v_mfpt, Go.Knb_v), 'Knb_v_mfpt is not the same'
    assert np.allclose(Gn.Knb_v_sub, Go.Knb_e - (2*Go.m - Go.n)), 'Knb_v_sub is not the same'



