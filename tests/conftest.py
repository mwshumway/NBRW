import sys
import os

# Make the local src/ package take precedence over any installed version
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
from sage.all import graphs
from NBRW import NBRW
from NBRW.extra_graphs import cycle_barbell, necklace, pinwheel


# ---------------------------------------------------------------------------
# Individual graph fixtures (session-scoped so each is built once)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def G_k4():
    return NBRW(graphs.CompleteGraph(4))


@pytest.fixture(scope="session")
def G_k5():
    return NBRW(graphs.CompleteGraph(5))


@pytest.fixture(scope="session")
def G_k33():
    return NBRW(graphs.CompleteBipartiteGraph(3, 3))


@pytest.fixture(scope="session")
def G_petersen():
    return NBRW(graphs.PetersenGraph())


@pytest.fixture(scope="session")
def G_barbell():
    # cycle_barbell(k, a, b): path of k vertices joining a-cycle and b-cycle
    return NBRW(cycle_barbell(2, 4, 5))


@pytest.fixture(scope="session")
def G_necklace():
    return NBRW(necklace(3))


@pytest.fixture(scope="session")
def G_pinwheel():
    return NBRW(pinwheel([3, 4, 5]), pinwheel=True)


@pytest.fixture(scope="session")
def G_cycle5():
    # cycle=True skips matrices that become singular for cycle graphs
    return NBRW(graphs.CycleGraph(5), cycle=True)


# ---------------------------------------------------------------------------
# Parametrized fixture: all non-cycle graphs (have full attribute set)
# ---------------------------------------------------------------------------

@pytest.fixture(
    scope="session",
    params=["k4", "k5", "k33", "petersen", "barbell", "necklace", "pinwheel"],
)
def G(request, G_k4, G_k5, G_k33, G_petersen, G_barbell, G_necklace, G_pinwheel):
    return {
        "k4":       G_k4,
        "k5":       G_k5,
        "k33":      G_k33,
        "petersen": G_petersen,
        "barbell":  G_barbell,
        "necklace": G_necklace,
        "pinwheel": G_pinwheel,
    }[request.param]
