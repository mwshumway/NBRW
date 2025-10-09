# NBRW

[[Paper]](https://arxiv.org/abs/2510.06650) | [[Documentation]](https://mwshumway.github.io/NBRW/) | [[PyPI]](https://pypi.org/project/NBRW/)

Non-backtracking random walks (NBRW). A nonbacktracking random walk is a random walk on a graph in which the walker is restricted from visiting the previous node.

This repository's main focus is to create the NBRW package, which contains the following files:

- `NBRW.py`: An NBRW class dedicated to storing relevant attributes such as Kemeny's constant, mean first passage times, the fundamental matrix, stationary vector, etc.

- `extra_graphs.py`: Includes several functions that create graph families as SageMath Graph objects. These graph families have been important to our research and are not built-in in SageMath.

This work was primarily motivated by research conducted alongside Adam Knudson, Mark Kempton, and Jane Breen. This code has been incredibly useful for modeling these walks and for exploring theoretical results through numerical experimentation.

Much of this code relies on SageMath, which has many built-in functions for graph theory.

## Instructions

You must have SageMath installed; refer to [the SageMath installation guide](https://doc.sagemath.org/html/en/installation/index.html) on how to do this. From my experience, it can be very cumbersome
to get Sage to import in my IDE, so I was in the habit of developing either in either the Jupyter notebook or terminal environment that SageMath opens upon launching.

You can then install this package with via

```bash
pip install NBRW
```

Refer to the [NBRW PyPI page](https://pypi.org/project/NBRW/) for any version changes.

### Sample Usage

```python
from nbrw import NBRW
from sage.all import *

g = graphs.PetersenGraph()
G = NBRW(g)
```

```python
from nbrw import extra_graphs as exg

N = exg.necklace(5)                     # necklace graph with 5 beads
C = exg.cycle_barbell(k=2, a=5, b=6)    # cycle barbell with 2-path, 5-cycle and a 6-cycle
P = exg.pinwheel([3, 4, 5])             # pinwheel graph with 3 spokes of 3-, 4-, and 5- cycles
```
