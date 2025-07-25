# NBRW

Non-backtracking random walks (NBRW). A nonbacktracking random walk is a random walk on a graph in which the walker is restricted from visiting the previous node. 

This repository's main focus is to create the NBRW package, which contains the following files:
	 
  NBRW.py : an NBRW class dedicated to storing relevant attributes such as Kemeny's constant, mean first passage times, the 
	fundamental matrix, stationary vector, etc.
	 
  extra_graphs.py : Includes several functions that create graph families as a SageMath Graph object. These graph families have been 
	important to our research and are not built-in in SageMath
	
This work was primarily motivated by research conducted alongside Adam Knudson, Dr. Mark Kempton, and Dr. Jane Breen. This code has been incredibly useful for modeling these walks and for exploring theoretical results through numerical experimentation.

Much of this code relies on SageMath, which has many built-in functions for graph theory. 


## Instructions:

You must have SageMath installed; refer [here](https://doc.sagemath.org/html/en/installation/index.html) on how to do this.

You can then install this package with via

```
pip install NBRW
```

Refer to the PyPI page for this package [here](https://pypi.org/project/NBRW/) for any version changes.

### Sample Usage

Visit [this website](https://mwshumway.github.io/NBRW/) for documentation and sample usage.
