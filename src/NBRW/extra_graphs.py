"""graphs.py : This module contains several functions to generate graphs that are not built-in in the SageMath library.
Matthew Shumway, 2024

Work in conjunction with Adam Knudson, Dr. Mark Kempton, and Dr. Jane Breen.

This module will be included in the NBRW package, but is not the main focus of the package. It is a utility module that will be used to 
generate graphs. For example, this module will contain functionality to generate the following graphs:
    - Cycle Barbell Graphs, a path of k vertices connecting two cycle graphs each respectively of a and b vertices.
    - Necklace graphs with k beads.
    - Pinwheel graphs with k blades each of potentially different size."""

# Imports
from sage.all import *
import numpy as np

def cycle_barbell(k: int, a: int, b: int) -> Graph:
    """
    Path on k verties, the end two are shared with a cycle of size a and a cycle of size b
    so |V(G)| = a + b + c - 2
    """
    CB = Graph(k + a + b - 2)
    
    # make the a-cycle
    for i in range(a - 1):
        CB.add_edge(i, i+1)
    CB.add_edge(0, a-1)

    for i in range(k-1):
        CB.add_edge(a-1+i, a+i)
    
    for i in range(b-1):
        CB.add_edge(a+k-2+i, a+k-1+i)
    CB.add_edge(a+k-2, a+b+k-3)

    return CB


def necklace(k):
    """
    |V| = n = 4k + 2
    k is number of "beads" including the end ones which are a little different
    (so only works if k\geq2)
    """
    n = 4*k + 2
    G = Graph(n)
    #first bead
    G.add_edges([(0,1), (0, 2), (0,3), (1,2), (1, 3), (2,4), (3,4)])
    #last bead
    G.add_edges([(n-1, n-2), (n-1, n-3), (n-1, n-4), (n-2, n-3), (n-2, n-4), (n-3, n-5), (n-4, n-5)])
    #All the middle ones
    for i in range(1, k-1):
        G.add_edge(4*i, 4*i + 1)
        G.add_edges([(4*i+1, 4*i + 2), (4*i+1, 4*i+3), (4*i+2, 4*i+3), (4*i+2, 4*(i+1)), (4*i+3, 4*(i+1))])
        
    #Final edge
    G.add_edge(n - 6, n-5)
    return G


def pinwheel(cycle_sizes: list[int]):
    """Accepts a list of integers called cycle_sizes, which will determine the number 
    of cycles and also their length."""
    # Create an empty graph
    G = Graph()
    
    # Add the central vertex
    central_vertex = 0
    
    # Track the next available vertex id
    next_vertex_id = 1
    
    for size in cycle_sizes:
        # Create a cycle graph with an additional vertex for the central vertex
        C = Graph()
        C.add_vertex(central_vertex)
        
        # Add the vertices for the cycle
        cycle_vertices = [next_vertex_id + i for i in range(size - 1)]
        C.add_vertices(cycle_vertices)
        
        # Add edges to form the cycle including the central vertex
        for i in range(size - 1):
            if i == 0:
                # Connect the central vertex to the first vertex in the cycle
                C.add_edge(central_vertex, cycle_vertices[i])
            else:
                # Connect the previous vertex to the current vertex
                C.add_edge(cycle_vertices[i-1], cycle_vertices[i])
        
        # Complete the cycle
        C.add_edge(cycle_vertices[-1], central_vertex)
        
        # Add the cycle graph to G
        G.add_vertices(C.vertices())
        G.add_edges(C.edges())
        
        # Update the next available vertex id
        next_vertex_id += size - 1
    
    return G