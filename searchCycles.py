import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import math
from collections import defaultdict


def bellman_ford_negative_cycles(g, s):
    """
    Bellman Ford, modified so that it returns cycles.
    Runtime is O(VE).
    :param g: graph
    :type g: networkx weighted DiGraph
    :param s: source vertex
    :type s: str
    :return: all negative-weight cycles reachable from a source vertex
    :rtype: str list (empty if no neg-weight cyc)
    """
    n = len(g.nodes())
    d = defaultdict(lambda: math.inf)  # distances dict
    p = defaultdict(lambda: -1)  # predecessor dict
    d[s] = 0

    for _ in range(n - 1):
        for u, v in g.edges():
            # Bellman-Ford relaxation
            weight = g[u][v]["weight"]
            if d[u] + weight < d[v]:
                d[v] = d[u] + weight
                p[v] = u  # update pred

    # Find cycles if they exist
    all_cycles = []
    seen = defaultdict(lambda: False)

    for u, v in g.edges():
        weight = g[u][v]["weight"]
        # If we can relax further, there must be a neg-weight cycle
        if seen[v]:
            continue

        if d[u] + weight < d[v]:
            cycle = []
            x = v
            while True:
                # Walk back along predecessors until a cycle is found
                seen[x] = True
                cycle.append(x)
                x = p[x]
                if x == v or x in cycle:
                    break
            # Slice to get the cyclic portion
            idx = cycle.index(x)
            cycle.append(x)
            all_cycles.append(cycle[idx:][::-1])
    return all_cycles


def all_negative_cycles(g):
    """
    Get all negative-weight cycles by calling Bellman-Ford on
    each vertex. O(V^2 E)
    :param g: graph
    :type g: networkx weighted DiGraph
    :return: list of negative-weight cycles
    :rtype: list of str list
    """
    all_paths = []
    for v in g.nodes():
        all_paths.append(bellman_ford_negative_cycles(g, v))
    flatten = lambda l: [item for sublist in l for item in sublist]
    return [list(i) for i in set(tuple(j) for j in flatten(all_paths))]