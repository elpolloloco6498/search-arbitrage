import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import math


def negative_log(v):
    if v != 0:
        return -math.log(v)
    else:
        return 0


def transform_matrix(matrix):
    vec_func = np.vectorize(negative_log, otypes=[float])
    return vec_func(matrix)

def getW(edge):
    return edge[2]['weight']


def getNegativeCycle(g):
    n = len(g.nodes())
    edges = list(g.edges().data())
    d = np.ones(n) * 10000
    p = np.ones(n)*-1
    x = -1
    for i in range(n):
        for e in edges:
            if d[int(e[0])] + getW(e) < d[int(e[1])]:
                d[int(e[1])] = d[int(e[0])] + getW(e)
                p[int(e[1])] = int(e[0])
                x = int(e[1])
    if x == -1:
        print("No negative cycle")
        return None
    for i in range(n):
        x = p[int(x)]
    cycle = []
    v = x
    while True:
        cycle.append(str(int(v)))
        if v == x and len(cycle) > 1:
            break
        v = p[int(v)]
    return list(reversed(cycle))


#pound:0, dollar:1, yen:2
matrix_forex_market = np.matrix([
    [0, 1/0.8, 100],
    [0.8, 0, 1/0.013],
    [1/100, 0.013, 0]
])
print(matrix_forex_market)
transformed_matrix = transform_matrix(matrix_forex_market)
graph = nx.from_numpy_matrix(transformed_matrix, create_using=nx.DiGraph)  # creating the graph representation via the adjacency matrix
print("negative cycle : ", nx.find_negative_cycle(graph, 0))
print("cycles :", getNegativeCycle(graph))
# draw graph test

labels = nx.get_edge_attributes(graph,'weight')
pos=nx.spring_layout(graph)
nx.draw(graph, pos=pos, node_size=200, with_labels=True, node_color='lightgray', edgecolors='red',
        font_size=10, width=0.5)
nx.draw_networkx_edge_labels(graph, pos, edge_labels=labels)
plt.show()