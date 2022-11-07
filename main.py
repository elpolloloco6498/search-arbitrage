from pprint import pprint
import ccxt
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# gather market data from API
# store the data into a file market_data.json
# FORMAT:
"""
[
    {
        'pair': 'ETH/BTC',
        'volume': ...,
        'base': ...,
        'quote': ...
    },
    ...
]
"""
# read the file
#

exchange = ccxt.binance()
markets = exchange.fetch_markets()[:200]
tickers = exchange.fetch_tickers()
#pprint(tickers)
print(tickers['ETH/BTC'])

"""
# filter markets
markets = [market for market in markets if market['type'] == 'spot' and market['active']]
print(len(markets))

# construct adjacency matrix for the markets
# get label nodes
nodes_label_to_id = dict()
base_quote_relation = dict()

i = 0
for market in markets:
    # select only spot markets that are active
    #print(market)
    base, quote = market['base'], market['quote']
    base_quote_relation[base] = quote
    if base not in nodes_label_to_id:
        nodes_label_to_id[base] = i
        i += 1
    if quote not in nodes_label_to_id:
        nodes_label_to_id[quote] = i
        i += 1

nb_nodes = len(nodes_label_to_id)
adjacency_matrix = [[0]*nb_nodes for i in range(nb_nodes)]

for market in markets:
    base, quote = market['base'], market['quote']
    id_base = nodes_label_to_id[base]
    id_quote = nodes_label_to_id[quote]
    adjacency_matrix[id_base][id_quote] = 1  # the rate has to be changed using ticker information
    adjacency_matrix[id_quote][id_base] = 1

adjacency_np_matrix = np.matrix(adjacency_matrix)
# creating the network graph
graph = nx.from_numpy_matrix(adjacency_np_matrix)

# display graph
labelNodes = {v: k for k, v in nodes_label_to_id.items()}

nx.draw(graph, labels=labelNodes, with_labels=True, node_size=200, node_color='lightgray', edgecolors='red', font_size=10, width=0.5)
plt.show()

# search cycles in the graph
#cycle = nx.find_negative_cycle(graph, 0)

H = graph.to_directed()
simple_cycles = list(nx.simple_cycles(H))
for cycle in simple_cycles:
    if len(cycle) > 2:
        for node in cycle:
            print(labelNodes[node], end=" ")
        print("\n")"""