# gather cryptocurrencies pair data from the api
# create a data structure to hold the data
# store the data structure into a file market_data.json
# read the file
# add price information to the data
# generate the adjacency matrix
# detect negative cycles
# compute arbitrage opportunities
# execute trade

from pprint import pprint
import ccxt
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import math

def load_static_data_into_file(exchange):
    markets = exchange.fetch_markets()[:200]
    # select markets that are active and type=spot
    data = {"pair": [], "base": [], "quote": []}
    # fill the dataframe with market data
    for market in markets:
        if market["active"] and market["type"] == "spot":
            base, quote = market["base"], market["quote"]
            symbol = f"{base}/{quote}"
            data["base"].append(base)
            data["quote"].append(quote)
            data["pair"].append(symbol)
    market_df = pd.DataFrame(data=data)

    # get all trading pairs in a list
    symbols = market_df.loc[:, "pair"].tolist()
    base_vol = []
    quote_vol = []
    tickers = exchange.fetch_tickers(symbols)
    for symbol in symbols:
        ticker_info = tickers[symbol]
        base_vol.append(ticker_info["baseVolume"])
        quote_vol.append(ticker_info["quoteVolume"])

    market_df["baseVol"] = base_vol
    market_df["quoteVol"] = quote_vol
    market_df = market_df.sort_values(by="baseVol", ascending=False).reset_index(drop=True)

    print(market_df)
    # store the dataframe to a file
    market_df.to_csv("./market_data.csv")


def read_data(path):
    #  read data to recreate the dataframe
    #  returns the data frame created
    return pd.read_csv(path)


def get_price_data(exchange, df):
    symbols = df.loc[:, "pair"].tolist()
    tickers = exchange.fetch_tickers(symbols)
    bid = []
    ask = []
    for symbol in symbols:
        ticker = tickers[symbol]
        bid.append(ticker["bid"])
        ask.append(ticker["ask"])
    df["bid"] = bid
    df["ask"] = ask

def symbol_to_matrix_id(df):
    symbol_to_id = dict()
    j = 0
    for i in range(len(df)):
        base, quote = df.loc[i, "base"], df.loc[i, "quote"]
        if base not in symbol_to_id:
            symbol_to_id[base] = j
            j += 1
        if quote not in symbol_to_id:
            symbol_to_id[quote] = j
            j += 1
    return symbol_to_id


def generate_adjacency_matrix(df, symbol_to_id):
    nb_nodes = len(symbol_to_id)
    adjacency_matrix = [[0] * nb_nodes for i in range(nb_nodes)]

    for i in range(len(df)):
        base, quote = df.loc[i, "base"], df.loc[i, "quote"]
        ask, bid = df.loc[i, "ask"], df.loc[i, "bid"]
        id_base, id_quote = symbol_to_id[base], symbol_to_id[quote]
        adjacency_matrix[id_base][id_quote] = ask  # the rate has to be changed using ticker information
        adjacency_matrix[id_quote][id_base] = 1/bid
    return np.matrix(adjacency_matrix, dtype=float)


def plot_market_graph(graph, nodes_label_to_id):
    label_nodes = {v: k for k, v in nodes_label_to_id.items()}
    nx.draw(graph, labels=label_nodes, with_labels=True, node_size=200, node_color='lightgray', edgecolors='red',
            font_size=10, width=0.5)
    plt.show()


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

def find_negative_cycles(graph):
    #label_nodes = {v: k for k, v in symbol_to_id.items()}
    #directed_graph = graph.to_directed()
    print(getNegativeCycle(graph))
    """
    simple_cycles = list(nx.simple_cycles(directed_graph))
    for cycle in simple_cycles:
        if len(cycle) > 2:
            for node in cycle:
                print(label_nodes[node], end=" ")
            print("\n")"""


def search_surface_rate_arbitrage():
    pass


def search_deep_orderbook_validated_arbitrage():
    pass


def main():
    # TODO calculate run time and print it
    """
    np.set_printoptions(precision=8, suppress=True)
    exch = ccxt.binance()
    #load_static_data_into_file(exch)  # load static data from the API
    df = read_data("./market_data.csv")  # read the stored data
    symbol_to_id = symbol_to_matrix_id(df)  # establish a correspondance relationship between the symbol and matrix ID
    get_price_data(exch, df)  # gather price information from the API
    matrix = generate_adjacency_matrix(df, symbol_to_id)  # generate the adjacency matrix of the problem
    """
    #potato:0 carrot:1 lettuce:2
    matrix_food_market = np.matrix([
        [0, 0.5, 0.5],
        [2, 0, 0.5],
        [2, 2, 0]
    ])
    #pound:0, dollar:1, yen:2
    matrix_forex_market = np.matrix([
        [0, 1/0.8, 100],
        [0.8, 0, 1/0.013],
        [1/100, 0.013, 0]
    ])
    print(matrix_forex_market)
    transformed_matrix = transform_matrix(matrix_forex_market)
    print(transformed_matrix)
    print(transformed_matrix.shape)
    sum = transformed_matrix[1,0]+transformed_matrix[0,2]+transformed_matrix[2,1]
    print(sum)

    graph = nx.from_numpy_matrix(matrix_forex_market)  # creating the graph representation via the adjacency matrix
    #print(getNegativeCycle(graph))
    print("negative cycle : ", nx.negative_edge_cycle(graph))
    cycle = nx.find_negative_cycle(graph, 0)
    print("negative cycle 2: ", cycle)
    #plot_market_graph(graph, symbol_to_id)

    # draw graph test
    nx.draw(graph, node_size=200, node_color='lightgray', edgecolors='red',
            font_size=10, width=0.5)
    plt.show()

if __name__ == "__main__":
    main()


"""
# search cycles in the graph
#cycle = nx.find_negative_cycle(graph, 0)

H = graph.to_directed()
simple_cycles = list(nx.simple_cycles(H))
for cycle in simple_cycles:
    if len(cycle) > 2:
        for node in cycle:
            print(labelNodes[node], end=" ")
        print("\n")
"""
