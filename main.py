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

exchange = ccxt.binance()


def load_static_data_into_file(exchange):
    markets = exchange.fetch_markets()[:100]
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
        id_base, id_quote = symbol_to_id[base], symbol_to_id[quote]
        adjacency_matrix[id_base][id_quote] = 1  # the rate has to be changed using ticker information
        adjacency_matrix[id_quote][id_base] = 1
    return np.matrix(adjacency_matrix)

def plot_market_graph(adjacency_np_matrix, nodes_label_to_id):
    graph = nx.from_numpy_matrix(adjacency_np_matrix)

    # display graph
    labelNodes = {v: k for k, v in nodes_label_to_id.items()}

    nx.draw(graph, labels=labelNodes, with_labels=True, node_size=200, node_color='lightgray', edgecolors='red',
            font_size=10, width=0.5)
    plt.show()


def find_negative_cycles():
    pass


def find_arbitrage():
    pass

load_static_data_into_file(exchange)
df = read_data("./market_data.csv")
get_price_data(exchange, df)
symbol_to_id = symbol_to_matrix_id(df)
matrix = generate_adjacency_matrix(df, symbol_to_id)
plot_market_graph(matrix, symbol_to_id)


"""
exchange = ccxt.binance()
markets = exchange.fetch_markets()[:200]
tickers = exchange.fetch_tickers()
# pprint(tickers)
#print(tickers['ETH/BTC'])

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
    # print(market)
    base, quote = market['base'], market['quote']
    base_quote_relation[base] = quote
    if base not in nodes_label_to_id:
        nodes_label_to_id[base] = i
        i += 1
    if quote not in nodes_label_to_id:
        nodes_label_to_id[quote] = i
        i += 1

nb_nodes = len(nodes_label_to_id)
adjacency_matrix = [[0] * nb_nodes for i in range(nb_nodes)]

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

nx.draw(graph, labels=labelNodes, with_labels=True, node_size=200, node_color='lightgray', edgecolors='red',
        font_size=10, width=0.5)
plt.show()


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
