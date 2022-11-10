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
from searchCycles import *

def load_static_data_into_file(exchange):
    markets = exchange.fetch_markets()[:500]
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
    #print(tickers)
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
        adjacency_matrix[id_base][id_quote] = float(ask)
        adjacency_matrix[id_quote][id_base] = 1/float(bid)
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


def calculate_arbitrage(cycle, g):
    sum_weights_cycle = sum([g[cycle[i]][cycle[i+1]]["weight"] for i in range(len(cycle)-1)])
    return math.exp(-sum_weights_cycle)


def pair_cycle_from_id_cycle(cycle, id_to_symbol):
    return [id_to_symbol[cycle[i]] for i in range(len(cycle))]


def search_surface_rate_arbitrage(graph, symbol_to_id):
    if nx.negative_edge_cycle(graph):
        print("Arbitrage opportunity FOUND")
        # find all cycles
        unique_cycles = all_negative_cycles(graph)
        unique_cycles = list(filter(lambda cycle_arb: len(cycle_arb) >= 4, unique_cycles)) # filter cycles under 3
        # calculate arbitrage
        label_nodes = {v: k for k, v in symbol_to_id.items()}
        for cycle in unique_cycles:
            arb = calculate_arbitrage(cycle, graph)
            pair_cycle = pair_cycle_from_id_cycle(cycle, label_nodes)
            profit_perc = (arb-1)*100
            print(f"{pair_cycle} returns : {profit_perc}%")

    else:
        print("No arbitrage opportunities")
        return None


def search_deep_orderbook_validated_arbitrage():
    pass


def main():
    # TODO calculate run time and print it
    np.set_printoptions(precision=8, suppress=True)
    exch = ccxt.binance()

    load_static_data_into_file(exch)  # load static data from the API
    df = read_data("./market_data.csv")  # read the stored data
    symbol_to_id = symbol_to_matrix_id(df)  # establish a correspondance relationship between the symbol and matrix
    get_price_data(exch, df)  # gather price information from the API

    matrix = generate_adjacency_matrix(df, symbol_to_id)  # generate the adjacency matrix of the problem
    transformed_matrix = transform_matrix(matrix)
    print(transformed_matrix)

    graph = nx.from_numpy_matrix(transformed_matrix, create_using=nx.DiGraph)  # creating the graph representation via the adjacency matrix
    search_surface_rate_arbitrage(graph, symbol_to_id)

    # graphical representation
    #plot_market_graph(graph, symbol_to_id)

if __name__ == "__main__":
    main()