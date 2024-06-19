import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import os


df = pd.read_csv('customdata/finaldata.csv')
G = nx.from_pandas_edgelist(df, 'source', 'destination')
graph = nx.from_pandas_edgelist(df, 'source', 'destination')
# nx.draw(G, with_labels=True)
# plt.show()

df1 = pd.read_csv('customdata/finaltestdata.csv')
G1 = nx.from_pandas_edgelist(df1, 'source', 'destination')
# nx.draw(G1, with_labels=True)
# plt.show()
