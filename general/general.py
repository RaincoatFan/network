import networkx as nx
import pandas as pd
from networkx.algorithms import approximation as approx
import csv

# Read the CSV file into a pandas DataFrame
first_set = []
second_set = []
distance_set = []

with open('data4_distance.csv', encoding='utf-8') as f:
    for row in csv.reader(f, skipinitialspace=True):
        first_set.append(row[0])
        second_set.append(row[1])
        distance_set.append(row[2])
    del first_set[0], second_set[0], distance_set[0]

# print(first_set,second_set,distance_set)

edge_set = []
for item in range(len(first_set)):
    edge_set.append((str(first_set[item]),str(second_set[item]),float(distance_set[item])))

# print(edge_set)
# Create an empty graph
G = nx.Graph()

G.add_weighted_edges_from(edge_set)

cycle = approx.simulated_annealing_tsp(G, "greedy", source="0")
cost = sum(G[n][nbr]["weight"] for n, nbr in nx.utils.pairwise(cycle))

print(cycle,cost)