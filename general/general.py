import networkx as nx
import pandas as pd
from networkx.algorithms.approximation import simulated_annealing_tsp

# Read the CSV file into a pandas DataFrame
df = pd.read_csv('center1_distance.csv')

# Create an empty graph
G = nx.Graph()

# Add nodes from the 'first' and 'second' columns
nodes = set(df['first']).union(df['second'])
G.add_nodes_from(nodes)

# Add edges from the 'first', 'second', and 'distance' columns
edges = [(row['first'], row['second'], {'distance': row['distance']}) for _, row in df.iterrows()]
G.add_edges_from(edges)

# Initialize the initial cycle as a list of nodes
init_cycle = list(nodes)

# Calculate the approximate shortest path using simulated annealing TSP
approx_shortest_path = simulated_annealing_tsp(G, init_cycle)

# Print the approximate shortest path
print("Approximate Shortest Path:", approx_shortest_path)
