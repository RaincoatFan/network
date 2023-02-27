import networkx as nx
import matplotlib.pyplot as plt

G = nx.Graph()

# G = nx.cubical_graph()

# subax2 = plt.subplot(122)
# nx.draw(G, pos=nx.circular_layout(G), node_color='r', edge_color='b')

e = [('a', 'b', 0.3), ('b', 'c', 0.9), ('a', 'c', 0.5), ('c', 'd', 1.2)]
G.add_weighted_edges_from(e)
print(nx.dijkstra_path(G, 'a', 'd'))

subax1 = plt.subplot(121)
nx.draw(G)   # default spring_layout