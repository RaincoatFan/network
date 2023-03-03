import networkx as nx
from networkx.algorithms import approximation as approx

G = nx.DiGraph()
G.add_weighted_edges_from({
    ("A", "B", 3), ("A", "C", 17), ("A", "D", 14), ("B", "A", 3),
    ("B", "C", 12), ("B", "D", 16), ("C", "A", 13),("C", "B", 12),
    ("C", "D", 4), ("D", "A", 14), ("D", "B", 15), ("D", "C", 2)
})

cycle = approx.simulated_annealing_tsp(G, "greedy", source="D")
cost = sum(G[n][nbr]["weight"] for n, nbr in nx.utils.pairwise(cycle))
cycle
['D', 'C', 'B', 'A', 'D']
cost
31
incycle = ["D", "B", "A", "C", "D"]
cycle = approx.simulated_annealing_tsp(G, incycle, source="D")
cost = sum(G[n][nbr]["weight"] for n, nbr in nx.utils.pairwise(cycle))
cycle
['D', 'C', 'B', 'A', 'D']
cost
31