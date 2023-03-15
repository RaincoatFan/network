from haversine import haversine

a = haversine((30.384303,-97.964343),(30.361568,-97.987454))
print(a)
# import networkx as nx
# from networkx.algorithms import approximation as approx
#
# G = nx.DiGraph()
# G.add_weighted_edges_from({
#     ("A", "B", 3), ("A", "C", 17), ("A", "D", 14), ("B", "A", 3),
#     ("B", "C", 12), ("B", "D", 16), ("C", "A", 13),("C", "B", 12),
#     ("C", "D", 4), ("D", "A", 14), ("D", "B", 15), ("D", "C", 2)
# })
#
# cycle = approx.simulated_annealing_tsp(G, "greedy", source="D")
# cost = sum(G[n][nbr]["weight"] for n, nbr in nx.utils.pairwise(cycle))
# # cycle
# # ['D', 'C', 'B', 'A', 'D']
# # cost
# # 31
# print('cycle',cycle)
# print('cost',cost)
# incycle = ["D", "B", "A", "C", "D"]
# cycle = approx.simulated_annealing_tsp(G, incycle, source="D")
# cost = sum(G[n][nbr]["weight"] for n, nbr in nx.utils.pairwise(cycle))
# print('cycle',cycle)
# print('cost',cost)
# cycle
# ['D', 'C', 'B', 'A', 'D']
# cost
# 31

# import numpy as np
# import matplotlib.pyplot as plt
# train_x = np.array([50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70]).reshape((1, -1))
# train_y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]).reshape((1, -1))
# plt.figure()
# plt.plot(train_x,train_y)
# plt.xlabel("Iteration")
# plt.ylabel("Costs")
# plt.show()

#
# def main():
#     G = nx.DiGraph()
#     # 添加对应的边和点
#     for i in range(1, 10):
#         G.add_node(i, desc='v'+str(i))  # 结点名称不能为str,desc为标签即结点名称
#     G.add_edge(1, 2, name='6')          # 添加边， 参数name为边权值
#     G.add_edge(1, 3, name='4')
#     G.add_edge(1, 4, name='5')
#     G.add_edge(2, 5, name='1')
#     G.add_edge(3, 5, name='1')
#     G.add_edge(4, 6, name='2')
#     G.add_edge(5, 7, name='9')
#     G.add_edge(5, 8, name='7')
#     G.add_edge(6, 8, name='4')
#     G.add_edge(7, 9, name='2')
#     G.add_edge(8, 9, name='4')
#
#     pos = [(1, 3), (1, 3), (2, 4), (2, 2),  (2, 1),  (3, 3),  (4, 1),  (5, 4),  (5, 2),  (6, 3)]  # pos列表从第0位开始，但我定义是从结点1开始，这里令前两个坐标相同
#     # 按pos所定位置画出节点,无标签无权值
#     nx.draw_networkx(G, pos, with_labels=None)
#     # 画出标签
#     node_labels = nx.get_node_attributes(G, 'desc')
#     nx.draw_networkx_labels(G, pos, labels=node_labels)
#     # 画出边权值
#     edge_labels = nx.get_edge_attributes(G, 'name')
#     nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
#
#     plt.title('AOE_CPM', fontsize=10)
#     plt.show()
#
#
# if __name__ == '__main__':
#     main()
