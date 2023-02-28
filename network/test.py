import csv

import networkx as nx
import matplotlib.pyplot as plt
import csv
import numpy as np

# with open('stops.csv', 'rb') as stopList:
#     reader = csv.reader(stopList)
#     id_tag = [row['id'] for row in reader]
#     sort = [row['center'] for row in reader]
#     print(reader)

id_tag = []
sort_tag = []
with open('stops.csv', encoding='utf-8-sig') as f:
    for row in csv.reader(f, skipinitialspace=True):
        id_tag.append(row[0])
        sort_tag.append(row[1])
    del id_tag[0], sort_tag[0]

id_list = []
for item in id_tag:
    id_list.append(int(item))
print(id_list)

sort_list = []
for item in sort_tag:
    sort_list.append(int(item))
print(sort_list)

node_0 = []
node_1 = []
node_color_r = []
node_shape_0 = []
node_color_g = []
node_shape_1 = []
node_color = []
node_shape = []

for i in range(len(sort_list)):
    if int(sort_list[i]) == 0:
        color = 'r'
        shape = 'o'
        node_0.append(i)
        node_color_r.append(color)
        node_shape_0.append(shape)
    else:
        color = 'g'
        shape = '*'
        node_1.append(i)
        node_color_g.append(color)
        node_shape_1.append(shape)

    node_color.append(color)
    node_shape.append(shape)
    print(node_color)
    print(node_shape)

edge = []
with open('edge-8.txt', 'r', encoding='utf-8') as file:
    # data = fedge.readline()
    for line in file:
        line = tuple(line.replace('\r','').replace('\n','').replace('\t','').split(','))
        edge.append(line)
    print(edge)

edge_color = []
edge_style = []

for item in edge:
    # if int(sort_list[int(item[0])]) == 0 or int(sort_list[int(item[1])]) == 0:
    #     color = 'r'
    # else:
    #     color = 'g'
    color = 'r'
    edge_color.append(color)

G = nx.Graph()


coordinates = [[1, 2], [2, 2], [3, 2], [3, 1], [2, 1], [1, 1], [1, 4], [3, 4], [4, 1], [4, 2]]
vnode= np.array(coordinates)
npos = dict(zip(id_list, vnode))

# G.add_node('111')
G.add_edges_from(edge)

# nx.draw(G, pos=nx.random_layout(G), node_color=node_color, node_size=10, node_shape='o', edge_color=edge_color,
#                  width=0.3, style='solid', font_size=8)
# nx.draw(G)

# nx.draw(G,pos = nx.random_layout(G),node_color = 'g',edge_color = '#000079',with_labels = True, font_size =10,node_size =200,node_shape='o',width=0.3, style='solid')
nx.draw(G, npos, node_size=50, node_color="#6CB6FF")  # 绘制节点
nx.draw(G, npos, edge)  # 绘制边

plt.show()







#
# G = nx.Graph()
#
# # subax2 = plt.subplot(122)
# # nx.draw(G, pos=nx.circular_layout(G), node_color='r', edge_color='b')
#
# # e = [('a', 'b', 0.3), ('b', 'c', 0.9), ('a', 'c', 0.5), ('c', 'd', 1.2)]
#
# elist = [('a', 'b'), ('b', 'c'), ('a', 'c'), ('c', 'd'),('d', 'e'),('c', 'e')]
#
#
#
# weight = [0.3, 0.9, 0.5, 1.2, 0.6, 0.8]
#
# G.add_edges_from(elist)
#
# # G.add_weighted_edges_from(e)
# # print(nx.dijkstra_path(G, 'a', 'd'))
# path = dict(nx.all_pairs_bellman_ford_path(G, weight='weight'))
#
# print(path['a']['e'])
#
# nx.draw(G)
#
# plt.show()
