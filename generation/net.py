import networkx as nx
import matplotlib.pyplot as plt
import csv
from networkx.algorithms import approximation as approx
import numpy as np

def generation(x):
    G = nx.Graph()

    # 从csv文件中读取节点id，类别，坐标
    id_tag = []
    sort_tag = []
    x_tag = []
    y_tag = []
    with open('stops.csv', encoding='utf-8') as f:
        for row in csv.reader(f, skipinitialspace=True):
            id_tag.append(row[0])
            sort_tag.append(row[1])
            x_tag.append(row[2])
            y_tag.append(row[3])
        del id_tag[0], sort_tag[0], x_tag[0], y_tag[0]

    id_list = []    #点
    x_list = [] #x坐标
    y_list = [] #y坐标
    for item in id_tag:
        id_list.append(int(item))
    for item in x_tag:
        x_list.append(int(item))
    for item in y_tag:
        y_list.append(int(item))

    coor_list = []  #将x,y组合成坐标
    for item in range(len(x_list)):
        coor = (x_list[item],y_list[item])
        coor_list.append(coor)

    for item in id_list:
        G.add_node(item, desc=str(item))

    # 分离配送中心 与 需求节点
    # 具体操作：将配送中心节点 至 center_list
    count = 0
    center_list = []    #配送中心节点集合
    for item in range(len(id_list)):
        if sort_tag[item] == '1':
            center_list.append(id_list[item + count])
            id_list.remove(id_list[item + count])
            count -= 1

    # 种群
    a = np.zeros((2, 8))
    for i in range(len(x)):
        a[int(x[i]), i] = 1

    row_count = 0
    for row in center_list:
        for column in id_list:
            s = pow(pow(x_list[row] - x_list[column], 2) + pow(y_list[row] - y_list[column], 2), 0.5)
            s = round(s, 2)
            if s <= 5:
                a[:, id_list.index(column)] = 0
                a[row_count][id_list.index(column)] = 1
        row_count += 1


    # a = [[0, 0, 0, 0, 0, 0, 0, 0],[1, 1, 1, 1, 1, 1, 1, 1]]
    b = []
    c = []
    num = 0
    for item in a:
        for id_count in range(len(item)):
            if item[id_count] == 1:
                b.append(id_list[id_count])
        b.append(center_list[num])
        c.append(b)
        b = []
        num += 1
    print('aaaaaaaaaaaaa',a)

    # 处理边（1）所有属于一个配送中心的点之间都有边
    # 先放入 edge.txt 中
    with open('edge.txt', 'w', encoding='utf-8') as file:
        for item in range(len(c)):
            # print('item',b[item])
            for k in range(len(c[item])):
                for j in range(k+1, len(c[item])):
                    # print('j',b[j])
                    file.writelines(str(c[item][k])+','+str(c[item][j])+'\n')
    file.close()

    # 处理边（2）将边放入 edge 集合中，用于network的边集
    edge = []   #边
    with open('edge.txt', 'r', encoding='utf-8') as file:
        for line in file:
            line = tuple(line.replace('\r', '').replace('\n', '').replace('\t', '').split(','))
            edge.append(line)

    # 处理距离 计算所有边的距离，放入 distance 集合，用于network的标签
    distance = []   #距离集合
    for item in range(len(edge)):
        s = pow(pow(x_list[int(edge[item][0])]-x_list[int(edge[item][1])],2)+pow(y_list[int(edge[item][0])]-y_list[int(edge[item][1])],2),0.5)
        s = round(s, 2)
        distance.append(s)

    # G加入 边 标签
    for item in range(len(edge)):
        G.add_edge(int(edge[item][0]), int(edge[item][1]), name=distance[item])

    # # 画图
    # pos = coor_list
    # # 按pos所定位置画出节点,无标签无权值
    # nx.draw_networkx(G, pos, with_labels=None)
    # # 画出标签
    # node_labels = nx.get_node_attributes(G, 'desc')
    # nx.draw_networkx_labels(G, pos, labels=node_labels)
    # # 画出边权值
    # edge_labels = nx.get_edge_attributes(G, 'name')
    # nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    # plt.title('Net', fontsize=10)
    # plt.show()

    # 计算所有子图的最短路径
    res_tsp = []    # 目标函数 sum(res_tsp)
    for center_item in center_list:
        # 提取子图，并计算子图 tsp-sa 花销
        G1 = nx.Graph()
        edlist = G.edges(center_item)
        subgraph_node_list = []
        subgraph_node_list.append(center_item)
        # print('subgraph_node_list',subgraph_node_list)
        for item, spot in edlist:
            subgraph_node_list.append(spot)

        with open('subgraph_edge.txt', 'w', encoding='utf-8') as file:
            for item in range(len(subgraph_node_list)):
                for k in range(item+1, len(subgraph_node_list)):
                    file.writelines(str(subgraph_node_list[item])+','+str(subgraph_node_list[k])+'\n')

        subgraph_edge_list = []

        with open('subgraph_edge.txt', 'r', encoding='utf-8') as file:
            for line in file:
                line = tuple(line.replace('\r', '').replace('\n', '').replace('\t', '').split(','))
                subgraph_edge_list.append(line)

        # 处理子图距离 计算所有边的距离，放入 distance 集合，用于network的标签
        subgraph_distance = []   #距离集合
        for item in range(len(subgraph_edge_list)):
            s = pow(pow(x_list[int(subgraph_edge_list[item][0])]-x_list[int(subgraph_edge_list[item][1])],2)+pow(y_list[int(subgraph_edge_list[item][0])]-y_list[int(subgraph_edge_list[item][1])],2),0.5)
            s = round(s, 2)
            subgraph_distance.append(s)

        subgraph_weight_node_edge_from = set()
        for item in range(len(subgraph_edge_list)):
            subgraph_weight_node_edge_from.add((str(subgraph_edge_list[item][0]),str(subgraph_edge_list[item][1]),subgraph_distance[item]))

        # 创建子图
        G1.add_weighted_edges_from(subgraph_weight_node_edge_from)

        # 用sa计算最短路径
        if len(edlist) == 0:
            cost = 0
        else:
            cycle = approx.simulated_annealing_tsp(G1, "greedy", source=str(center_item))
            cost = sum(G1[n][nbr]["weight"] for n, nbr in nx.utils.pairwise(cycle))
        res_tsp.append(cost)
    print('res_tsp',res_tsp)
    print('sum_cost', sum(res_tsp))
    return sum(res_tsp)
# # 执行
# generation([0, 0, 0, 0, 1, 0, 0, 1])