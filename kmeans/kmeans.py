import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
import networkx as nx
from haversine import haversine
from networkx.algorithms import approximation as approx

# 计算欧拉距离
def calcDis(dataSet, centroids, k):
    clalist = []
    for data in dataSet:
        diff = np.tile(data, (k,
                              1)) - centroids  # 相减   (np.tile(a,(2,1))就是把a先沿x轴复制1倍，即没有复制，仍然是 [0,1,2]。 再把结果沿y方向复制2倍得到array([[0,1,2],[0,1,2]]))
        squaredDiff = diff ** 2  # 平方
        squaredDist = np.sum(squaredDiff, axis=1)  # 和  (axis=1表示行)
        distance = squaredDist ** 0.5  # 开根号
        clalist.append(distance)
    clalist = np.array(clalist)  # 返回一个每个点到质点的距离len(dateSet)*k的数组
    return clalist


# 计算质心
def classify(dataSet, centroids, k):
    # 计算样本到质心的距离
    clalist = calcDis(dataSet, centroids, k)
    # 分组并计算新的质心
    minDistIndices = np.argmin(clalist, axis=1)  # axis=1 表示求出每行的最小值的下标
    newCentroids = pd.DataFrame(dataSet).groupby(
        minDistIndices).mean()  # DataFramte(dataSet)对DataSet分组，groupby(min)按照min进行统计分类，mean()对分类结果求均值
    newCentroids = newCentroids.values

    # 计算变化量
    changed = newCentroids - centroids

    return changed, newCentroids


# 使用k-means分类
def kmeans(dataSet, k):
    # 随机取质心
    centroids = random.sample(dataSet, k)

    # 更新质心 直到变化量全为0
    changed, newCentroids = classify(dataSet, centroids, k)
    while np.any(changed != 0):
        changed, newCentroids = classify(dataSet, newCentroids, k)
    print('newCentroids',newCentroids)
    centroids = sorted(newCentroids.tolist())  # tolist()将矩阵转换成列表 sorted()排序

    # 根据质心计算每个集群
    cluster = []
    clalist = calcDis(dataSet, centroids, k)  # 调用欧拉距离
    minDistIndices = np.argmin(clalist, axis=1)
    for i in range(k):
        cluster.append([])
    for i, j in enumerate(minDistIndices):  # enymerate()可同时遍历索引和遍历元素
        cluster[j].append(dataSet[i])

    return centroids, cluster


# 创建数据集
def createDataSet():
    id_tag = []
    x_tag = []
    y_tag = []
    with open('../datahandle/points.csv', encoding='utf-8') as f:
        # with open('stops.csv', encoding='utf-8') as f:
        for row in csv.reader(f, skipinitialspace=True):
            id_tag.append(row[0])
            x_tag.append(row[2])
            y_tag.append(row[3])
        del id_tag[0], x_tag[0], y_tag[0]
    id_list = []    #点
    x_list = [] #x坐标
    y_list = [] #y坐标
    for item in id_tag:
        id_list.append(int(item))
    for item in x_tag:
        x_list.append(float(item))
    for item in y_tag:
        y_list.append(float(item))
    set_stops = []
    for item in range(len(x_list)):
        set_stops.append([x_list[item],y_list[item]])
    print('set_stops',set_stops)
    return set_stops

def cost_tsp(x):
    with open('test.txt', 'w', encoding='utf-8') as file:
        for item in range(len(x)):
            for k in range(item + 1, len(x)):
                file.writelines(str(item) + ',' + str(k) + '\n')

    subgraph_edge_list = []
    with open('test.txt', 'r', encoding='utf-8') as file:
        for line in file:
            line = tuple(line.replace('\r', '').replace('\n', '').replace('\t', '').split(','))
            subgraph_edge_list.append(line)

    subgraph_distance = []  # 距离集合
    for item in range(len(subgraph_edge_list)):
        s = haversine((x[int(subgraph_edge_list[item][0])][1], x[int(subgraph_edge_list[item][0])][0]),
                      (x[int(subgraph_edge_list[item][1])][1], x[int(subgraph_edge_list[item][1])][0]))
        # s = pow(pow(x_list[int(subgraph_edge_list[item][0])]-x_list[int(subgraph_edge_list[item][1])],2)+pow(y_list[int(subgraph_edge_list[item][0])]-y_list[int(subgraph_edge_list[item][1])],2),0.5)
        # s = round(s, 2)
        subgraph_distance.append(s)
    # print('distance',subgraph_distance)

    subgraph_weight_node_edge_from = []
    for item in range(len(subgraph_edge_list)):
        subgraph_weight_node_edge_from.append(
            (str(subgraph_edge_list[item][0]), str(subgraph_edge_list[item][1]), subgraph_distance[item]))
    G = nx.Graph()
    # 创建子图
    res_tsp = []
    G.add_weighted_edges_from(subgraph_weight_node_edge_from)
    if len(x) == 0:
        cost = 0
    else:
        cycle = approx.simulated_annealing_tsp(G, "greedy", source=str(0))
    sum = 0
    for item in range(len(cycle)):
        if item+1 >= len(cycle):
            break
        else:
            s = haversine((x[int(cycle[item])][1], x[int(cycle[item])][0]),
                      (x[int(cycle[item+1])][1], x[int(cycle[item+1])][0]))
            sum += s
        # cost = sum(G[n][nbr]["weight"] for n, nbr in nx.utils.pairwise(cycle))
    # res_tsp.append(cost)
    # print('sum(res_tsp)',sum(res_tsp))
    return sum


if __name__ == '__main__':
    dataset = createDataSet()
    centroids, cluster = kmeans(dataset, 43)
    print('质心为：%s' % centroids)
    print('集群为：%s' % cluster)

    # print('集群0',cluster[0])
    sumall = 0
    all = 0
    sum_arr = []
    for item in range(43):
        coos =len(cluster[item])
        all += coos
        sum_arr.append(cost_tsp(cluster[item]))
        sumall += float(cost_tsp(cluster[item]))
    print('sum',sumall)
    print('集群len', all)
    c1 = []
    c2 = []
    center = []
    min = 999
    for item in range(43):
        for j in range(len(cluster[item])):
            s = haversine((centroids[item][1], centroids[item][0]),
                      (cluster[item][j][1], cluster[item][j][0]))
            if s < min:
                min = s
                min = 999
                k = j
        c1.append(cluster[item][k][0])
        c2.append(cluster[item][k][1])
        print('aaaaaaaaa',item,k)
        print('sssssssssssaaa', cluster[item][k])
        center.append(cluster[item][k])
    mincenter = 999
    minlist = []
    for item in range(43):
        for k in range(43):
            if item == k:
                break
            else:
                s = haversine((center[item][1], center[item][0]),
                          (center[k][1], center[k][0]))
                if s < mincenter:
                    mincenter = s
        minlist.append(mincenter)
        mincenter = 999
    print('minlist',minlist)
    df = pd.DataFrame({'column1': c1, 'column2': c2})
    df.to_csv("test1.csv", index=False)
    variance = np.var(sum_arr)
    print('sum_arr',sum(sum_arr))
    print('cost数组',sum_arr)
    print('方差',variance)
    # for item in range(43):
    #     print('质心',centroids[item])
    #     c1.append(centroids[item][0])
    #     c2.append(centroids[item][1])



    for i in range(len(dataset)):
        plt.scatter(dataset[i][0], dataset[i][1], marker='o', color='green', s=40, label='原始点')
        #  记号形状       颜色      点的大小      设置标签
    for j in range(len(centroids)):
        plt.scatter(centroids[j][0], centroids[j][1], marker='x', color='red', s=50, label='质心')
    plt.show()
