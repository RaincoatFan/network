import networkx as nx
import matplotlib.pyplot as plt
import csv

def generation():
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
        print(id_list, center_list)

    # 种群
    a = [[0, 1, 0, 1, 0, 1, 0, 1],[1, 0, 1, 0, 1, 0, 1, 0]]
    b = []
    c = []
    num = 0
    for item in a:
        print('000',item)
        for id_count in range(len(item)):
            print('111',item[id_count])
            if item[id_count] == 1:
                print('222',id_count)
                b.append(id_list[id_count])
        b.append(center_list[num])
        c.append(b)
        b = []
        num += 1

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
    print('edge',edge)

    # 处理距离 计算所有边的距离，放入 distance 集合，用于network的标签
    distance = []   #距离集合
    for item in range(len(edge)):
        s = pow(pow(x_list[int(edge[item][0])]-x_list[int(edge[item][1])],2)+pow(y_list[int(edge[item][0])]-y_list[int(edge[item][1])],2),0.5)
        s = round(s, 2)
        distance.append(s)

    # G加入 边 标签
    for item in range(len(edge)):
        G.add_edge(int(edge[item][0]), int(edge[item][1]), name=distance[item])

    #画图
    pos = coor_list
    # 按pos所定位置画出节点,无标签无权值
    nx.draw_networkx(G, pos, with_labels=None)
    # 画出标签
    node_labels = nx.get_node_attributes(G, 'desc')
    nx.draw_networkx_labels(G, pos, labels=node_labels)
    # 画出边权值
    edge_labels = nx.get_edge_attributes(G, 'name')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    plt.title('Net', fontsize=10)
    plt.show()

generation()