import networkx as nx
import matplotlib.pyplot as plt
import csv

def main():
    G = nx.Graph()
    # 添加对应的边和点
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

    edge = []   #边
    with open('edge-8.txt', 'r', encoding='utf-8') as file:
        for line in file:
            line = tuple(line.replace('\r', '').replace('\n', '').replace('\t', '').split(','))
            edge.append(line)

    distance = []   #距离集合
    for item in range(len(edge)):
        s = pow(pow(x_list[int(edge[item][0])]-x_list[int(edge[item][1])],2)+pow(y_list[int(edge[item][0])]-y_list[int(edge[item][1])],2),0.5)
        s = round(s, 2)
        distance.append(s)

    for item in range(len(edge)):
        G.add_edge(int(edge[item][0]), int(edge[item][1]), name=distance[item])

    #0302
    count = 0
    center_list = []
    for item in range(len(id_list)):
        if sort_tag[item] == '1':
            center_list.append(id_list[item + count])
            id_list.remove(id_list[item + count])
            count -= 1
        print(id_list, center_list)

    a = [[0, 1, 0, 1, 0, 1, 0, 1],[1, 0, 1, 0, 1, 0, 1, 0]]
    b = []
    for item in a:
        print('000',item)
        for center_count in range(len(item)):
            print('111',item[center_count])
            if item[center_count] == 1:
                print('444',center_count)
                b.append(id_list[center_count])
                print('333', b)
        b = []



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

if __name__ == '__main__':
    main()
