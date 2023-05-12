# 为了减少调用高德api的次数，该文件的作用
# 获取所有两点间的实际路面距离，并保存至csv文件中
import csv
from gdmap.distance import main

id_tag = []
sort_tag = []
x_tag = []
y_tag = []
with open('../general/center2.csv', encoding='utf-8') as f:
    for row in csv.reader(f, skipinitialspace=True):
        id_tag.append(row[0])
        sort_tag.append(row[1])
        x_tag.append(row[2])
        y_tag.append(row[3])
    del id_tag[0], sort_tag[0], x_tag[0], y_tag[0]

# center_list = [] #中心点集合
# need_list = [] #需求点集合
# for item in range(sort_tag):
#     print(item)
#     if sort_tag[item] == 1:
#         center_list.append(id_tag[item])
#     else:
#         need_list.append(id_tag[item])

list1 = []
list2 = []
list_distance = []

for i in range(len(id_tag)):
    for j in range(i + 1, len(id_tag)):
        print(i,j)
        warehouse_location = '{},{}'.format(y_tag[i], x_tag[i])
        farm_location = '{},{}'.format(y_tag[j], x_tag[j])
        s = main(warehouse_location, farm_location)
        list1.append(i)
        list2.append(j)
        list_distance.append(s)

data = zip(list1, list2, list_distance)
header = ['first', 'second', 'distance']

with open('../general/center2_distance.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(header)
    writer.writerows(data)
