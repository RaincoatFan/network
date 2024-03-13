import random
import math
import numpy as np
import matplotlib.pyplot as plt

def read_tsp(path):
    lines = open(path, 'r').readlines()
    assert 'NODE_COORD_SECTION\n' in lines
    index = lines.index('NODE_COORD_SECTION\n')
    data = lines[index + 1:-1]
    tmp = []
    for line in data:
        line = line.strip().split(' ')
        if line[0] == 'EOF':
            continue
        tmpline = []
        for x in line:
            if x == '':
                continue
            else:
                tmpline.append(float(x))
        if tmpline == []:
            continue
        tmp.append(tmpline)
    data = tmp
    return data

data = read_tsp('D1')

data = np.array(data)
data = data[:, 1:]
show_data = np.vstack([data, data[0]])

# print(show_data)

distance = np.zeros((100, 100))

for i in range(100):
    for j in range(100):
        dis = math.sqrt(math.pow((show_data[i][0]-show_data[j][0]),2)+math.pow((show_data[i][1]-show_data[j][1]),2))
        # print(distance[i][j])
        distance[i][j] = dis

print(distance)

