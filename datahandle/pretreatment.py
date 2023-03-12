import csv
import random

def handle_center():
    r = csv.reader(open('stops_demo_2.csv'))
    lines = list(r)
    print(lines)

    for item in range(len(lines)):
        number = random.random()
        if number <= 0.01:  #一个配送中心服务约100个点
            lines[item+1][1] = '1'
    writer = csv.writer(open('stops_with_center_3.csv', 'w'))
    writer.writerows(lines)

    # 处理空行
    with open('stops_with_center_3.csv', 'rt') as f:
        lines = ''
        for line in f:
            if line != '\n':
                lines += line
    with open('stops_with_center_3.csv', 'wt') as d:
        d.write(lines)

handle_center()