import csv
import random
import pandas as pd

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
    data = pd.read_csv("stops_with_center_3.csv")
    res = data.dropna(how="all")
    res.to_csv("stops_with_center_3.csv", index=False)

handle_center()