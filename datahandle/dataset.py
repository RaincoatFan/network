import csv
import random
from math import radians, cos, sin, asin, sqrt

# 定义经纬度矩形范围
MIN_LAT, MAX_LAT = radians(30.0), radians(31.0)
MIN_LON, MAX_LON = radians(118.0), radians(119.0)

# 定义中心点个数和最小距离
NUM_CENTERS = 43
MIN_DISTANCE = 2000.0

# 计算两个经纬度坐标之间的距离
def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles
    return c * r * 1000

# 判断两个经纬度坐标之间的距离是否大于最小距离
def is_far_enough(center, centers):
    for c in centers:
        if haversine(center[0], center[1], c[0], c[1]) < MIN_DISTANCE:
            return False
    return True

# 生成中心点坐标
centers = []
while len(centers) < NUM_CENTERS:
    center = [random.uniform(MIN_LON, MAX_LON), random.uniform(MIN_LAT, MAX_LAT)]
    if is_far_enough(center, centers):
        centers.append(center)

# 生成随机点坐标并标注中心点
points = []
for i in range(2000):
    lon, lat = random.uniform(MIN_LON, MAX_LON), random.uniform(MIN_LAT, MAX_LAT)
    if [lon, lat] in centers:
        points.append([lon, lat, 1])
    else:
        points.append([lon, lat, 0])

# 写入csv文件
with open('points.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['lon', 'lat', 'center'])
    for point in points:
        writer.writerow(point)
