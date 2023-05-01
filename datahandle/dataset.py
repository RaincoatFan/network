import math
import random
import csv

# 地球半径，单位：米
EARTH_RADIUS = 6371000

# 某经纬度范围
LATITUDE_RANGE = (30, 30.02)
LONGITUDE_RANGE = (120, 120.02)

# 中心点个数和最小距离
CENTER_COUNT = 15
MIN_DISTANCE = 500

# 生成点的数量
POINT_COUNT = 500

# 计算两点之间的距离（单位：米）
def calc_distance(lat1, lon1, lat2, lon2):
    rad_lat1 = math.radians(lat1)
    rad_lat2 = math.radians(lat2)
    a = rad_lat1 - rad_lat2
    b = math.radians(lon1) - math.radians(lon2)
    s = 2 * math.asin(math.sqrt(math.pow(math.sin(a/2), 2) + math.cos(rad_lat1) * math.cos(rad_lat2) * math.pow(math.sin(b/2), 2)))
    s *= EARTH_RADIUS
    return s

# 生成随机点，返回经度和纬度
def generate_random_point():
    return (random.uniform(*LATITUDE_RANGE), random.uniform(*LONGITUDE_RANGE))

# 判断两点之间的距离是否小于最小距离
def is_distance_valid(center_points, new_point):
    for point in center_points:
        if calc_distance(*point, *new_point) < MIN_DISTANCE:
            return False
    return True

# 生成中心点
center_points = []
while len(center_points) < CENTER_COUNT:
    new_center = generate_random_point()
    if is_distance_valid(center_points, new_center):
        center_points.append(new_center)

# 生成点并保存到csv文件
with open('points.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['id', 'center', 'x', 'y'])
    for i in range(POINT_COUNT):
        is_center = 1 if i < CENTER_COUNT else 0
        if is_center:
            point = center_points.pop()
        else:
            point = generate_random_point()
            while not is_distance_valid(center_points, point):
                point = generate_random_point()
        writer.writerow([i, is_center, *point])
