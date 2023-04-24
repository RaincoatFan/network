import math
import random
import pandas as pd

# 定义矩形经纬度区域
min_lon, max_lon = -98, -97.92
min_lat, max_lat = 30, 30.5

# 定义函数计算两个经纬度点之间的距离（单位：米）
def haversine(lon1, lat1, lon2, lat2):
    R = 6371000  # 地球平均半径，单位为米
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi / 2) ** 2 + \
        math.cos(phi1) * math.cos(phi2) * \
        math.sin(delta_lambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    distance = R * c
    return distance

# 定义函数随机生成经纬度点
def generate_point():
    lon = random.uniform(min_lon, max_lon)
    lat = random.uniform(min_lat, max_lat)
    return lon, lat

# 定义函数检查距离是否满足要求
def check_distance(center_points, new_point):
    for center in center_points:
        if haversine(center[0], center[1], new_point[0], new_point[1]) < 2000:
            return False
    return True

# 定义函数生成所有点
def generate_all_points(n_center_points, n_points):
    center_points = []
    all_points = []
    for i in range(n_center_points):
        # 随机生成中心点，直到满足要求为止
        while True:
            new_point = generate_point()
            if check_distance(center_points, new_point):
                center_points.append(new_point)
                all_points.append(new_point + (1,))
                break
        # 在中心点周围随机生成若干个点
        for j in range(n_points // n_center_points - 1):
            while True:
                new_point = generate_point()
                if check_distance(center_points, new_point):
                    all_points.append(new_point + (0,))
                    break
    return all_points

# 生成所有点
all_points = generate_all_points(43, 2000)

# 将所有点保存到CSV文件中
df = pd.DataFrame(all_points, columns=['lon', 'lat', 'center'])
df.to_csv('points.csv', index=False)
