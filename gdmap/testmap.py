import requests
from OpenSSL import SSL
import numpy as np
import folium
from folium import plugins

# ##########################################全局变量#############################################################
# 替换为你的高德API密钥,在高德开放平台里注册-》应用管理-》创建新应用-》服务平##台那里选择web服务-》记住生成的key，这里没有密钥的。后续填写到amap_key中
amap_key = '3e4b14b34c1eb4db37d8fd72f69573af'
list_latlon=[]
Lon = []
Lat = []

# 仓库经纬度（格式：经度,纬度）
warehouse_location = '113.549090,22.198951'

# 农田经纬度（格式：经度,纬度）,换成自己的
farm_location = '113.547,22.19199'

# 路径规划策略，可以选择0~5的整数，具体含义可以参考高德API文档 具体介绍见：https://lbs.amap.com/api/webservice/guide/api/direction
# 本文用的是驾车路径，
# 0，速度优先，此路线不一定距离最短
# 1，费用优先，不走收费路段，且耗时最少的路线
# 2，距离优先，仅走距离最短的路线，但是可能存在穿越小路/小区的情况
# 3，速度优先，不走快速路，例如京通快速路（因为策略迭代，建议使用13）
# 4，躲避拥堵，但是可能会存在绕路的情况，耗时可能较长
# 5，多策略（同时使用速度优先、费用优先、距离优先三个策略计算路径）。
# 其中必须说明，就算使用三个策略算路，会根据路况不固定的返回一~三条路径规划信息。
# 6，速度优先，不走高速，但是不排除走其余收费路段
# 7，费用优先，不走高速且避免所有收费路段
# 8，躲避拥堵和收费，可能存在走高速的情况，并且考虑路况不走拥堵路线，但有可能存在绕路和时间较长
# 9，躲避拥堵和收费，不走高速
strategy = 2
# #######################################################################################################



# 获取高德地图的路径规划点
def get_route(start, end, mode, amap_key):
    # 这里的url中选择是步行，公交还是驾车路径，本文中driving?表示驾车，具体介绍见：https://lbs.amap.com/api/webservice/guide/api/direction
    url = f'https://restapi.amap.com/v3/direction/driving?origin={start}&destination={end}&strategy={mode}&key={amap_key}'
    response = requests.get(url)
    data = response.json()

    if data['status'] == '1':
        route = data['route']['paths'][0]['steps']
        return route
    else:
        print('请求失败，请检查输入参数。')
        return None


route = get_route(warehouse_location, farm_location, strategy, amap_key)

if route:
    for i, step in enumerate(route):
        list_latlon.append(step["polyline"])
        print(f'步骤 {i+1}: {step["instruction"]}:{step["polyline"]}')

else:
    print('无法获取路线规划。')

# 获取街道地图
for item in list_latlon:
    points = item.split(';')
    for point in points:
        coords = point.split(',')
        Lon.append(float(coords[0]))
        Lat.append(float(coords[1]))


# 绘制地图
def PlotLineOnMap(Lat, Lon):
    # 给出的坐标系为GCJ-02，如果需要测试google地图，需要进行坐标转换

    tri = np.array(list(zip(Lat, Lon)))

    san_map = folium.Map(
        location=[23.243466, 113.637713],
        zoom_start=16,
        # 高德街道图
        # tiles='http://webrd02.is.autonavi.com/appmaptile?lang=zh_cn&size=1&scale=1&style=7&x={x}&y={y}&z={z}',
        tiles='http://webst02.is.autonavi.com/appmaptile?style=6&x={x}&y={y}&z={z}', # 高德卫星图
        attr='default')

    folium.PolyLine(tri, color='#3388ff').add_to(san_map)
    marker_cluster = plugins.MarkerCluster().add_to(san_map)
    for lat, lon in zip(Lat, Lon):
        folium.Marker([lat, lon], color='red').add_to(marker_cluster)
    san_map.save('showpoint.html')

def main():

    PlotLineOnMap(Lat, Lon)

if __name__ == '__main__':
    main()

