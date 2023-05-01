from gdmap.distance import main

warehouse_location = '{},{}'.format(120.01534548663,30.0102869508384)
farm_location = '{},{}'.format(120.015232228827,30.0194616070962)

# # 仓库经纬度（格式：经度,纬度）
# warehouse_location = '113.296467,22.209200'
#
# # 农田经纬度（格式：经度,纬度）,换成自己的
# farm_location = '113.294467,22.207200'

# print(warehouse_location)


s = main(warehouse_location,farm_location)

print(s)