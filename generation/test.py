from gdmap.distance import main

warehouse_location = '{},{}'.format(113.505562,22.230699)
farm_location = '{},{}'.format(113.511648,22.233317)

# # 仓库经纬度（格式：经度,纬度）
# warehouse_location = '113.296467,22.209200'
#
# # 农田经纬度（格式：经度,纬度）,换成自己的
# farm_location = '113.294467,22.207200'

# print(warehouse_location)


s = main(warehouse_location,farm_location)

print(s)