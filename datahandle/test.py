import csv
import math
from gdmap.distance import main

def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    r = 6371  # Radius of earth in kilometers. Use 3956 for miles
    return c * r

with open('ttt.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)  # Skip header row
    points = [(float(row[0]), float(row[1])) for row in reader]

distances = []
for i in range(len(points)):
    for j in range(i + 1, len(points)):
        warehouse_location = '{},{}'.format(points[i][1], points[i][0])
        farm_location = '{},{}'.format(points[j][1], points[j][0])
        print(warehouse_location,farm_location)
        s = main(warehouse_location, farm_location)
        distances.append(s)

print(distances)
