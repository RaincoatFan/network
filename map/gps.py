import folium
import pandas as pd

# read the CSV file into a pandas dataframe
df = pd.read_csv('Texasdemo.csv')

# create a map centered on the first location in the dataframe
my_map = folium.Map(location=[df.iloc[0]['latitude'], df.iloc[0]['longitude']], zoom_start=10)

# iterate over each row in the dataframe and add a marker to the map
for index, row in df.iterrows():
    # create an icon from a local image file
    if int(row['center']) == 0:
        icon_path = 'point.png'
        icon = folium.features.CustomIcon(icon_path, icon_size=(20, 20))

        # create a marker with the icon and other parameters
        marker = folium.Marker(location=[row['latitude'], row['longitude']], icon=icon)
        # add the marker to the map
        marker.add_to(my_map)
    else:
        icon_path = 'star.png'
        icon = folium.features.CustomIcon(icon_path, icon_size=(40, 40))

        # create a marker with the icon and other parameters
        marker = folium.Marker(location=[row['latitude'], row['longitude']], icon=icon)
        # add the marker to the map
        marker.add_to(my_map)

my_map.add_child(folium.LatLngPopup())
# save the map to an HTML file
my_map.save('my_map.html')
