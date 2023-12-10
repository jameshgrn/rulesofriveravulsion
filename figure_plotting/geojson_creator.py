#%%
import geopandas as gpd
from shapely.geometry import Point

# con 30.460576, 1.257461 30 steps
# mahajamba 47.395423, -16.382440 75000 steps
# malawi 34.575200, -13.886431 500 steps
# kaz_2011 60.862819, 46.104325 550 steps
# other_afr 37.979975, 6.606345 2500 steps
# ja_26 36.207364, 4.709230 3500 steps
# betsiboka_2004 46.932228, -16.795185 35000 steps
# v11 -71.323425, 7.176474

def create_geojson_from_lat_lon(lat, lon, filename):
    # Create a GeoDataFrame from the latitude and longitude
    gdf = gpd.GeoDataFrame([{'geometry': Point(lon, lat)}], crs="EPSG:4326")

    # Save the GeoDataFrame as a GeoJSON file
    gdf.to_file(filename, driver='GeoJSON')

create_geojson_from_lat_lon(7.176474, -71.323425, "/Users/jakegearon/CursorProjects/rulesofriveravulsion/data/Figure4_data/v11_point.geojson")

# %%
