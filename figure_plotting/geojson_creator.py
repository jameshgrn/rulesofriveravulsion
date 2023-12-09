#%%
import geopandas as gpd
from shapely.geometry import Point

#con 30.460576, 1.257461
# 47.395423, -16.382440


def create_geojson_from_lat_lon(lat, lon, filename):
    # Create a GeoDataFrame from the latitude and longitude
    gdf = gpd.GeoDataFrame([{'geometry': Point(lon, lat)}], crs="EPSG:4326")

    # Save the GeoDataFrame as a GeoJSON file
    gdf.to_file(filename, driver='GeoJSON')

create_geojson_from_lat_lon(-16.380180, 47.394254, "/Users/jakegearon/CursorProjects/rulesofriveravulsion/data/Figure4_data/mahajamba_point.geojson")

# %%
