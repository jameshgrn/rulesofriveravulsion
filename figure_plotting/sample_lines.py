import os
import ee
import glob
import geemap
import pandas as pd
import geopandas as gpd
import numpy as np
import json
from pyproj import Transformer

ee.Initialize()
fabdem = ee.ImageCollection("projects/sat-io/open-datasets/FABDEM").mosaic().setDefaultProjection('EPSG:4326', None, 30)
elevation = fabdem

for root, dirs, files in os.walk("data/avulsion_data"):
    dirs[:] = [d for d in dirs if d not in ['old']]
    for file in files:
        if file.endswith(('_1.geojson', '_2.geojson', '_3.geojson')):
            line = os.path.join(root, file)
            
            transect_file = line.replace('.geojson', '_transect.csv')
                
            gdf = gpd.read_file(line).to_crs("EPSG:4326")
            fc = geemap.gdf_to_ee(gdf)
            line_geojson = ee.Feature(fc.first())

            print(line_geojson.getInfo())
            leng = line_geojson.length(maxError=1).getInfo()
            print(elevation.projection().getInfo())

            denominator = 30
            max_attempts = 50
            attempt = 0

            while attempt < max_attempts:
                try:
                    n_segments = np.floor(leng/denominator)
                    print(f"leng: {leng}, denominator: {denominator}, n_segments: {n_segments}")
                    transect = geemap.extract_transect(elevation, line_geojson.geometry(), n_segments=n_segments, reducer='mean', to_pandas=True)
                    if 'distance' in transect.columns and 'mean' in transect.columns:
                        break
                except Exception as e:
                    print(f"Failed to process file: {file}. Error: {str(e)}")
                    import traceback
                    print(traceback.format_exc())
                denominator += 5
                attempt += 1

            if attempt == max_attempts:
                print(f"Failed to process file: {file} after {max_attempts} attempts.")
            else:
                transect_file = transect_file.replace('.csv', f'_spacing_{denominator}.csv')
                transect.to_csv(transect_file, index=False)

