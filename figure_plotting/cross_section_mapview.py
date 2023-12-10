import argparse
import plotly.io as pio
import geopandas as gpd
import plotly.express as px

# Parse command line arguments
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--filename', type=str, help='Path to the feather file')
parser.add_argument('--renderer', type=str, help='Plotly renderer')
parser.add_argument('--fraction', type=float, help='Subsample fraction')
args = parser.parse_args()

df_sr = gpd.read_feather(args.filename)

df_sr['lat'] = df_sr.geometry.y  # lat col
df_sr['lon'] = df_sr.geometry.x  # lon col
# let's add a unique identifier (UID) and calculate along-track distances
min_d = df_sr.segment_dist.min()
df_sr['UID'] = df_sr.groupby(['rgt', 'cycle', 'track', 'pair']).ngroup().add(1)
df_sr['along_track'] = (((df_sr["segment_dist"] + df_sr["x_atc"])) - min_d) - \
                       (((df_sr["segment_dist"] + df_sr["x_atc"])) - min_d)[0]

df_sr = df_sr.reset_index()
pio.renderers.default = args.renderer
subsample = df_sr.groupby("UID").sample(frac=args.fraction)

fig = px.scatter_mapbox(subsample,
                        lat="lat",
                        lon="lon",
                        hover_name="height",
                        color="height",
                        hover_data=["height", "lat", "lon", "along_track", "UID", "time"],
                        height=500)
fig.update_layout(
        mapbox_style="white-bg",
        mapbox_layers=[
            {
                "below": "traces",
                "sourcetype": "raster",
                "sourceattribution": "Planet",
                "source": [
                    "https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}"
                ]
            }])
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show()

# example usage:
# python cross_
# section_viewer.py --filename /path/to/file.feather --renderer browser --fraction 0.05