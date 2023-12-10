import argparse
import plotly.io as pio
import geopandas as gpd
import plotly.express as px
import glob
import pandas as pd

# Parse command line arguments
parser = argparse.ArgumentParser(description='Process some cross sections.')
parser.add_argument('--cross_section_name', type=str, help='Name of the cross section')
parser.add_argument('--cross_section_number', type=str, help='Cross section number')
parser.add_argument('--renderer', type=str, help='Plotly renderer')
parser.add_argument('--fraction', type=float, help='Subsample fraction')
args = parser.parse_args()

# Use glob to find the file regardless of its extension
if args.cross_section_number == 'val_fabdem':
    files = glob.glob(f"data/avulsion_data/{args.cross_section_name}/*val*transect*.csv")
elif args.cross_section_number == 'val_is2':
    files = glob.glob(f"data/avulsion_data/{args.cross_section_name}/*val*.feather")
    if not files:
        files = glob.glob(f"data/avulsion_data/{args.cross_section_name}/*val*.gpkg")
else:
    files = glob.glob(f"data/avulsion_data/{args.cross_section_name}/*{args.cross_section_name}*_{args.cross_section_number}.*")

if not files:
    raise ValueError("File not found.")
elif len(files) > 1:
    raise ValueError("Multiple files with the same name but different extensions found. Please ensure there's only one file for each cross section.")

file = files[0]

# Check the file extension and read the file accordingly
if file.endswith('.feather') or file.endswith('.gpkg'):
    df_sr = gpd.read_feather(file) if file.endswith('.feather') else gpd.read_file(file, driver='GPKG')
    print(df_sr.columns)
    print(df_sr.head())
    #if gpkg or feather, do this:
    df_sr['lat'] = df_sr.geometry.y  # lat col
    df_sr['lon'] = df_sr.geometry.x  # lon col
    # let's add a unique identifier (UID) and calculate along-track distances
    min_d = df_sr.segment_dist.min()
    df_sr['UID'] = df_sr.groupby(['rgt', 'cycle', 'track', 'pair']).ngroup().add(1)
    df_sr = df_sr.reset_index()
    if 'x_atc' in df_sr.columns:
        try:
            df_sr['along_track'] = (((df_sr["segment_dist"] + df_sr["x_atc"])) - min_d) - \
                                (((df_sr["segment_dist"] + df_sr["x_atc"])) - min_d)[0] if not df_sr.empty else 0
        except KeyError:
            df_sr['along_track'] = (((df_sr["segment_dist"] + df_sr["distance"])) - min_d) - \
                                (((df_sr["segment_dist"] + df_sr["distance"])) - min_d)[0] if not df_sr.empty else 0
    else:
        df_sr['along_track'] = (((df_sr["segment_dist"] + df_sr["distance"])) - min_d) - \
                            (((df_sr["segment_dist"] + df_sr["distance"])) - min_d)[0] if not df_sr.empty else 0

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
                    "sourceattribution": "google",
                    "source": [
                        "https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}"
                    ]
                }])
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    fig.show()

elif file.endswith('.csv'):
    df_sr = pd.read_csv(file)
    fig = px.line_mapbox(df_sr, lat="lat", lon="lon", height=500)
    fig.update_layout(
        mapbox_style="white-bg",
        mapbox_layers=[
            {
                "below": "traces",
                "sourcetype": "raster",
                "sourceattribution": "google",
                "source": [
                    "https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}"
                ]
            }])
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    fig.show()

else:
    raise ValueError("Unsupported file format. Please use .feather, .geojson, .gpkg, or .csv.")


# example usage:
# python cross_section_mapview.py --cross_section_name A002 --cross_section_number val_fabdem --renderer browser --fraction 0.05
# python cross_section_mapview.py --cross_section_name A002 --cross_section_number val_is2 --renderer browser --fraction 0.05
# python cross_section_mapview.py --cross_section_name A002 --cross_section_number 1 --renderer browser --fraction 0.05

