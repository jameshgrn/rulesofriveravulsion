import pandas as pd
import plotly.express as px
import plotly.io as pio
import sys
import os


import argparse
import plotly.io as pio
import geopandas as gpd
import plotly.express as px
import glob
import pandas as pd

# Parse command line arguments
parser = argparse.ArgumentParser(description='Process some cross sections.')
parser.add_argument('--cross_section_name', type=str, help='Name of the cross section')
parser.add_argument('--cross_section_number', type=int, choices=[1, 2, 3], help='Cross section number')
parser.add_argument('--renderer', type=str, help='Plotly renderer')
parser.add_argument('--fraction', type=float, help='Subsample fraction')
parser.add_argument('--atl03_cnf', nargs='*', type=int, help='ATL03 confidence level')
parser.add_argument('--atl08_class', nargs='*', type=int, help='ATL08 classification')
args = parser.parse_args()

# Use glob to find the file regardless of its extension
files = glob.glob(f"data/avulsion_data/{args.cross_section_name}/*{args.cross_section_name}*_{args.cross_section_number}.*")

if not files:
    raise ValueError("File not found.")
elif len(files) > 1:
    raise ValueError("Multiple files with the same name but different extensions found. Please ensure there's only one file for each cross section.")

file = files[0]

# Check the file extension and read the file accordingly
if file.endswith('.feather'):
    df_sr = gpd.read_feather(file)
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

    if args.atl08_class is not None:
        df_sr = df_sr[df_sr['atl08_class'].isin(args.atl08_class)]
    if args.atl03_cnf is not None:
        df_sr = df_sr[df_sr['atl03_cnf'].isin(args.atl03_cnf)]

    pio.renderers.default = args.renderer
    graph = px.scatter(df_sr, x="along_track", y="height").update_traces(marker={"size":2})
    # Load figure2 data
    df_fig2 = pd.read_csv('data/figure2_data/fig2_data.csv')
    df_fig2.rename(columns={'Avulsion Name': 'avulsion_name'}, inplace=True)
    # Filter the data for the current cross section
    df_fig2 = df_fig2[df_fig2['avulsion_name'] == args.cross_section_name]
    # Add horizontal lines for are_{i}_m, fpe_{i}_m, and wse_{i}_m
    graph.add_shape(type="line", x0=df_sr['along_track'].min(), x1=df_sr['along_track'].max(), y0=df_fig2[f'are{args.cross_section_number}_m'].values[0], y1=df_fig2[f'are{args.cross_section_number}_m'].values[0], line=dict(color="red"))
    graph.add_shape(type="line", x0=df_sr['along_track'].min(), x1=df_sr['along_track'].max(), y0=df_fig2[f'fpe{args.cross_section_number}_m'].values[0], y1=df_fig2[f'fpe{args.cross_section_number}_m'].values[0], line=dict(color="green"))
    graph.add_shape(type="line", x0=df_sr['along_track'].min(), x1=df_sr['along_track'].max(), y0=df_fig2[f'wse{args.cross_section_number}_m'].values[0], y1=df_fig2[f'wse{args.cross_section_number}_m'].values[0], line=dict(color="blue"))
    graph.show()

if file.endswith('.geojson'):
    #get corresponding file that has the same name plus '_transect_spacing_30.csv' (a different extension as well!)
    df_sr = pd.read_csv(file.replace('.geojson', '_transect_spacing_30.csv'))
    graph = px.line(df_sr, x="distance", y="mean")
    # Load figure2 data
    df_fig2 = pd.read_csv('data/figure2_data/fig2_data.csv')
    df_fig2.rename(columns={'Avulsion Name': 'avulsion_name'}, inplace=True)

    # Filter the data for the current cross section
    df_fig2 = df_fig2[df_fig2['avulsion_name'] == args.cross_section_name]
    # Add horizontal lines for are_{i}_m, fpe_{i}_m, and wse_{i}_m
    graph.add_shape(type="line", x0=df_sr['distance'].min(), x1=df_sr['distance'].max(), y0=df_fig2[f'are{args.cross_section_number}_m'].values[0], y1=df_fig2[f'are{args.cross_section_number}_m'].values[0], line=dict(color="red"))
    graph.add_shape(type="line", x0=df_sr['distance'].min(), x1=df_sr['distance'].max(), y0=df_fig2[f'fpe{args.cross_section_number}_m'].values[0], y1=df_fig2[f'fpe{args.cross_section_number}_m'].values[0], line=dict(color="green"))
    graph.add_shape(type="line", x0=df_sr['distance'].min(), x1=df_sr['distance'].max(), y0=df_fig2[f'wse{args.cross_section_number}_m'].values[0], y1=df_fig2[f'wse{args.cross_section_number}_m'].values[0], line=dict(color="blue"))
    graph.show()
    

elif file.endswith('.gpkg'):
    df_sr = gpd.read_file(file, driver='GPKG')
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

    if args.atl08_class is not None:
        df_sr = df_sr[df_sr['atl08_class'].isin(args.atl08_class)]
    if args.atl03_cnf is not None:
        df_sr = df_sr[df_sr['atl03_cnf'].isin(args.atl03_cnf)]

    pio.renderers.default = args.renderer
    graph = px.scatter(df_sr, x="along_track", y="height").update_traces(marker={"size":2})
    # Load figure2 data
    df_fig2 = pd.read_csv('data/figure2_data/fig2_data.csv')
    df_fig2.rename(columns={'Avulsion Name': 'avulsion_name'}, inplace=True)

    # Filter the data for the current cross section
    df_fig2 = df_fig2[df_fig2['avulsion_name'] == args.cross_section_name]
    graph.add_shape(type="line", x0=df_sr['along_track'].min(), x1=df_sr['along_track'].max(), y0=df_fig2[f'are{args.cross_section_number}_m'].values[0], y1=df_fig2[f'are{args.cross_section_number}_m'].values[0], line=dict(color="red"))
    graph.add_shape(type="line", x0=df_sr['along_track'].min(), x1=df_sr['along_track'].max(), y0=df_fig2[f'fpe{args.cross_section_number}_m'].values[0], y1=df_fig2[f'fpe{args.cross_section_number}_m'].values[0], line=dict(color="green"))
    graph.add_shape(type="line", x0=df_sr['along_track'].min(), x1=df_sr['along_track'].max(), y0=df_fig2[f'wse{args.cross_section_number}_m'].values[0], y1=df_fig2[f'wse{args.cross_section_number}_m'].values[0], line=dict(color="blue"))
    graph.show()
    
