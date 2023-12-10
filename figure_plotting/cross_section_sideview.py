import pandas as pd
import plotly.express as px
import plotly.io as pio
import sys
import os

def plot_cross_section(foldername, renderer="vscode", atl03_cnf=2, atl08_class=1, UID=None, cross_section_number=1):
    pio.renderers.default = renderer
    
    feather_files = []
    geojson_files = []
    for file in os.listdir(foldername):
        if file.endswith(f"_{cross_section_number}.feather"):
            feather_files.append(os.path.join(foldername, file))
        elif file.endswith(f"_{cross_section_number}.geojson"):
            geojson_files.append(os.path.join(foldername, file))

    cross_sections = feather_files + geojson_files
    
    UIDs = df_sr['UID'].unique()

    if UID is None:
        if len(UIDs) > 1:
            raise ValueError("Multiple UIDs found in file. Please specify a UID.")
        else:
            UID = UIDs[0]
            
    # Select the file based on the cross_section_number
    selected_file = cross_sections[0]
    
    # Load the data from the selected file
    if selected_file.endswith(".feather"):
        temp_df_pre = pd.read_feather(selected_file)
    else:
        temp_df_pre = gpd.read_file(selected_file)
    
    temp_df_pre = temp_df_pre[(temp_df_pre['atl03_cnf'] >= atl03_cnf) & (temp_df_pre['atl08_class'] == atl08_class)]
    temp_df_pre["along_track"] = temp_df_pre["along_track"] - temp_df_pre["along_track"].min()
    
    # Plot the data
    if selected_file.endswith(".geojson"):
        graph = px.line(temp_df_pre, x="along_track", y="height")
    else:
        graph = px.scatter(temp_df_pre, x="along_track", y="height").update_traces(marker={"size":2})
    
    graph.show()

if __name__ == "__main__":
    foldername = sys.argv[1]
    renderer = sys.argv[2] if len(sys.argv) > 2 else "vscode"
    atl03_cnf = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    atl08_class = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    UID = int(sys.argv[5]) if len(sys.argv) > 5 else None
    cross_section_number = int(sys.argv[6]) if len(sys.argv) > 6 else 1

    plot_cross_section(foldername, renderer, atl03_cnf, atl08_class, UID, cross_section_number)
    
# example usage:
# python cross_section_viewer.py /path/to/folder vscode 2 1 3


#%%
    # Load the geojson file
    geojson_file = gpd.read_file('path_to_your_geojson_file.geojson')

    # Plot the data with px.line
    graph = px.line(geojson_file, x="along_track", y="height")
    
    graph.show()
    
#%%
import pandas as pd
import plotly.express as px
import plotly.io as pio
pio.renderers.default = "browser"  # or "notebook", "png", "jpg", "pdf", "svg", etc.

temp_df_pre = pd.read_feather("/Users/jakegearon/CursorProjects/rulesofriveravulsion/data/avulsion_data/V7/V7_UID_9.feather")
#temp_df_pre = temp_df_pre[(temp_df_pre['atl08_class'] >= 0) & (temp_df_pre['atl08_class'] <= 0)]
#temp_df_pre = temp_df_pre[temp_df_pre['atl03_cnf'] >= 1 or temp_df_pre['atl03_cnf'] <= 2]
temp_df_pre["along_track"] = temp_df_pre["along_track"] - temp_df_pre["along_track"].min()
graph = px.scatter(temp_df_pre, x="along_track", y="height").update_traces(marker={"size":2})
graph


# %%
import plotly.io as pio
pio.renderers.default = "browser"


fig = px.scatter_mapbox(temp_df_pre,
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
                    #"https://tiles0.planet.com/basemaps/v1/planet-tiles/global_monthly_%s_%s_mosaic/gmap/{z}/{x}/{y}.png?api_key=722017f6234b4160aa8d26fc6d39fa31" % (year, month)
                    "https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}"
                ]
            }])
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show()
# %%
import pandas as pd
import plotly.express as px
import plotly.io as pio
pio.renderers.default = "browser"

file = '/Users/jakegearon/CursorProjects/rulesofriveravulsion/data/avulsion_data/B12/B12_1_transect_spacing_30.csv'
df = pd.read_csv(file)
graph = px.line(df, x="distance", y="mean")
graph
# %%
