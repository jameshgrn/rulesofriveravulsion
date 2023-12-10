# README

## Rules of River Avulsion

This repository contains the code and data associated with the manuscript "Rules of River Avulsion". The codebase is primarily written in Python and uses a variety of scientific computing libraries.

### Data Download

The data used in the manuscript is available for download through [Zenodo](https://zenodo.org/records/10338686). The data is zipped and should be downloaded and moved into the `data/avulsion_data` directory. 

### Dependencies

We used Python 3.9.6 to develop the codebase. It is recommended that you use a virtual environment to run the code. We prefer PyEnv. To install PyEnv, run the following command:

```zsh
curl https://pyenv.run | zsh
```

*note: zsh commands can be replaced with bash commands*

The project uses several Python libraries, including but not limited to:

- numpy
- matplotlib
- geopandas
- PyQt5
- scipy
- scikit-image
- rasterio
- shapely
- pysheds
- plotly-express
- pyarrow
- nbformat
- earthengine-api
- geemap
- pycrs
- seaborn
- xgboost
- scikit-learn
- openpyxl
- scikit-optimize

The specific versions of these dependencies are listed in the `pyproject.toml` file. The project uses [Poetry](https://python-poetry.org/) to manage dependencies. To install the dependencies, run the following command:

```zsh
curl -sSL https://install.python-poetry.org | python -
```

This will install Poetry on your system. Once installed, you can use the following command to install the project dependencies:

```zsh
poetry install
```

To activate the virtual environment created by Poetry, use the following command:

```zsh
poetry shell
```

Now, you are ready to run the scripts in this repository.


### Structure

The codebase is organized into several directories:

- `figure_plotting`: Contains scripts for generating figures for the manuscript.
- `data`: Contains the data used in the manuscript. The data is in various formats, including GeoJSON files.
- `based`: Contains the code and data for the BASED model.

### Usage

It is important to run the files in order. A helper zsh script is provided to run the files in order. To run the files, run the following command:

```zsh
poetry run zsh plot_figures.zsh
```

Currently, the BASED model is commented out of the zsh script. To fully train the BASED model, you can uncomment the line in the zsh script. Note that the BASED model may require a separate environment due to the dated dependencies of the scikit-optimize library. However, models, data, and results are provided in the `based` directory.

### Figures

After running the `plot_figures.zsh` script, the plotting elements will be saved in the `figures` directory. Extended data figures are saved in the `figures/extended_data` directory.

Figure 4 is demonstrated interactively. running the command:

```zsh
poetry run python figure_plotting/Figure4_interactive.py
```
will open a GUI that allows you to load a DEM and GeoJSON file and perform a random walk simulation on the DEM. See [the specific README](figure_plotting/Figure4_interactive_README.md) for this figure for more details.

### Cross Section Data
To view cross section data it is first recommended to understand the structure of the data files. After downloading from Zenodo and storing the data files in the `data/avulsion_data` directory, you can run the following command to view the cross section data in *map view*:

```zsh
python cross_section_mapview.py --cross_section_name A002 --cross_section_number 1 --renderer browser --fraction 0.05
```

Here we've supplied the cross section name (A002) and number (1). You can supply 1, 2, or 3 to see the relevant cross sections. You can also supply "val_fabdem" or "val_is2" to the cross_section_number argument to view the cross section data for the validation data. The fraction argument is used to downsample the data for faster rendering. The default is 0.05, which means that 5% of the data will be rendered. The renderer argument can be used to specify the renderer. The default is "browser", which will open a browser window to view the data. The other option is "matplotlib", which will render the data in a matplotlib window.

To view cross section data in *profile view*, run the following command:

```zsh
poetry run python figure_plotting/cross_section_sideview.py --cross_section_name A012 --cross_section_number 1 --renderer browser --atl08_class 1 --fraction 0.05

```

### License

This project is licensed under the MIT License. For more details, see the `LICENSE` file.

### Contact

For any questions or concerns, please contact James (Jake) Gearon at jhgearon@iu.edu.
