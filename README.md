# README

## Rules of River Avulsion

This repository contains the code and data associated with the manuscript "Rules of River Avulsion". The codebase is primarily written in Python and uses a variety of scientific computing libraries.

### Dependencies

We used Python 3.9.6 to develop the codebase.

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

The specific versions of these dependencies are listed in the `pyproject.toml` file. The project uses [Poetry](https://python-poetry.org/) to manage dependencies. To install the dependencies, run the following command:

```bash 
poetry install
```


### Structure

The codebase is organized into several directories:

- `figure_plotting`: Contains scripts for generating figures for the manuscript. For example, `Figure4_interactive.py` is a PyQt5 application for interactive exploration of the data.
- `data`: Contains the data used in the manuscript. The data is in various formats, including GeoJSON files.

### Usage

To run the interactive figure plotting script, navigate to the `figure_plotting` directory and run the script with Python:

```bash
poetry run plot_figures.py
```

Figure 4 is an interactive figure that allows you to load a DEM and GeoJSON file and perform a random walk simulation on the DEM. See the [the speceific readme](figure_plotting/Figure4_interactive_README.md) 


### License

This project is licensed under the MIT License. For more details, see the `LICENSE` file.

### Contact

For any questions or concerns, please contact James (Jake) Gearon at jhgearon@iu.edu.
