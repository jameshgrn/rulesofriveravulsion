# Figure4_interactive.py

This script is a PyQt5 application for visualizing and interacting with Digital Elevation Models (DEMs) and GeoJSON files. It also includes a random walk simulation on the loaded DEM.

## Main Features

1. **Load DEM**: This button allows you to load a Digital Elevation Model (DEM) in .tif format. The DEM is displayed as a grayscale image.

2. **Load GeoJSON**: This button allows you to load a GeoJSON file. The GeoJSON data is clipped to the extent of the loaded DEM and displayed on top of it.

3. **Load Lat/Lon**: This button allows you to load a GeoJSON file containing a single Point geometry. The point's coordinates are used as the starting point for the random walk.

4. **Start Random Walk**: This button initiates a random walk simulation from the selected starting point. The parameters for the random walk (theta, alpha, beta, number of steps, number of trials, threshold percentage, and sigma) can be adjusted in the provided input fields.

5. **Save Plot**: This button allows you to save the current plot as a .png or .jpeg image.

6. **Reset**: This button clears the current selection and resets the plot.

7. **Save Random Walker Cloud**: This button allows you to save the result of the random walk simulation as a GeoTIFF file.

## Usage

To run the script, you need to have Python installed along with the necessary libraries (PyQt5, numpy, matplotlib, geopandas, rasterio, etc.). You can run the script using the following command:


Once the application is running, you can use the provided buttons to load a DEM, load a GeoJSON file, start a random walk, and save your results.

```bash
bash
python Figure4interactive.py
```

## Note

This script assumes that the DEM and the GeoJSON files are in the same Coordinate Reference System (CRS). If they are not, you may need to reproject one of them to match the other.
