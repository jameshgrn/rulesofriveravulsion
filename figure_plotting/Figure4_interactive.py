import sys
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QLabel, QFileDialog, QLineEdit, QHBoxLayout, QSlider, QErrorMessage, QComboBox
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from dem_ops import load_elevation, pixel_to_coordinate, coordinate_to_pixel, func_hillshade
from walk_funcs import walk, precompute_neighbors
from scipy.ndimage import gaussian_filter
from skimage.morphology import remove_small_objects
import rasterio
from rasterio.features import rasterize
from shapely.geometry import box

class App(QMainWindow):
    def __init__(self, lat=None, long=None):
        super().__init__()
        self.cmap = 'gist_gray'
        self.title = 'Random DEM Walker'
        self.left = 800
        self.top = 600
        self.width = 800
        self.height =1000
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.theta_input = QLineEdit("1")
        self.selected_points = []
        self.current_marker = None
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        layout = QHBoxLayout()  # Change to QHBoxLayout to place inputs and buttons side by side
        self.central_widget.setLayout(layout)

        # Create a vertical layout for the inputs and buttons
        input_layout = QVBoxLayout()
        layout.addLayout(input_layout)

        # Add the input fields
        self.theta_input = QLineEdit('10')
        self.alpha_input = QLineEdit('1')
        self.beta_input = QLineEdit('0.5')
        self.steps_input = QLineEdit('200')
        input_layout.addWidget(QLabel("Theta:"))
        input_layout.addWidget(self.theta_input)
        input_layout.addWidget(QLabel("Alpha:"))
        input_layout.addWidget(self.alpha_input)
        input_layout.addWidget(QLabel("Beta:"))
        input_layout.addWidget(self.beta_input)
        input_layout.addWidget(QLabel("Steps:"))
        input_layout.addWidget(self.steps_input)
        self.num_trials_input = QLineEdit('100')
        input_layout.addWidget(QLabel("Num Trials:"))
        input_layout.addWidget(self.num_trials_input)
        self.vmin_slider = QSlider(Qt.Horizontal, self)
        self.vmin_slider.setEnabled(False)
        self.vmin_slider.valueChanged.connect(self.on_vmin_slider_change)
        self.vmin_value_label = QLabel("Min elevation (vmin): 0", self)
        self.vmax_slider = QSlider(Qt.Horizontal, self)
        self.vmax_slider.setEnabled(False)
        self.vmax_slider.valueChanged.connect(self.on_vmax_slider_change)
        self.vmax_value_label = QLabel("Max elevation (vmax): 0", self)
        self.apply_button = QPushButton('Apply changes', self)
        self.apply_button.clicked.connect(self.apply_elevation_threshold)
        self.elevation_label = QLabel("Elevation: N/A")
        input_layout.addWidget(self.elevation_label)
        input_layout.addWidget(self.vmin_value_label)
        input_layout.addWidget(self.vmin_slider)
        input_layout.addWidget(self.vmax_value_label)
        input_layout.addWidget(self.vmax_slider)
        input_layout.addWidget(self.apply_button)
        # Add the "Load Lat/Lon" button to the vertical layout
        load_lat_lon_button = QPushButton('Load Lat/Lon')
        load_lat_lon_button.clicked.connect(self.load_lat_lon)
        input_layout.addWidget(load_lat_lon_button)
        

        # Add the "Load DEM" button to the vertical layout
        button = QPushButton('Load DEM')
        button.clicked.connect(self.load_dem)
        input_layout.addWidget(button)

        # Add the "Load GeoJSON" button to the vertical layout
        load_geojson_button = QPushButton('Load GeoJSON')
        load_geojson_button.clicked.connect(self.load_geojson)
        input_layout.addWidget(load_geojson_button)

        # Add the QLineEdit widgets for pct_over and sigma
        self.pct_over_input = QLineEdit('0.075')
        self.sigma_input = QLineEdit('2')
        input_layout.addWidget(QLabel("Threshold Percentage (pct_over):"))
        input_layout.addWidget(self.pct_over_input)
        input_layout.addWidget(QLabel("Sigma (sigma):"))
        input_layout.addWidget(self.sigma_input)

        # Add the "Start Random Walk" button at the bottom
        start_walk_button = QPushButton('Start Random Walk')
        start_walk_button.clicked.connect(self.initiate_random_walk)
        input_layout.addWidget(start_walk_button)

        # Add the "Save Plot" button at the bottom
        save_plot_button = QPushButton('Save Plot')
        save_plot_button.clicked.connect(self.save_plot)
        input_layout.addWidget(save_plot_button)

        # Add the "Reset" button at the bottom
        reset_button = QPushButton('Reset')
        reset_button.clicked.connect(self.reset_selection)
        input_layout.addWidget(reset_button)

        # Add the "Save Random Walker Cloud" button at the bottom
        save_cloud_button = QPushButton('Save Random Walker Cloud')
        save_cloud_button.clicked.connect(self.save_random_walker_cloud)
        input_layout.addWidget(save_cloud_button)

        # Create the plot area
        self.figure = Figure(figsize=(10, 10))
        self.canvas = FigureCanvas(self.figure)
        self.canvas.mpl_connect('button_press_event', self.onclick)
        self.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        self.label = QLabel("Select a point")
        input_layout.addWidget(self.label)
        layout.addWidget(self.canvas)
        self.ax = self.figure.add_subplot(111)

        # Initialize the average visit frequency attribute
        self.average_visit_frequency = None

        # Initialize lat and long if provided
        self.lat = lat
        self.long = long
        
    def save_plot(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getSaveFileName(
            self, "Save Plot", "", "PNG Files (*.png);;JPEG Files (*.jpg *.jpeg);;All Files (*)", options=options
        )
        if file_name:
            self.figure.savefig(file_name)

    def on_vmin_slider_change(self, value):
        self.vmin_value_label.setText(f"Min elevation (vmin): {value}")

    def on_vmax_slider_change(self, value):
        self.vmax_value_label.setText(f"Max elevation (vmax): {value}")

    def apply_elevation_threshold(self):
        vmin_value = self.vmin_slider.value()
        vmax_value = self.vmax_slider.value()

        # Update the colormap of the existing image
        self.ax.images[0].set_clim(vmin=vmin_value, vmax=vmax_value)

        # Redraw the canvas to reflect the changes
        self.canvas.draw()

    def on_mouse_move(self, event):
        if event.inaxes is not None:
            x, y = event.xdata, event.ydata
            col, row = coordinate_to_pixel(self.transform, x, y)
            if 0 <= row < self.elevation.shape[0] and 0 <= col < self.elevation.shape[1]:
                elevation = self.elevation[row, col]
                self.elevation_label.setText(f"Elevation: {elevation:.2f}")
            else:
                self.elevation_label.setText("Elevation: Out of bounds")
        else:
            self.elevation_label.setText("Elevation: N/A")

    def load_dem(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(
            self, "Select DEM file", "", "DEM Files (*.tif);;All Files (*)", options=options
        )
        if fileName:
            self.elevation, self.crs, self.transform = load_elevation(fileName)
            self.hillshade = func_hillshade(self.elevation)
            print(self.transform, self.crs)
            min_elevation = int(np.min(self.elevation))
            max_elevation = int(np.max(self.elevation))
            self.vmin_slider.setRange(min_elevation, max_elevation)
            self.vmax_slider.setRange(min_elevation, max_elevation)
            self.vmin_slider.setEnabled(True)
            self.vmax_slider.setEnabled(True)
            self.vmin_slider.setValue(min_elevation)
            self.vmax_slider.setValue(max_elevation)
            self.vmin_slider.valueChanged.connect(self.on_vmin_slider_change)
            self.vmax_slider.valueChanged.connect(self.on_vmax_slider_change)
            self.on_vmin_slider_change(min_elevation)
            self.on_vmax_slider_change(max_elevation)
            self.average_visited_frequency = None
            self.ax.imshow(
                self.hillshade, cmap=self.cmap,
                extent=[self.transform[2], self.transform[2] + self.transform[0] * self.elevation.shape[1],
                        self.transform[5] + self.transform[4] * self.elevation.shape[0], self.transform[5]]
            )
            self.canvas.draw()

            # If lat and long are provided, initiate a click at that location
            if self.lat is not None and self.long is not None:
                self.onclick(self.lat, self.long)

    def load_geojson(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(
            self, "Select GeoJSON file", "", "GeoJSON Files (*.geojson);;All Files (*)", options=options
        )
        if fileName:
            try:
                original_vmin, original_vmax = self.vmin_slider.value(), self.vmax_slider.value()

                # Load the GeoJSON file
                gdf = gpd.read_file(fileName)

                # Check if the GeoDataFrame is empty
                if gdf.empty:
                    raise ValueError("The GeoJSON file is empty.")

                # Check if the DEM data is loaded before proceeding
                if not hasattr(self, 'elevation') or self.elevation is None:
                    raise ValueError("DEM data is not loaded. Please load a DEM before clipping.")

                # Extract the bounds of the DEM
                dem_bounds = [
                    self.transform[2],  # left
                    self.transform[5] + self.transform[4] * self.elevation.shape[0],  # bottom
                    self.transform[2] + self.transform[0] * self.elevation.shape[1],  # right
                    self.transform[5]  # top
                ]

                # Create a Polygon from bounds
                
                dem_extent_box = box(*dem_bounds)

                # The coordinate system of the loaded GeoJSON should match the DEM's.
                # If not, you should re-project (transform) the GeoJSON to the DEM's CRS.
                gdf = gdf.to_crs(self.crs)  # self.crs is the CRS of your DEM

                # Clip the data with the polygon
                clipped_gdf = gdf.clip(dem_extent_box)
                self.geojson_data = clipped_gdf

                # Convert the clipped GeoJSON to a raster mask
                raster_mask = rasterize([(geom, 1) for geom in self.geojson_data.geometry],
                                        out_shape=(self.elevation.shape[0], self.elevation.shape[1]),
                                        transform=self.transform,
                                        all_touched=True)

                # Multiply the raster mask with a large elevation value and add it to the DEM
                large_elevation_value = 1000
                self.elevation = self.elevation + large_elevation_value * raster_mask

                # Clear the current axes and re-draw the DEM and the clipped GeoJSON
                self.ax.clear()
                self.ax.imshow(
                    self.hillshade, cmap=self.cmap,
                    extent=[self.transform[2], self.transform[2] + self.transform[0] * self.elevation.shape[1],
                            self.transform[5] + self.transform[4] * self.elevation.shape[0], self.transform[5]]
                )

                # Update the vmin and vmax values
                self.vmin_slider.setRange(np.min(self.elevation), np.max(self.elevation))
                self.vmax_slider.setRange(np.min(self.elevation), np.max(self.elevation))
                self.vmin_slider.setValue(np.min(self.elevation))
                self.vmax_slider.setValue(np.max(self.elevation))
                self.on_vmin_slider_change(np.min(self.elevation))
                self.on_vmax_slider_change(np.max(self.elevation))

                # Redraw the canvas to reflect the changes
                self.canvas.draw()
                # Plotting the clipped GeoJSON data
                self.geojson_data.plot(ax=self.ax, color='#4FA0CA', edgecolor='k', linewidth=0.75)  # You can change color and other properties
                
                if hasattr(self, 'starting_point') and self.starting_point is not None:
                    self.ax.plot(*self.starting_point, 'ro', markersize=5)

                # Redraw the canvas after the updates
                self.canvas.draw()

            except Exception as e:
                error_dialog = QErrorMessage()
                error_dialog.showMessage(f"An error occurred while loading and clipping the GeoJSON file: {str(e)}")
                error_dialog.exec_()
    
    def load_lat_lon(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(
            self, "Select GeoJSON file", "", "GeoJSON Files (*.geojson);;All Files (*)", options=options
        )
        if fileName:
            try:
                # Load the GeoJSON file
                gdf = gpd.read_file(fileName)

                # Check if the GeoDataFrame is empty
                if gdf.empty:
                    raise ValueError("The GeoJSON file is empty.")

                # Convert the GeoDataFrame to the same CRS as the DEM
                gdf = gdf.to_crs(self.crs)
                
                # Extract the first point's coordinates
                first_point = gdf.geometry[0]
                if first_point.type == 'Point':
                    self.lat, self.long = first_point.y, first_point.x
                else:
                    raise ValueError("The GeoJSON file does not contain Point geometry.")

                # Convert the geographic coordinates to pixel coordinates
                self.starting_point = (self.long, self.lat)
                
                                # Clear the current axes and re-draw the DEM and the clipped GeoJSON
                self.ax.clear()
                self.ax.imshow(
                    self.hillshade, cmap=self.cmap,
                    extent=[self.transform[2], self.transform[2] + self.transform[0] * self.elevation.shape[1],
                            self.transform[5] + self.transform[4] * self.elevation.shape[0], self.transform[5]]
                )
                
                    # After plotting the lat/long data, check if GeoJSON data exists and plot it
                if hasattr(self, 'geojson_data') and self.geojson_data is not None:
                    self.geojson_data.plot(ax=self.ax, color='#4FA0CA', edgecolor='k', linewidth=0.75)


                # Plotting the clipped GeoJSON data
                gdf.plot(ax=self.ax, color='#4FA0CA', edgecolor='k', linewidth=0.75)  # You can change color and other properties

                # Redraw the canvas after the updates
                self.canvas.draw()
                

            except Exception as e:
                error_dialog = QErrorMessage()
                error_dialog.showMessage(f"An error occurred while loading the GeoJSON file: {str(e)}")
                error_dialog.exec_()

    def onclick(self, event):
        if event.xdata is not None and event.ydata is not None:
            ix, iy = event.xdata, event.ydata
            print(f"Clicked at ({ix:.2f}, {iy:.2f})")

            # If a marker already exists, remove it before placing a new one
            if self.current_marker:
                self.current_marker.remove()
                self.selected_points = []  # Clear previous selection since we allow only one point

            self.selected_points.append((ix, iy))
            self.label.setText(f"Selected point: ({ix:.6f}, {iy:.6f}).\n Press 'd' to save, 'u' to undo.")
            self.current_marker, = self.ax.plot(ix, iy, 'ro', markersize=5)  # Place the new marker
            self.canvas.draw()

    def reset_selection(self):
        if self.current_marker:  # Check if a marker exists
            self.current_marker.remove()  # Remove the current marker
            self.current_marker = None  # Reset the current marker
            self.selected_points = []  # Clear the selection list

        # Clear any random walk paths and reset the average visit frequency
        self.ax.clear()  # This clears the entire plot area
        self.average_visit_frequency = None  # Reset the visit frequency data

        # If you have other elements like images, you would need to redraw them here
        # For example, if you have a background image or DEM, redraw it here
        if self.elevation is not None and self.transform is not None:
            self.ax.imshow(
                self.hillshade, cmap=self.cmap,
                extent=[self.transform[2], self.transform[2] + self.transform[0] * self.elevation.shape[1],
                        self.transform[5] + self.transform[4] * self.elevation.shape[0], self.transform[5]]
            )

        self.label.setText("Selection reset. Select a new point.")
        self.canvas.draw()  # Redraw the canvas to reflect changes

    def initiate_random_walk(self):
        if hasattr(self, 'starting_point') and self.starting_point is not None:
            last_point = self.starting_point
            print(f"Point saved: {last_point}")
            try:
                theta = float(self.theta_input.text())
                alpha = float(self.alpha_input.text())
                beta = float(self.beta_input.text())
                steps = int(self.steps_input.text())
                num_trials = int(self.num_trials_input.text())
                pct_over = float(self.pct_over_input.text())
                sigma = float(self.sigma_input.text())
            except ValueError:
                self.label.setText("Invalid inputs for theta, alpha, beta, or number of trials.")
                return
            self.start_random_walk(last_point, theta, alpha, beta, steps=steps, num_trials=num_trials, pct_over=pct_over, sigma=sigma)
        else:
            self.label.setText("No point selected to initiate walk!")

    def start_random_walk(self, seed, theta, alpha, beta, steps, num_trials, pct_over, sigma):
        pixel_coords = (seed[0], seed[1])
        print(f"Starting random walk from pixel coordinates: {pixel_coords}")
        neighbors_dict = precompute_neighbors(self.elevation.shape)
        visited_pixels_raster = np.zeros_like(self.elevation)
        total_visit_frequency = np.zeros_like(self.elevation)
        final_num_trials = num_trials
        for i in range(final_num_trials):
            _, _, visit_frequency, _ = walk(
                self.elevation, self.crs, self.transform, pixel_coords, steps, theta, alpha, beta, neighbors_dict
            )
            visited_pixels_raster[visit_frequency > 0] = 1
            total_visit_frequency += visit_frequency
        self.average_visited_frequency = total_visit_frequency / final_num_trials
        self.plot_paths_on_dem(self.elevation, self.transform, self.average_visited_frequency, pct_over, sigma)

    def plot_paths_on_dem(self, dem, transform, average_visit_frequency, pct_over, sigma):
        # self.current_marker.set_markersize(3)  # Resize the marker at the beginning of the random walk
        # self.canvas.draw()
        print(np.min(average_visit_frequency), np.max(average_visit_frequency))

        norm = plt.Normalize(vmin=np.min(average_visit_frequency), vmax=np.max(average_visit_frequency))
        normalized_data = norm(average_visit_frequency)
        rgba_data = plt.get_cmap('Reds')(normalized_data)
        print(np.min(rgba_data[..., :3]), np.max(rgba_data[..., :3]))  # RGB channels
        print(np.min(rgba_data[..., 3]), np.max(rgba_data[..., 3]))  # Alpha channel

        rgba_data[..., 3] = np.where(average_visit_frequency > 0, 1, 0)
        print(np.unique(rgba_data[..., 3]))


        smoothed_mask = gaussian_filter(average_visit_frequency, sigma=sigma)

        # Apply a threshold to remove small objects
        min_object_size = 2  # Adjust the minimum object size as needed
        cleaned_mask = remove_small_objects(smoothed_mask > pct_over, min_size=min_object_size)

        # Update the alpha channel of the RGBA data based on the cleaned mask
        rgba_data[..., 3] = np.where(cleaned_mask > pct_over, 1, 0)
        print(np.min(rgba_data[..., :3]), np.max(rgba_data[..., :3]))
        print(np.min(rgba_data[..., 3]), np.max(rgba_data[..., 3]))
        self.average_visit_frequency = rgba_data

        self.ax.imshow(
            rgba_data, extent=[transform[2], transform[2] + transform[0] * dem.shape[1],
                               transform[5] + transform[4] * dem.shape[0], transform[5]]
        )

        if hasattr(self, 'geojson_data') and self.geojson_data is not None:
            self.geojson_data.plot(ax=self.ax, facecolor="#4FA0CA", edgecolor='k')  # Makes borders red and transparent
        # self.ax.legend(loc='upper right')
        self.canvas.draw()

    def save_random_walker_cloud(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getSaveFileName(
            self, "Save Random Walker Cloud", "", "GeoTIFF Files (*.tif);;All Files (*)", options=options
        )
        if file_name:
            try:
                # Assuming `self.average_visit_frequency` is your RGBA data in uint8 format
                rgba_data_uint8 = (self.average_visit_frequency * 255).astype(np.uint8)

                print(f"Shape of elevation: {self.elevation.shape}")
                print(f"Dtype of elevation: {self.elevation.dtype}")
                print(f"Shape of rgba_data_uint8: {rgba_data_uint8.shape}")
                print(f"Dtype of rgba_data_uint8: {rgba_data_uint8.dtype}")
                print(f"CRS: {self.crs}")
                print(f"Transform: {self.transform}")

                color_interps = [
                    rasterio.enums.ColorInterp.red,
                    rasterio.enums.ColorInterp.green,
                    rasterio.enums.ColorInterp.blue,
                    rasterio.enums.ColorInterp.alpha,
                ]

                with rasterio.open(file_name, 'w', driver='GTiff', height=rgba_data_uint8.shape[0],
                                    width=rgba_data_uint8.shape[1], count=4, dtype='uint8',
                                    crs=self.crs, transform=self.transform) as dst:
                    for band in range(4):
                        dst.write(rgba_data_uint8[:, :, band], band+1)
                    dst.colorinterp = color_interps


            except Exception as e:
                error_dialog = QErrorMessage()
                error_dialog.showMessage(f"An error occurred while saving the Random Walker Cloud: {str(e)}")
                error_dialog.exec_()


app = QApplication(sys.argv)
ex = App()
ex.resize(1200, 800)  # Set the initial size of the window
ex.figure.set_size_inches(10, 8)  # Set the size of the plot area
ex.show()
sys.exit(app.exec_())

