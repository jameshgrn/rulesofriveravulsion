import numpy as np
from pysheds.grid import Grid

def pixel_to_coordinate(transform, x, y):
    if transform is None:
        return x, y
    return transform * (x, y)


def coordinate_to_pixel(transform, x, y):
    if transform is None:
        return x, y
    col, row = ~transform * (x, y)
    return round(row), round(col)


def load_elevation(data):
    if isinstance(data, str):  # If the input is a file path
        grid = Grid.from_raster(data, data_name='dem')
        dem = grid.read_raster(data)
        pit_filled_dem = grid.fill_pits(dem)
        flooded_dem = grid.fill_depressions(pit_filled_dem)
        inflated_dem = grid.resolve_flats(flooded_dem)
        crs = grid.crs
        transform = grid.affine
        return inflated_dem, crs, transform
    else:
        raise ValueError("Unsupported data format. Must be a raster")
    

def func_hillshade(arr, azimuths=[315], altitudes=[45], combine='mean'):
    """
    Create hillshade from a numpy array containing elevation data.
    
    Parameters:
    - arr: 2D numpy array containing elevation data
    - azimuths: list of azimuth angles in degrees (default to [315])
    - altitudes: list of altitude angles in degrees (default to [45])
    - combine: method to combine multi-angle hillshades ('mean' or 'max')
    
    Returns:
    - hillshade: 2D numpy array with hillshade values
    """
    
    if arr.ndim != 2:
        raise ValueError("Input array should be two-dimensional")
    
    # Calculate gradients
    x, y = np.gradient(arr)
    
    # Calculate slope and aspect
    slope = np.pi / 2.0 - np.arctan(np.sqrt(x**2 + y**2))
    aspect = np.arctan2(-x, y)
    
    hillshades = []
    
    for azimuth, altitude in [(az, alt) for az in azimuths for alt in altitudes]:
        # Convert degrees to radians and constrain values
        azimuth = np.clip(360.0 - azimuth, 0, 360) * np.pi / 180.0
        altitude = np.clip(altitude, 0, 90) * np.pi / 180.0
        
        # Calculate hillshade
        shaded = np.sin(altitude) * np.sin(slope) + np.cos(altitude) * np.cos(slope) * np.cos(azimuth - np.pi / 2.0 - aspect)
        hillshades.append((shaded + 1) / 2)
    
    # Combine multi-angle hillshades
    if combine == 'mean':
        hillshade = np.mean(hillshades, axis=0)
    elif combine == 'max':
        hillshade = np.max(hillshades, axis=0)
    else:
        raise ValueError("Invalid 'combine' method. Use 'mean' or 'max'")
    
    return (hillshade * 255).astype(np.uint8)

# Example usage with multiple azimuths and altitudes
# shade = hillshade(squeezed_dem, azimuths=[0, 90, 180, 270], altitudes=[30, 60])
