import numpy as np
from scipy.special import softmax as scipy_softmax
from dem_ops import pixel_to_coordinate, coordinate_to_pixel

def precompute_neighbors(dims):
    """Precomputes neighbors for all pixels in a given dimension."""
    neighbors = {}
    max_x, max_y = dims

    for x in range(max_x):
        for y in range(max_y):
            coord = (x, y)
            neighbors[coord] = [
                (x-1, y), (x+1, y), (x, y-1), (x, y+1),
                (x-1, y-1), (x+1, y-1), (x-1, y+1), (x+1, y+1)
            ]

            # Ensure neighbors are within valid dimensions
            neighbors[coord] = [
                (nx, ny) for nx, ny in neighbors[coord]
                if 0 <= nx < max_x and 0 <= ny < max_y
            ]

    return neighbors

def calculate_slope(elevation, current, neighbor, no_data_value):
    current_elevation = elevation[current]
    neighbor_elevation = elevation[neighbor]

    if np.ma.is_masked(current_elevation) or np.ma.is_masked(neighbor_elevation):
        return np.NINF

    if current_elevation == no_data_value or neighbor_elevation == no_data_value:
        return np.NINF

    dx = abs(current[0] - neighbor[0])
    dy = abs(current[1] - neighbor[1])

    # using precalculated values for distances
    distance = np.sqrt(2) if dx == dy else 1

    rise = current_elevation - neighbor_elevation

    if np.isnan(rise):
        return np.NINF

    slope = rise / distance

    if np.isinf(slope):
        return np.copysign(np.inf, slope)

    return slope


def walk(elevation, crs = None, transform = None, seed = None,
     steps = 0, theta = 0, alpha = 0, beta = 0,
     neighbors_dict = None, no_data_value = 9999):
    if neighbors_dict is None:
        neighbors_dict = precompute_neighbors(elevation.shape)

    if seed is None:
        raise ValueError("Seed value is required!")

    if not np.ma.is_masked(elevation):
        elevation = np.ma.masked_where(elevation == no_data_value, elevation)

    visit_frequency = np.zeros_like(elevation)

    if transform:  # If transform is provided
        x, y = coordinate_to_pixel(transform, *seed)
    else:  # If just a plain numpy array is given
        x, y = seed

    if x < 0 or x >= elevation.shape[0] or y < 0 or y >= elevation.shape[1]:
        print("Seed value:", seed)
        if transform:
            print("Transformed to pixel:", coordinate_to_pixel(transform, *seed))
        raise ValueError(f"Invalid starting coordinates: ({x}, {y})")

    path = [pixel_to_coordinate(transform, x, y) if transform else (x, y)]
    last_position = None
    actual_steps = 0
    visited_elevations = []
    last_vector = np.array([0, 0])  # Initially no direction, so set to a zero vector
    using_transform = bool(transform)

    for _ in range(steps):
        neighbors = neighbors_dict[(x, y)]

        valid_neighbors = [(nx, ny) for nx, ny in neighbors if
                           0 <= nx < elevation.shape[0] and 0 <= ny < elevation.shape[1] and (nx, ny) != last_position]
        neighbor_slopes = [calculate_slope(elevation, (x, y), (nx, ny), no_data_value) for nx, ny in valid_neighbors]

        momentum_weights = []
        for nx, ny in valid_neighbors:
            direction = np.array([nx - x, ny - y])
            normalized_last_vector = last_vector / np.linalg.norm(last_vector) if np.linalg.norm(
                last_vector) > 0 else last_vector
            normalized_direction = direction / np.linalg.norm(direction)
            cosine_similarity = np.dot(normalized_last_vector, normalized_direction)
            momentum_weight = alpha * (1 + cosine_similarity) / 2
            momentum_weights.append(momentum_weight)

        slope_weights = [theta * slope for slope in neighbor_slopes]

        # Combine the weights directly without applying softmax to them individually
        combined_weights_raw = [(1 - beta) * m + beta * s for m, s in zip(momentum_weights, slope_weights)]

        # Apply softmax to the combined weights
        combined_weights = scipy_softmax(combined_weights_raw)

        if np.isnan(np.sum(combined_weights)):
            combined_weights = [1 / len(valid_neighbors) for _ in valid_neighbors]
        else:
            combined_weights = np.divide(combined_weights, np.sum(combined_weights))
        index = np.random.choice(len(valid_neighbors), p = combined_weights)
        new_position = valid_neighbors[index]

        new_x, new_y = new_position
        last_vector = np.array([new_x - x, new_y - y])  # Update the direction vector

        if not np.isscalar(elevation.mask) and elevation.mask[new_x, new_y]:
            continue

        visit_frequency[new_x, new_y] += 1
        last_position = (x, y)
        x, y = new_position
        if using_transform:
            path.append(pixel_to_coordinate(transform, *new_position))
        else:
            path.append(new_position)
        visited_elevations.append(elevation[new_x, new_y])
        actual_steps += 1

    return path, crs if crs else None, visit_frequency, visited_elevations
