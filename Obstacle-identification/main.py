from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from os import path
from osgeo import gdal

def flood_fill(array: np.ndarray, index: list[int]) -> np.ndarray:
    """Collect all connected cells using a flood fill algorithm
    
    Parameters
    ----------
    array : np.ndarray
        Obstacle array
    index : list[int]
        Index of starting point for obstacle
        
    Returns
    -------
    np.ndarray:
        Array of indices of cells connected to the original"""
    
    next_indices = [index]
    all_indices = []

    offsets = [[0, 1], [0, -1], [1, 0], [-1, 0]]

    while next_indices:
        idx = next_indices.pop(0)
        try:
            # If index on obstacle,
            if array[*idx] == 1:
                # Seach neighbours
                next_indices.extend([[idx[0] + offsets[i][0], idx[1] + offsets[i][1]] for i in range(4)])

                # Add index to returned list
                all_indices.append(idx)

                # Mark as searched
                array[*idx] = 0
        except IndexError:
            continue

    return np.array(all_indices)

def identify_obstacle(mask_array: np.ndarray, index: list[int], roughness: np.ndarray, identity_array: np.ndarray) -> None:
    """Classify an obstacle from boolean obstacle and roughness arrays
    
    Parameters
    ----------
    mask_array : np.ndarray
        Boolean array representing obstacles
    index : list[int]
        Position on obstacle
    roughness : np.ndarray
        Array containing cell roughness data
    identity-array : np.ndarray
        Array to add the obstacle classification to"""
    
    indices = flood_fill(mask_array, index)

    identity_array[indices[:,0], indices[:, 1]] = 1 if roughness[indices[:,0], indices[:,1]].mean() < 10 else 2

if __name__ == "__main__":
    gdal.DontUseExceptions()

    DSM_FILENAME = "TQ28se_DSM_1m.tif"
    DTM_FILENAME = "TQ28se_DTM_1m.tif"
    ROUGHNESS_FILENAME = "roughness.tif"

    fp = path.dirname(path.abspath(__file__))

    # Load DEMs
    ds = gdal.Open(path.join(fp, DSM_FILENAME))
    surface = ds.GetRasterBand(1).ReadAsArray()

    ds = gdal.Open(path.join(fp, DTM_FILENAME))
    terrain = ds.GetRasterBand(1).ReadAsArray()

    ds = gdal.Open(path.join(fp, ROUGHNESS_FILENAME))
    roughness = ds.GetRasterBand(1).ReadAsArray()
    roughness[roughness == -9999] = 0

    # Create obstacle mask
    obstacles: np.ndarray = surface - terrain

    obstacles[obstacles < 0.5] = 0
    obstacles[obstacles > 0.25] = 1

    # Classify all separate obstacles
    obstacle_types = np.ones_like(obstacles) * np.nan

    for y in range(obstacles.shape[0]):
        for x in range(obstacles.shape[1]):
            if obstacles[y, x] == 1 and np.isnan(obstacle_types[y, x]):
                print(x, y)
                identify_obstacle(obstacles, [y, x], roughness, obstacle_types)

    # Plot results
    plt.rcParams.update({'font.size': 14})

    map = ListedColormap(['navy', 'yellowgreen'])

    ax = plt.subplot()
    im = ax.imshow(obstacle_types, cmap=map, vmax=2)
    ax.set_xlabel("x position / m")
    ax.set_ylabel("y position / m")
    ax.set_title("Obstacle classification")

    # Custom colourbar axis
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    plt.colorbar(im, cax=cax)

    cax.yaxis.set_major_locator(ticker.FixedLocator([1.25, 1.75]))
    cax.yaxis.set_major_formatter(ticker.FixedFormatter(["Building", "Vegetation"]))


    plt.show()
