import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
from os import path
import time

from radar_funcs import *

fp = path.dirname(path.abspath(__file__))

##### Variables to change #####

target_elevation_m = 1500 / 3.28
target_rcs_m2 = 100

terrain = Terrain(path.join(fp, "50m_DTM.tif"), 50)
surface = Terrain(path.join(fp, "50m_DSM.tif"), 50)
radar = Radar(terrain, surface, [50950, 41550], 10, 600e6, 1500, 1e-6, 1e6, 15, 7.5, 1000)
radar.update_minimum_snr(0.9, 1e-6, 1)

target_grid_size = 50

output_file_name = "large_aircraft_SNR.csv"


##### Main code #####

# Test visibility over map
snr = np.inf * np.ones((int(np.floor(terrain.terrain.shape[1] / target_grid_size)), int(np.floor(terrain.terrain.shape[0] / target_grid_size))))

t = Point([0,0], target_elevation_m, relative_terrain=False)

tic = time.time()

# Record visibility for each point
for y in range(0, terrain.terrain.shape[1], target_grid_size):
    t.pos[1] = y * terrain.scale
    for x in range(0, terrain.terrain.shape[0], target_grid_size):
        t.pos[0] = x * terrain.scale

        snr[int(round(y / target_grid_size)), int(round(x / target_grid_size))] = radar.target_snr(t, target_rcs_m2, step_distance=50)

    if y % 100 == 0:
        print(f"{y / terrain.terrain.shape[1] * 100:.2f}% complete")

toc = time.time()

print(f"Finished in {toc - tic:.2f} seconds\n")


# Plot results
plt.rcParams.update({'font.size': 12})

fig, (ax1, blank, ax2) = plt.subplots(1, 3)
fig.suptitle("Line of sight radar simulation")

blank.axis("off")

# Plot terrain, and radar location
ax1.set_title("Radar location")
im1 = ax1.imshow(surface.get_terrain(), cmap='gist_earth', vmin=0)
ax1.plot(radar.pos[0] / terrain.scale, radar.pos[1] / terrain.scale, 'ro')

# Format axes
ax1.set_xlabel("x position / km")
ax1.set_ylabel("y position / km")
ax1.xaxis.set_major_locator(ticker.MultipleLocator(10000 / terrain.scale))
ax1.yaxis.set_major_locator(ticker.MultipleLocator(10000 / terrain.scale))
ax1.xaxis.set_major_formatter(lambda x, pos: str(int(round(x * terrain.scale / 1000))))
ax1.yaxis.set_major_formatter(lambda x, pos: str(int(round(x * terrain.scale / 1000))))

# Custom colourbar axis
divider = make_axes_locatable(ax1)
cax1 = divider.append_axes("right", size="5%", pad=0.05)

plt.colorbar(im1, label="Elevation / m", cax=cax1)


# Plot visibility
ax2.set_title(r"Visibility of large aircraft w/ vegetation ($\sigma=100$)")
im2 = ax2.imshow(snr, cmap='plasma', vmin=0)
ax2.plot(radar.pos[0] / (terrain.scale * target_grid_size), radar.pos[1] / (terrain.scale * target_grid_size), 'ro')

# Format axes
ax2.set_xlabel("x position / km")
ax2.set_ylabel("y position / km")
ax2.xaxis.set_major_locator(ticker.MultipleLocator(10000 / (target_grid_size * terrain.scale)))
ax2.yaxis.set_major_locator(ticker.MultipleLocator(10000 / (target_grid_size * terrain.scale)))
ax2.xaxis.set_major_formatter(lambda x, pos: str(int(round(x * target_grid_size * terrain.scale / 1000))))
ax2.yaxis.set_major_formatter(lambda x, pos: str(int(round(x * target_grid_size * terrain.scale / 1000))))

# Custom colourbar axis
divider = make_axes_locatable(ax2)
cax2 = divider.append_axes("right", size="5%", pad=0.05)

plt.colorbar(im2, label="Signal to noise ratio", cax=cax2)
cax2.set_ylabel("SNR / dB")

print("Saving to file...")
np.savetxt(path.join(fp, output_file_name), snr, delimiter=',')

plt.show()
