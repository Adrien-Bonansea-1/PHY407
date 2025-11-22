import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

#Part B
print("Part B ----------------------------------")
N = 5000
points_the = np.random.random(N)
points_phi = np.random.random(N)

theta = np.arccos(1 - (2 * points_the))
phi = 2 * np.pi * points_phi

lat = 90 - np.degrees(theta)
lon = np.degrees(phi)

#2D plot
plt.figure(figsize=(10, 6))
plt.scatter(lon, lat, s=1, alpha=0.5)
plt.xlabel('Longitude (degrees)')
plt.ylabel('Latitude (degrees)')
plt.title('2D plot: Random Points on Earth Surface')
plt.xlim(0, 360)
plt.ylim(-90, 90)
plt.grid(True)
plt.savefig("part b: 2d")
plt.show()

#3D plot

#spherical coordinates
r = 1
x = r * np.sin(theta) * np.cos(phi)
y = r * np.sin(theta) * np.sin(phi)
z = r * np.cos(theta)

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(x, y, z, s=1, alpha=0.5, c='blue')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('5000 Random Points on Sphere')

ax.set_box_aspect([1,1,1])

plt.savefig("part b: 3d")
plt.show()

#Part C
print("Part C ----------------------------------")
loaded = np.load('Earth.npz')
data = loaded['data']
lon_array = loaded['lon']
lat_array = loaded['lat']

theta_array = np.pi/2 - np.radians(lat_array)
sin_theta = np.sin(theta_array)
land_weighted = data * sin_theta
land_area = np.sum(land_weighted)

total_area = np.sum(sin_theta) * len(lon_array)

land_fraction = land_area / total_area
print(f"Land percentage: {land_fraction * 100:.2f}%")

#Part D
print("Part D ----------------------------------")
loaded = np.load('Earth.npz')
data = loaded['data']
lon_array = loaded['lon']
lat_array = loaded['lat']

interp = RegularGridInterpolator((lon_array, lat_array), data,
                                 method='nearest', bounds_error=False, fill_value=0)


def land_frac_func(N, interp):
    points_the = np.random.random(N)
    points_phi = np.random.random(N)

    theta = np.arccos(1 - (2 * points_the))
    phi = 2 * np.pi * points_phi

    lat = 90 - np.degrees(theta)
    lon = np.degrees(phi)
    lon = np.where(lon > 180, lon - 360, lon)

    points = np.column_stack([lon, lat])
    land = interp(points)

    num_land = np.sum(land)
    land_frac = num_land / N

    return land_frac, lon, lat, land


N_vals = [50, 500, 5000, 50000]

results = {}
for N in N_vals:
    land_frac, lons, lats, land = land_frac_func(N, interp)
    results[N] = (land_frac, lons, lats, land)

    print(f"N = {N:5d}: Land fraction = {land_frac:.4f} ({land_frac * 100:.2f}%)")

N = 50000
land_frac, lons, lats, land = results[N]

land_lons = lons[land == 1]
land_lats = lats[land == 1]
not_land_lons = lons[land == 0]
not_land_lats = lats[land == 0]

plt.figure(figsize=(14, 7))
plt.scatter(not_land_lons, not_land_lats, s=1, c='blue', alpha=0.3, label='Other')
plt.scatter(land_lons, land_lats, s=1, c='green', alpha=0.5, label='Land')
plt.xlabel('Longitude (degrees)')
plt.ylabel('Latitude (degrees)')
plt.title(f'Random Sampling Points Only (N={N})')
plt.xlim(lon_array.min(), lon_array.max())
plt.ylim(lat_array.min(), lat_array.max())
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('map')
plt.show()