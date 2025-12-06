import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R

H, W = 180, 360

def gw_detect(phi, theta):
    return np.sqrt(np.absolute(1/2*(1+np.cos(theta)**2)*np.cos(2*phi))**2+np.absolute(np.cos(theta)*np.sin(2*phi))**2)

def spherical_to_cartesian(lat, lon):
    """
    Inputs: lat, lon in radians
    Output: x, y, z vectors
    """
    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon)
    z = np.sin(lat)
    return x, y, z

def cartesian_to_spherical(x, y, z):
    """
    Inputs: x, y, z vectors
    Output: lat, lon in radians
    """
    lat = np.arcsin(np.clip(z, -1, 1))
    lon = np.arctan2(y, x)
    return lat, lon

hanford=np.array([46.45528, 119.40778])/360*2*np.pi
livingston=np.array([30.56306, 90.77444])/360*2*np.pi

r_h = R.from_rotvec([hanford[0], hanford[1],0.])

fig = plt.figure()
ax = fig.add_subplot(111, projection='mollweide')

lon = np.linspace(-np.pi, np.pi,W)
lat = np.linspace(-np.pi/2., np.pi/2.,H)
Lon,Lat = np.meshgrid(lon,lat)

gx, gy, gz = spherical_to_cartesian(Lat, Lon)

c_p, s_p = np.cos(-hanford[1]), np.sin(-hanford[1])
Rz_inv = np.array([
    [c_p, -s_p, 0],
    [s_p,  c_p, 0],
    [0,    0,   1]
])


c_t, s_t = np.cos(-hanford[0]), np.sin(-hanford[0])
Ry_inv = np.array([
    [ c_t, 0, s_t],
    [ 0,   1, 0  ],
    [-s_t, 0, c_t]
])

points_global = np.array([gx.flatten(), gy.flatten(), gz.flatten()])

points_rotated = Rz_inv @ points_global # First un-spin
points_rotated = Ry_inv @ points_rotated # Then un-tilt


rx, ry, rz = points_rotated[0], points_rotated[1], points_rotated[2]
rlat, rlon = cartesian_to_spherical(rx, ry, rz)


values = gw_detect(rlat, rlon)
heatmap = values.reshape(H,W)
plt.figure(figsize=(10, 6))
plt.imshow(heatmap, extent=[-180, 180, -90, 90], origin='lower', cmap='rainbow')
plt.title("Heatmap Rotated")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.colorbar(label="Intensity")
plt.grid(color='white', alpha=0.3, linestyle='--')
plt.show()
