import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from mpl_toolkits.mplot3d import Axes3D

def target_function(theta, phi):
    term1 = 0.5 * (1 + np.cos(theta)**2) * np.cos(2 * phi)
    term2 = np.cos(theta) * np.sin(2 * phi)
    return np.sqrt(np.abs(term1)**2 + np.abs(term2)**2)

def cartesian_to_spherical(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(np.clip(z / r, -1, 1))
    phi = np.arctan2(y, x)
    phi = (phi + 2 * np.pi) % (2 * np.pi)
    return theta, phi

def get_detector_transformation(lat_deg, lon_deg, azimuth_deg):
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    az  = np.radians(azimuth_deg)

    u_zenith = np.array([
        np.cos(lat) * np.cos(lon),
        np.cos(lat) * np.sin(lon),
        np.sin(lat)
    ])
    
    
    u_north = np.array([
        -np.sin(lat) * np.cos(lon),
        -np.sin(lat) * np.sin(lon),
        np.cos(lat)
    ])
    
    
    u_east = np.array([
        -np.sin(lon),
        np.cos(lon),
        0
    ])

    
    u_x_arm = np.cos(az) * u_north + np.sin(az) * u_east
    
    u_y_arm = np.cross(u_zenith, u_x_arm)

    R = np.array([
        u_x_arm,   
        u_y_arm,  
        u_zenith 
    ])
    
    return R

def main():
    H1_LAT = 46.4551
    H1_LON = -119.4075
    H1_AZ  = 125.99    
    H1_AZ_REAL = 324.0 
    

    L1_LAT = 30.5628
    L1_LON = -90.7742
    L1_AZ_REAL = 252.0 

    res = 300
    theta_grid = np.linspace(0, np.pi, res)    
    phi_grid = np.linspace(-np.pi, np.pi, res)   
    THETA, PHI = np.meshgrid(theta_grid, phi_grid)

    
    x = np.sin(THETA) * np.cos(PHI)
    y = np.sin(THETA) * np.sin(PHI)
    z = np.cos(THETA)
    
    vectors = np.array([x.flatten(), y.flatten(), z.flatten()])
    
 
    R_H1 = get_detector_transformation(H1_LAT, H1_LON, H1_AZ_REAL)
    R_L1 = get_detector_transformation(L1_LAT, L1_LON, L1_AZ_REAL)
    
    
    v_h1 = R_H1 @ vectors
    v_l1 = R_L1 @ vectors
    
    
    theta_h1, phi_h1 = cartesian_to_spherical(v_h1[0], v_h1[1], v_h1[2])
    theta_l1, phi_l1 = cartesian_to_spherical(v_l1[0], v_l1[1], v_l1[2])
    
    
    THETA_H1 = theta_h1.reshape(THETA.shape)
    PHI_H1 = phi_h1.reshape(PHI.shape)
    THETA_L1 = theta_l1.reshape(THETA.shape)
    PHI_L1 = phi_l1.reshape(PHI.shape)

    sens_h1 = target_function(THETA_H1, PHI_H1)
    sens_l1 = target_function(THETA_L1, PHI_L1)
    
   
    network_sens = np.sqrt(sens_h1**2 + sens_l1**2)/np.sqrt(2)

  
    fig = plt.figure(figsize=(12, 6))

   
    ax1 = fig.add_subplot(1, 2, 1)
    im = ax1.imshow(np.transpose(network_sens), 
                    extent=[-180, 180, -90, 90], 
                    cmap='turbo', aspect='auto')
    
    ax1.set_title("LIGO Network Sensitivity (H1+L1)\n(Earth Coordinates)")
    ax1.set_xlabel("Longitude (deg)")
    ax1.set_ylabel("Latitude (deg)")
    
    
    ax1.plot(H1_LON, H1_LAT, 'w*', markersize=10, markeredgecolor='k', label='H1')
    ax1.plot(L1_LON, L1_LAT, 'w^', markersize=10, markeredgecolor='k', label='L1')
    ax1.legend()
    plt.colorbar(im, ax=ax1, label='Antenna Pattern Magnitude')

    
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    
    norm = colors.Normalize(vmin=network_sens.min(), vmax=network_sens.max())
    surface_colors = cm.turbo(norm(network_sens))
    
    ax2.plot_surface(x, y, z, facecolors=surface_colors, 
                     rstride=1, cstride=1, shade=False, antialiased=True)
    
    ax2.set_title("3D Projection")
    ax2.set_box_aspect([1,1,1])
    ax2.set_axis_off()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()