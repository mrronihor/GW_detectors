import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors

def target_function(theta, phi):
    term1 = 0.5 * (1 + np.cos(theta)**2) * np.cos(2 * phi)
    term2 = np.cos(theta) * np.sin(2 * phi)
    return np.sqrt(np.abs(term2)**2+np.abs(term1)**2)

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

def get_detector_tensor(lat_deg, lon_deg, azimuth_deg):
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    az  = np.radians(azimuth_deg)


    u_zenith = np.array([np.cos(lat)*np.cos(lon), np.cos(lat)*np.sin(lon), np.sin(lat)])
    u_north  = np.array([-np.sin(lat)*np.cos(lon), -np.sin(lat)*np.sin(lon), np.cos(lat)])
    u_east   = np.array([-np.sin(lon), np.cos(lon), 0])


    u_x = np.cos(az) * u_north + np.sin(az) * u_east
    u_y = np.cross(u_zenith, u_x)


    D = 0.5 * (np.outer(u_x, u_x) - np.outer(u_y, u_y))
    return D

def get_polarization_basis_sky(theta_colat, phi_lon):
    u_north = np.array([
        np.cos(theta_colat) * np.cos(phi_lon),
        np.cos(theta_colat) * np.sin(phi_lon),
        -np.sin(theta_colat)
    ])
    

    u_east = np.array([
        -np.sin(phi_lon),
        np.cos(phi_lon),
        0
    ])
    

    return u_north, u_east

def main():

    # Hanford (H1)
    H1_LAT, H1_LON, H1_AZ = 46.45, -119.41, 324.0
    # Livingston (L1)
    L1_LAT, L1_LON, L1_AZ = 30.56, -90.77, 252.0
    

    D_H1 = get_detector_tensor(H1_LAT, H1_LON, H1_AZ)
    D_L1 = get_detector_tensor(L1_LAT, L1_LON, L1_AZ)


    res = 300
    

    lon_deg = np.linspace(-180, 180, 2*res)  # X-axis
    lat_deg = np.linspace(-90, 90, res)    # Y-axis
    

    LON, LAT = np.meshgrid(lon_deg, lat_deg)
    

    THETA_COLAT = np.radians(90 - LAT)
    PHI_LON     = np.radians(LON)


    flat_theta = THETA_COLAT.flatten()
    flat_phi   = PHI_LON.flatten()
    flat_ratio = np.zeros_like(flat_theta)
    
    for i in range(len(flat_theta)):
        th = flat_theta[i]
        ph = flat_phi[i]
        

        u, v = get_polarization_basis_sky(th, ph)
        

        e_plus  = np.outer(u, u) - np.outer(v, v)
        e_cross = np.outer(u, v) + np.outer(v, u)
        

        fp_h1 = np.sum(D_H1 * e_plus)
        fx_h1 = np.sum(D_H1 * e_cross)
        
        fp_l1 = np.sum(D_L1 * e_plus)
        fx_l1 = np.sum(D_L1 * e_cross)
        

        A = fp_h1**2 + fp_l1**2       
        B = fx_h1**2 + fx_l1**2       
        C = fp_h1*fx_h1 + fp_l1*fx_l1 
        

        trace = A + B
        det   = A*B - C**2
        

        if det < 0: det = 0
        
        delta = np.sqrt(trace**2 - 4*det)
        
        lam1 = (trace + delta) / 2.0 # Max
        lam2 = (trace - delta) / 2.0 # Min
        
        if lam1 > 1e-15:
            ratio = np.sqrt(lam2 / lam1)
        else:
            ratio = 0.0
            
        flat_ratio[i] = ratio

    ratio_map = flat_ratio.reshape(LAT.shape)


    fig, ax = plt.subplots(figsize=(10, 5))
    
    

    im = ax.imshow(ratio_map, extent=[-180, 180, -90, 90], origin='lower',
                   cmap='jet', aspect='auto')


    ax.scatter([H1_LON], [H1_LAT], facecolors='black', edgecolors='white', 
               marker='*', s=200, label='H1', zorder=10)

    ax.scatter([L1_LON], [L1_LAT], facecolors='white', edgecolors='black', 
               marker='*', s=200, label='L1', zorder=10)
    

    ax.set_title("LIGO Network Alignment Factor (Polarization Capability)")
    ax.set_xlabel("Longitude $\phi$ (deg)")
    ax.set_ylabel("Latitude $\theta$ (deg)")
    

    ax.grid(True, linestyle=':', color='k', alpha=0.3)
    

    cbar = plt.colorbar(im, ax=ax, aspect=20)
    cbar.set_label(r"Ratio $\sqrt{\lambda_{min}/\lambda_{max}}$", rotation=270, labelpad=20)
    
    plt.tight_layout()
    plt.show()
    plt.clf()
    
    H1_LAT = 46.4551
    H1_LON = -119.4075
    H1_AZ  = 125.99    
    H1_AZ_REAL = 324.0 
    

    L1_LAT = 30.5628
    L1_LON = -90.7742
    L1_AZ_REAL = 252.0 

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
                    cmap='turbo', aspect='auto', vmin=0., vmax=1.)
    
    ax1.set_title("LIGO Network Sensitivity (H1+L1)\n(Earth Coordinates)")
    ax1.set_xlabel("Longitude (deg)")
    ax1.set_ylabel("Latitude (deg)")
    
    
    ax1.plot(H1_LON, H1_LAT, 'w*', markersize=10, markeredgecolor='k', label='H1')
    ax1.plot(L1_LON, L1_LAT, 'w^', markersize=10, markeredgecolor='k', label='L1')
    ax1.legend()
    plt.colorbar(im, ax=ax1, label='Antenna Pattern Magnitude')

    
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    
    norm = colors.Normalize(vmin=0., vmax=1.)
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