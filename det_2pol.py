import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

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


    res = 200
    

    lon_deg = np.linspace(-180, 180, res)  # X-axis
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

if __name__ == "__main__":
    main()