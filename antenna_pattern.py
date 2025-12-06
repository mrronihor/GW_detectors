import matplotlib.pyplot as plt
import numpy as np

scale=200

data_plus=np.zeros((scale, 2*scale))
data_cross=np.zeros((scale, 2*scale))
data_all=np.zeros((scale, 2*scale))

for i in range(scale):
    for j in range(2*scale):
        data_plus[i,j]=np.absolute(1/2*(1+np.cos(i*np.pi/scale)**2)*np.cos(2*np.pi*j/scale))#-np.cos(i*np.pi/scale)*np.sin(2*np.pi*j/scale)#-(np.cos(i*np.pi/scale))**2+2*(np.cos(np.pi*j/scale))**2-1
        data_cross[i,j]=np.absolute(np.cos(i*np.pi/scale)*np.sin(2*np.pi*j/scale))
        data_all[i,j]=np.sqrt(data_plus[i,j]**2+data_cross[i,j]**2)

plt.imshow(data_plus, cmap='coolwarm')
plt.colorbar()
#plt.savefig('antenna_plus.png')
#plt.show()

plt.clf()
plt.imshow(data_cross, cmap='coolwarm')
plt.colorbar()
#plt.savefig('antenna_cross.png')
#plt.show()

plt.clf()
plt.imshow(data_all, cmap='coolwarm')
plt.colorbar()
#plt.savefig('antenna_all.png')
#plt.show()

plt.clf()

theta = np.linspace(0, np.pi, scale)
phi = np.linspace(0, 2 * np.pi, scale)

THETA, PHI = np.meshgrid(theta, phi)

RHO = 1/2*(1+np.cos(THETA)**2)*np.cos(2*PHI)

X = RHO * np.sin(THETA) * np.cos(PHI)
Y = RHO * np.sin(THETA) * np.sin(PHI)
Z = RHO * np.cos(THETA)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(X, Y, 2*Z, cmap='coolwarm',linewidth=0, antialiased=False, rstride=1, cstride=1)

ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')
ax.set_zlabel('Z Axis')
ax.set_title('Spherical Coordinate Plot')

fig.colorbar(surf, shrink=0.5, aspect=10, label='Z-value')


ax.set_box_aspect([np.ptp(X), np.ptp(Y), np.ptp(Z)]) 
#plt.savefig('antenna_plus_3d.png')
#plt.show()

plt.clf()
theta = np.linspace(0, np.pi, scale)
phi = np.linspace(0, 2*np.pi, scale)

THETA, PHI = np.meshgrid(theta, phi)

RHO_CROSS = np.cos(THETA)*np.sin(2*PHI)

X = RHO_CROSS * np.sin(THETA) * np.cos(PHI)
Y = RHO_CROSS * np.sin(THETA) * np.sin(PHI)
Z = RHO_CROSS * np.cos(THETA)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(X, Y, 2*Z, cmap='coolwarm',linewidth=0, antialiased=False, rstride=1, cstride=1)

ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')
ax.set_zlabel('Z Axis')
ax.set_title('Spherical Coordinate Plot')

fig.colorbar(surf, shrink=0.5, aspect=10, label='Z-value')


ax.set_box_aspect([np.ptp(X), np.ptp(Y), np.ptp(Z)]) 
plt.show()