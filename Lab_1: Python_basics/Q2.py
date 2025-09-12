# Q2:
# Pseudocode:
# Define constants Ms, G, Mj, and the conversion of 1 AU to meters
# Define initial conditions, i.e., the coordinates and velocities for both Earth and Jupiter
# Initialize current_time, end_time, and dt
# While current_time is less than end_time
#     Increment the locations (xj and yj) and velocities (Vxj and Vyj) of Jupiter using Euler-Cromer
#     Store the new coordinates of Jupiter
#     Calculate the distance between the Earth and Jupiter's new location
#     Increment the locations (xe and ye) and velocities (Vxe and Vye) of Earth using Euler-Cromer
#     Store the new coordinates of the Earth
#     Increment current_time by dt
# Plot both Earth's and Jupiter's orbits

# Import dependencies
import numpy as np
import matplotlib.pyplot as plt

# Define constants
Ms = 2e30 # in kg
Mj = Ms * 1e-3 # in kg --> For 2a and 2c
# Mj = 1000 * Ms * 1e-3 # in kg --> For 2b
G = 6.673e-11 # in m^3kg^-1s^-2
AU = 1.496e11 # in m
yr = 365.25 * 24 * 3600

# Define initial conditions
xj = 5.2 * AU
yj = 0.0 * AU
xe = 1.0 * AU
ye = 0.0 * AU
xa = 3.3 * AU
ya = 0.0 * AU

#Conversions
vxj = 0.0 * AU/yr
vyj = 2.63 * AU/yr
vj = np.sqrt(vxj**2 + vyj**2)
vxe = 0.0 * AU/yr
vye = 6.18 * AU/yr
ve = np.sqrt(vxe**2 + vye**2)
vxa = 0.0 * AU/yr
vya = 3.46 * AU/yr
va = np.sqrt(vxa**2 + vya**2)

# Initialize start time, end time, and dt
t_current = 0
dt = 0.0001 * yr
t_end = 10 * yr # --> For 2a
# t_end = 3 * yr # --> For 2b
# t_end = 20 * yr # --> For 2c

# Initialize storage
time_list, xj_list, yj_list, vj_list, xe_list, ye_list, ve_list, xa_list, ya_list, va_list = [], [], [], [], [], [], [], [], [], []

time_list.append(t_current)
xj_list.append(xj)
yj_list.append(yj)
vj_list.append(vj)
xe_list.append(xe)
ye_list.append(ye)
ve_list.append(ve)
xa_list.append(xa)
ya_list.append(ya)
va_list.append(va)

# Loop 
while t_current <= t_end:
    vxj = vxj + (-G * Ms * xj / (np.sqrt(xj**2 + yj**2))**3) * dt
    vyj = vyj + (-G * Ms * yj / (np.sqrt(xj**2 + yj**2))**3) * dt
    xj = xj + vxj * dt
    yj = yj + vyj * dt

    xj_list.append(xj)
    yj_list.append(yj)
    vj_list.append(np.sqrt(vxj**2 + vyj**2))

    rej = np.sqrt((xj - xe)**2 + (yj - ye)**2)

    vxe = vxe + (-G * Ms * xe / (np.sqrt(xe**2 + ye**2))**3) * dt + (-G * Mj * (xe-xj) / (rej)**3) * dt
    vye = vye + (-G * Ms * ye / (np.sqrt(xe**2 + ye**2))**3) * dt + (-G * Mj * (ye-yj) / (rej)**3) * dt
    xe = xe + vxe * dt
    ye = ye + vye * dt

    xe_list.append(xe)
    ye_list.append(ye)
    ve_list.append(np.sqrt(vxe**2 + vye**2))

    raj = np.sqrt((xj - xa)**2 + (yj - ya)**2)

    vxa = vxa + (-G * Ms * xa / (np.sqrt(xa**2 + ya**2))**3) * dt + (-G * Mj * (xa-xj) / (raj)**3) * dt
    vya = vya + (-G * Ms * ya / (np.sqrt(xa**2 + ya**2))**3) * dt + (-G * Mj * (ya-yj) / (raj)**3) * dt
    xa = xa + vxa * dt
    ya = ya + vya * dt

    xa_list.append(xa)
    ya_list.append(ya)
    va_list.append(np.sqrt(vxa**2 + vya**2))

    t_current += dt
    time_list.append(t_current)

# Convert from m to AU
xj_list = [item / AU for item in xj_list]
yj_list = [item / AU for item in yj_list]
xe_list = [item / AU for item in xe_list]
ye_list = [item / AU for item in ye_list]
xa_list = [item / AU for item in xa_list]
ya_list = [item / AU for item in ya_list]

plt.figure(figsize=(6,6))
plt.plot(xj_list, yj_list, color="orange", label="Jupiter")
plt.plot(xe_list, ye_list, color="blue", label="Earth") # --> For 2a and 2b
# plt.plot(xa_list, ya_list, color="green", label="Asteroid") # --> For 2c
plt.xlabel("x (AU)")
plt.ylabel("y (AU)")
plt.title("Three Body Problem")
plt.scatter(0, 0, color="yellow", marker="o", label="Sun")
plt.legend(loc='upper right') 
plt.grid()
plt.show()

