# This code takes the initial position and velocity of the planet, uses the Euler-Cromer method to update
# the position and velocity of the planet under the Newtonian gravity force, and output two plots:
# velocity as a function of time and the orbit (x vs y) in space.
# Authored by Adrien Bonansea for PHY407 coursework.

# Pseudocode for Q1c:
# Define constants: Ms (mass of the Sun), G (gravitational constant), and conversion factor from AU to meters
# Define initial conditions: positions (x, y) and velocities (vx, vy) for Mercury in SI units.
# Initialize time variables: current time current_t = 0, end time t_end = 1 year, and time step dt = 0.0001 year.
# While t < t_end, repeat:
#     Update Mercury's velocity using the Euler-Cromer method.
#     Update Mercury's position using the Euler-Cromer method.
#     Store Mercury's new coordinates.
#     Increment time by dt.
# Calculate the specific angular momentum per mass
# Plot the angular momentum, velocity, and orbital paths of Mercury.

# For Q1c
# Import dependencies
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as spc

# Define constants
G = spc.G                   # gravitational constant in m^3kg^-1s^-2
Ms = 2e30                   # solar mass in kg
AU = 1.496e11               # conversion from AU to m
yr = 365.25 * 24 * 3600     # conversion from years to seconds

# Define initial positions and velocities of planet (assumed to be Mercury)
x = float(0.47) * AU            # x position of Mercury
y = float(0) * AU               # y position of Mercury
Vx = float(0) * AU / yr         # x component of Mercury's velocity
Vy = float(8.17) * AU / yr      # y component of Mercury's velocity
V = np.sqrt(Vx**2 + Vy**2)      # total velocity of Mercury

# Initialize start time, end time, and dt
current_t = 0
dt = 0.0001 * yr
t_end = 1 * yr

# Initialize storage
time_list, x_list, y_list, v_list, vx_list, vy_list = [], [], [], [], [], []

time_list.append(current_t)
x_list.append(x)
y_list.append(y)
v_list.append(V)
vx_list.append(Vx)
vy_list.append(Vy)

# Loop: Updates the positions and velocities of Mercury using Euler-Cromer under Newtonian gravitational forces
while current_t <= t_end:
    Vx = Vx + (-G * Ms * x / (np.sqrt(x**2 + y**2))**3) * dt
    Vy = Vy + (-G * Ms * y / (np.sqrt(x**2 + y**2))**3) * dt
    x = x + Vx * dt
    y = y + Vy * dt

    vx_list.append(Vx)
    vy_list.append(Vy)
    x_list.append(x)
    y_list.append(y)

    V = np.sqrt(Vx ** 2 + Vy ** 2)
    v_list.append(V)

    current_t += dt
    time_list.append(current_t)

# Convert lists to arrays
x_list = np.array(x_list)
y_list = np.array(y_list)
vx_list = np.array(vx_list)
vy_list = np.array(vy_list)
time_list = np.array(time_list)

# Calculate the z-component of the angular momentum per mass
l_list = x_list * vy_list - y_list * vx_list

# Plot angular momentum vs time
plt.figure(figsize=(8,6))
plt.plot(time_list, l_list)
plt.xlabel('Time (s)')
plt.ylabel(r'$\frac{L_z}{m}$')
plt.title('Specific Angular Momentum per Mass vs Time')
plt.grid(True)
plt.savefig("Angular Momentum vs Time")
plt.show()

# Plot the x-component of velocity as a function of time
plt.figure(figsize=(10,5))
plt.plot(time_list, vx_list)
plt.xlabel("Time (s)")
plt.ylabel(r"$v_x$ (m/s)")
plt.xlim(1, t_end)
plt.title("Mercury: $v_x$ vs Time (Euler-Cromer)")
plt.grid(True)
plt.savefig("xVelocity_vs_Time.png")
plt.show()

# Plot the y-component of velocity as a function of time
plt.figure(figsize=(10,5))
plt.plot(time_list, vy_list)
plt.xlabel("Time (s)")
plt.ylabel(r"$v_y$ (m/s)")
plt.xlim(1, t_end)
plt.title("Mercury: $v_y$ vs Time (Euler-Cromer)")
plt.grid(True)
plt.savefig("yVelocity_vs_Time.png")
plt.show()

# Plot the orbit
plt.figure(figsize=(6,6))
plt.plot(x_list, y_list)
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.title("Mercury's Orbit using Euler-Cromer under Newtonian Gravity")
plt.scatter(0, 0, color="yellow", marker="o", label="Sun")
plt.legend()
plt.axis("equal")
plt.grid()
plt.savefig("Mercury Orbit")
plt.show()

# Pseudocode for Q1d
# Define extra constants: Mp (mass of Mercury) and alpha
# Define initial conditions: positions (x, y) and velocities (vx, vy) for Mercury in SI units.
# Initialize time variables: current time current_t = 0, end time t_end = 1 year, and time step dt = 0.0001 year.
# While t < t_end, repeat:
#     Update the gravitational force under general relativity
#     Update Mercury's velocity using the Euler-Cromer method under general relativity.
#     Update Mercury's position using the Euler-Cromer method under general relativity.
#     Store Mercury's new coordinates.
#     Increment time by dt.
# Compute r from x and y
# Find the largest r, i.e., the aphelions
# Plot the orbital paths of Mercury with the moving-around aphelions

# For Q1d
# Define constants
G = spc.G                   # gravitational constant in m^3kg^-1s^-2
Ms = 2e30                   # solar mass in kg
AU = 1.496e11               # conversion from AU to m
yr = 365.25 * 24 * 3600     # conversion from years to seconds
alpha = 0.01 * AU**2        # alpha in m^2
Mp = 3.3 * 10**23           # Mercury's mass in kg

# Define initial positions and velocities of planet (assumed to be Mercury)
x = float(0.47) * AU            # x position of Mercury
y = float(0) * AU               # y position of Mercury
Vx = float(0) * AU / yr         # x component of Mercury's velocity
Vy = float(8.17) * AU / yr      # y component of Mercury's velocity
V = np.sqrt(Vx**2 + Vy**2)      # total velocity of Mercury

# Initialize start time, end time, and dt
current_t = 0
dt = 0.0001 * yr
t_end = 1 * yr

# Initialize storage
time_list, x_list, y_list, v_list = [], [], [], []

time_list.append(current_t)
x_list.append(x)
y_list.append(y)
v_list.append(V)

# Loop: Updates the positions and velocities of Mercury using Euler-Cromer under general relativity
while current_t <= t_end:
    r = np.sqrt(x ** 2 + y ** 2)

    F = (-G * Ms * Mp / r ** 3)
    Fx = F * (1 + alpha / r ** 2) * x
    Fy = F * (1 + alpha / r ** 2) * y

    Vx = Vx + Fx * dt / Mp
    Vy = Vy + Fy * dt / Mp
    V = np.sqrt(Vx ** 2 + Vy ** 2)

    x = x + Vx * dt
    y = y + Vy * dt

    time_list.append(current_t)
    x_list.append(x)
    y_list.append(y)
    v_list.append(V)

    current_t += dt

# Convert lists to arrays
r = np.sqrt((np.array(x_list))**2 + (np.array(y_list))**2)

# Initialize storage for aphelion's position
aphelion_x, aphelion_y = [], []

# Find the aphelion
i = 0
while i < len(r):
    if r[i] > r[i-1] and r[i] > r[i+1]:
        aphelion_x.append(x_list[i])
        aphelion_y.append(y_list[i])
    i += 1

# Plot the orbit with the moving-around aphelion
plt.figure(figsize=(6,6))
plt.plot(x_list, y_list)
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.title("Mercury's Orbit using Euler-Cromer under General Relativity")
plt.scatter(0, 0, color="yellow", marker="o", label="Sun")
plt.scatter(aphelion_x, aphelion_y, color="red", marker="o", label="Aphelions locations")
plt.legend()
plt.axis("equal")
plt.grid()
plt.savefig("Mercury Orbit GR")

plt.show()
