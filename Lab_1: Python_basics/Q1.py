# Q1:
# The equations requested in part A will be in the Questions_and_Explanations txt file
# The Pseudo code for part B will be done in the Questions_and_Explanations txt file

#imports
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as spc

#Part C

#Define Constants
G = spc.G
Ms = 2.0 * 10**30

#Define Initial conditions
x = float(0.47)
y = float(0)
Vx = float(0)
Vy = float(8.17)
V = np.sqrt(Vx**2 + Vy**2)

#Conversions:
x = x * (1.496 * 10**11)
Vy = Vy * (1.496 * 10**11)/(3.156 * 10**7)

#Define dt:
current_t = 0
dt = 0.0001 * (3.156 * 10**7)
t_end = 3.156 * 10**7 #1 year in s

time_list, x_list, y_list, v_list = [], [], [], []

#Store initial
time_list.append(current_t)
x_list.append(x)
y_list.append(y)
v_list.append(V)

'''
while current_t <= t_end:
    Vx = Vx + (-G * Ms * x / (np.sqrt(x**2 + y**2))**3) * dt
    Vy = Vy + (-G * Ms * y / (np.sqrt(x**2 + y**2))**3) * dt
    x = x + Vx * dt
    y = y + Vy * dt

    time_list.append(current_t)
    x_list.append(x)
    y_list.append(y)
    V = np.sqrt(Vx ** 2 + Vy ** 2)
    v_list.append(V)

    current_t += dt

#ChatGPT was used to create the matplotlib script, which was then edited by us
plt.figure(figsize=(10,5))
plt.plot(time_list, v_list)
plt.xlabel("Time (s)")
plt.ylabel("Velocity (m/s)")
plt.xlim(1,t_end)
plt.title("Velocity vs Time")
plt.grid()
plt.show()

plt.figure(figsize=(6,6))
plt.plot(x_list, y_list)
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.title("Orbit around central mass")
plt.scatter(0,0, color="blue", marker="o", label="Central mass")
plt.legend()
plt.axis("equal")
plt.grid()
plt.show()
'''

#Part D
#Define alpha and Mp
alpha = 0.01 * (1.496 * 10**11)**2
Mp = 3.285 * 10**23

time_list, x_list, y_list, v_list = [], [], [], []

#Store initial
time_list.append(current_t)
x_list.append(x)
y_list.append(y)
v_list.append(V)

while current_t <= t_end:
    Fx = -G * Ms * Mp / (np.sqrt(x ** 2 + y ** 2)) ** 3 * (1 + alpha / (np.sqrt(x ** 2 + y ** 2)) ** 2) * x
    Fy = -G * Ms * Mp / (np.sqrt(x ** 2 + y ** 2)) ** 3 * (1 + alpha / (np.sqrt(x ** 2 + y ** 2)) ** 2) * y
    Vx = Vx + Fx * dt
    Vy = Vy + Fy * dt
    x = x + Vx * dt
    y = y + Vy * dt

    time_list.append(current_t)
    x_list.append(x)
    y_list.append(y)
    V = np.sqrt(Vx ** 2 + Vy ** 2)
    v_list.append(V)

    current_t += dt

plt.figure(figsize=(6,6))
plt.plot(x_list, y_list)
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.title("Mercury orbit with relativity")
plt.scatter(0,0, color="blue", marker="o", label="Central mass")
plt.legend()
plt.axis("equal")
plt.grid()
plt.show()
