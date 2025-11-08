import numpy as np
import matplotlib.pyplot as plt
import os

x_vals = np.arange(0, 20.1, 0.1)
y_vals = np.arange(0, 8.1, 0.1)
X, Y = np.meshgrid(x_vals, y_vals)
T = np.zeros((len(y_vals), len(x_vals)))


def compute_segment(T, start_T, end_T, coord_1, coord_2, x_vals, y_vals):
    num_points = 500
    x_list = np.linspace(coord_1[0], coord_2[0], num_points)
    y_list = np.linspace(coord_1[1], coord_2[1], num_points)
    if start_T == end_T:
        T_list = np.full(num_points, start_T)

    else:
        T_list = np.linspace(start_T, end_T, num_points)

    i = 0
    while i < len(x_list):
        xi = np.argmin(np.abs(x_vals - x_list[i]))
        yi = np.argmin(np.abs(y_vals - y_list[i]))
        T[yi, xi] = T_list[i]
        i += 1


def gauss_seidel_step(T, interior, omega, T_boundary):  # ADDED T_boundary parameter
    for i in range(1, len(y_vals) - 1):
        for j in range(1, len(x_vals) - 1):
            if interior[i, j]:
                T[i, j] = ((1 + omega) / 4) * (
                        T[i + 1, j] + T[i - 1, j] + T[i, j + 1] + T[i, j - 1]
                ) - omega * T[i, j]

    T[~interior] = T_boundary[~interior]
    return T


A, B, C, D, E, F, G, H = [0, 0], [5, 0], [5, 3], [15, 3], [15, 0], [20, 0], [20, 8], [0, 8]
compute_segment(T, 0, 5, A, B, x_vals, y_vals)
compute_segment(T, 5, 7, B, C, x_vals, y_vals)
compute_segment(T, 7, 7, C, D, x_vals, y_vals)
compute_segment(T, 7, 5, D, E, x_vals, y_vals)
compute_segment(T, 5, 0, E, F, x_vals, y_vals)
compute_segment(T, 0, 10, F, G, x_vals, y_vals)
compute_segment(T, 10, 10, G, H, x_vals, y_vals)
compute_segment(T, 10, 0, H, A, x_vals, y_vals)

# Creating rectangle
vals_in = np.ones_like(T, dtype=bool)

vals_in[0, :] = False
vals_in[-1, :] = False
vals_in[:, 0] = False
vals_in[:, -1] = False

# Removing small recatngle
for i in range(len(y_vals)):
    for j in range(len(x_vals)):
        x, y = x_vals[j], y_vals[i]
        if 5 <= x <= 15 and 0 <= y <= 3:
            vals_in[i, j] = False

omega = 0.9

precision = 1e-6
max_iterations = 500

output_dir = "Q1c_images"
os.makedirs(output_dir, exist_ok=True)

iteration = 0
converged = False

T_boundary = T.copy()

while not converged and iteration < max_iterations:
    T_old = T.copy()
    T = gauss_seidel_step(T, vals_in, omega, T_boundary)

    max_change = np.max(np.abs(T[vals_in] - T_old[vals_in]))

    if max_change < precision:
        converged = True
        print(f"Converged after {iteration} iterations")

    plt.figure(figsize=(8, 4))
    plt.contourf(X, Y, T, levels=20, cmap='hot')
    plt.colorbar(label='Temperature (°C)')
    plt.title(f'Iteration {iteration}, Max Change: {max_change:.2e}°C')

    filename = os.path.join(output_dir, f"iteration_{iteration:03d}.png")
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

    iteration += 1
    print(f"cur it = {iteration}")  # we expect it to be between 200 and 400 iterations so we stop it after 500

import numpy as np
import matplotlib.pyplot as plt
import os

x_vals = np.arange(0, 20.1, 0.1)
y_vals = np.arange(0, 8.1, 0.1)
X, Y = np.meshgrid(x_vals, y_vals)
T = np.zeros((len(y_vals), len(x_vals)))


def compute_segment(T, start_T, end_T, coord_1, coord_2, x_vals, y_vals):
    num_points = 500
    x_list = np.linspace(coord_1[0], coord_2[0], num_points)
    y_list = np.linspace(coord_1[1], coord_2[1], num_points)
    if start_T == end_T:
        T_list = np.full(num_points, start_T)

    else:
        T_list = np.linspace(start_T, end_T, num_points)

    i = 0
    while i < len(x_list):
        xi = np.argmin(np.abs(x_vals - x_list[i]))
        yi = np.argmin(np.abs(y_vals - y_list[i]))
        T[yi, xi] = T_list[i]
        i += 1


def gauss_seidel_step(T, interior, omega, T_boundary):  # ADDED T_boundary parameter
    for i in range(1, len(y_vals) - 1):
        for j in range(1, len(x_vals) - 1):
            if interior[i, j]:
                T[i, j] = ((1 + omega) / 4) * (
                        T[i + 1, j] + T[i - 1, j] + T[i, j + 1] + T[i, j - 1]
                ) - omega * T[i, j]

    T[~interior] = T_boundary[~interior]
    return T


A, B, C, D, E, F, G, H = [0, 0], [5, 0], [5, 3], [15, 3], [15, 0], [20, 0], [20, 8], [0, 8]
compute_segment(T, 0, 5, A, B, x_vals, y_vals)
compute_segment(T, 5, 7, B, C, x_vals, y_vals)
compute_segment(T, 7, 7, C, D, x_vals, y_vals)
compute_segment(T, 7, 5, D, E, x_vals, y_vals)
compute_segment(T, 5, 0, E, F, x_vals, y_vals)
compute_segment(T, 0, 10, F, G, x_vals, y_vals)
compute_segment(T, 10, 10, G, H, x_vals, y_vals)
compute_segment(T, 10, 0, H, A, x_vals, y_vals)

# Creating rectangle
vals_in = np.ones_like(T, dtype=bool)

vals_in[0, :] = False
vals_in[-1, :] = False
vals_in[:, 0] = False
vals_in[:, -1] = False

# Removing small recatngle
for i in range(len(y_vals)):
    for j in range(len(x_vals)):
        x, y = x_vals[j], y_vals[i]
        if 5 <= x <= 15 and 0 <= y <= 3:
            vals_in[i, j] = False

omega = 0.9

tolerance = 1e-6
max_iterations = 500

output_dir = "Q1c_images"
os.makedirs(output_dir, exist_ok=True)

iteration = 0
converged = False

T_boundary = T.copy()

while not converged and iteration < max_iterations:
    T_old = T.copy()
    T = gauss_seidel_step(T, vals_in, omega, T_boundary)

    max_change = np.max(np.abs(T[vals_in] - T_old[vals_in]))

    if max_change < precision:
        converged = True
        print(f"Converged after {iteration} iterations")

    plt.figure(figsize=(8, 4))
    plt.contourf(X, Y, T, levels=20, cmap='hot')
    plt.colorbar(label='Temperature (°C)')
    plt.title(f'Iteration {iteration}, Max Change: {max_change:.2e}°C')

    filename = os.path.join(output_dir, f"iteration_{iteration:03d}.png")
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

    iteration += 1
    print(f"cur it = {iteration}")  # we expect it to be between 200 and 400 iterations so we stop it after 500

x_target = 2.5
y_target = 1.0
xi = np.argmin(np.abs(x_vals - x_target))
yi = np.argmin(np.abs(y_vals - y_target))
T_target = T[yi, xi]

print(f"Temperature at (x={x_target} cm, y={y_target} cm): {T_target:.6f}°C")

plt.figure(figsize=(10, 5))
plt.contourf(X, Y, T, levels=20, cmap='hot')
plt.colorbar(label='Temperature (°C)')
plt.title(f'Converged Solution (ω=0.9, precision=10⁻⁶°C)\nIterations: {iteration}')
plt.xlabel('x (cm)')
plt.ylabel('y (cm)')

plt.plot(x_target, y_target, 'g*', markersize=10, label=f'T({x_target},{y_target})={T_target:.3f}°C')
plt.legend()

plt.savefig('Q1c_sol.png', dpi=150, bbox_inches='tight')
plt.show()