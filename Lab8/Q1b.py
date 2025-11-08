import numpy as np
import matplotlib.pyplot as plt
import os

x_vals = np.arange(0, 20.1, 0.1)
y_vals = np.arange(0, 8.1, 0.1)
X, Y = np.meshgrid(x_vals, y_vals)
T = np.zeros((len(y_vals), len(x_vals)))


def compute_segment(T, start_T, end_T, coord_1, coord_2, x_vals, y_vals):
    num_points = 1000
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


def gauss_seidel_step(T, interior, omega, T_boundary):
    for i in range(1, len(y_vals) - 1):
        for j in range(1, len(x_vals) - 1):
            if interior[i, j]:
                T[i, j] = ((1 + omega) / 4) * (
                        T[i + 1, j] + T[i - 1, j] + T[i, j + 1] + T[i, j - 1]
                ) - omega * T[i, j]
    T[~interior] = T_boundary[~interior]
    return T

A, B, C, D, E, F, G, H = [0,0], [5,0], [5,3], [15,3], [15,0], [20,0], [20,8], [0,8]

compute_segment(T, 0, 5, A, B, x_vals, y_vals)
compute_segment(T, 5, 7, B, C, x_vals, y_vals)
compute_segment(T, 7, 7, C, D, x_vals, y_vals)
compute_segment(T, 7, 5, D, E, x_vals, y_vals)
compute_segment(T, 5, 0, E, F, x_vals, y_vals)
compute_segment(T, 0, 10, F, G, x_vals, y_vals)
compute_segment(T, 10, 10, G, H, x_vals, y_vals)
compute_segment(T, 10, 0, H, A, x_vals, y_vals)

#Creating rectangle
vals_in = np.ones_like(T, dtype=bool)

vals_in[0, :] = False
vals_in[-1, :] = False
vals_in[:, 0] = False
vals_in[:, -1] = False

#Removing small recatngle
for i in range(len(y_vals)):
    for j in range(len(x_vals)):
        x, y = x_vals[j], y_vals[i]
        if 5 <= x <= 15 and 0 <= y <= 3:
            vals_in[i, j] = False

omega = 0.0
iterations = 100

output_dir = "Q1.21 images" #change to "Q1.22 images" for w = 0.9
os.makedirs(output_dir, exist_ok=True)

T_boundary = T.copy()
for iteration in range(iterations):
    T = gauss_seidel_step(T, vals_in, omega, T_boundary)

    plt.figure(figsize=(8, 4))
    plt.contourf(X, Y, T, levels=20, cmap='hot')
    plt.colorbar(label='Temperature (°C)')
    plt.title(f'Iteration {iteration}')

    filename = os.path.join(output_dir, f"iteration_{iteration:03d}.png")
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

