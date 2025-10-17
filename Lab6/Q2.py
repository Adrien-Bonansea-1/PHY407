import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


def matrix_set_up(N):
    # setting up a N * N matrix like in eq(19)

    center_diag = - 2 * np.ones(N)
    front_diag = np.ones(N - 1)
    last_diag = np.ones(N - 1)

    return np.diag(front_diag, 1) + np.diag(center_diag, 0) + np.diag(last_diag, -1)


def verlet(matrix, dt, x0_list, v0_list, T):
    t_list = np.arange(0, T + dt / 2, dt)
    len_time = t_list.size

    x_list, v_list = [], []
    x = x0_list.copy()
    v = v0_list.copy()

    a = matrix @ x

    x_list.append(x.copy())
    v_list.append(v.copy())

    v_next = v_list[0] + 0.5 * dt * a

    i = 1
    while (i < len_time):
        x_new = x + dt * v_next
        a_new = matrix @ x_new
        v_new = v_next + 0.5 * dt * a_new
        v_next = v_next + dt * a_new

        x_list.append(x_new.copy())
        v_list.append(v_new.copy())

        x = x_new

        i += 1
    return t_list, np.array(x_list), np.array(v_list)


Ns = [3, 10]
k_m = 400
dt = 0.001
T = 2

for N in Ns:
    matrix = k_m * matrix_set_up(N)
    x0s = np.zeros(N)
    x0s[0] = 0.1
    v0s = np.zeros(N)

    t, x, v = verlet(matrix, dt, x0s, v0s, T)

    plt.figure(figsize=(10, 6))
    for i in range(N):
        plt.plot(t, x[:, i], label=f"Floor {i}")

    plt.xlabel("Time (s)")
    plt.ylabel("Displacement (m)")
    plt.title(f"Building Vibrations (N = {N} floors)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"Q2a Vibrations Full N = {N}.png")
    plt.show()

for N in Ns:
    matrix = k_m * matrix_set_up(N)
    x0s = np.zeros(N)
    x0s[0] = 0.1
    v0s = np.zeros(N)

    t, x, v = verlet(matrix, dt, x0s, v0s, T)

    mask = t <= 0.5

    plt.figure(figsize=(10, 6))
    for i in range(N):
        plt.plot(t[mask], x[mask, i], label=f"Floor {i}", linewidth=2)

    plt.xlabel("Time (s)")
    plt.ylabel("Displacement (m)")
    plt.title(f"Zoomed in Building Vibrations (N = {N} floors)")
    plt.legend(loc='best', ncol=2 if N > 5 else 1)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"Q2a Masked vibrations Full N = {N}.png")
    plt.show()

#Part B
print("___________________Part B ________________________")
N = 3
matrix = k_m * matrix_set_up(N)
eigenvalues, eigenvectors = np.linalg.eig(matrix)
print(eigenvectors[1])

freqs = []
for eigval in eigenvalues:
    freqs.append((np.sqrt(- eigval)/(2 * np.pi)))

N = 3
matrix = k_m * matrix_set_up(N)
eigenvalues, eigenvectors = np.linalg.eig(matrix)

freqs = []
for eigval in eigenvalues:
    freqs.append((np.sqrt(- eigval) / (2 * np.pi)))

dt = 0.001
T = 2.0
i = 0
while (i < N):
    x0s = eigenvectors[:, i].real
    v0s = np.zeros(N)

    t_test, x_test, v_test = verlet(matrix, dt, x0s, v0s, T)

    plt.figure(figsize=(10, 6))
    j = 0
    while j < N:
        plt.plot(t_test, x_test[:, j], label=f"Floor {j}", linewidth=2)
        j += 1

    plt.xlabel("Time (s)")
    plt.ylabel("Displacement (m)")
    plt.title(f"Normal Mode {i + 1}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"Q2b Normal Mode {i + 1}.png")
    plt.show()

    # finding peaks
    max_each_floor = np.argmax(np.abs(x0s))
    peaks, _ = find_peaks(x_test[:, max_each_floor])

    # Comparing
    Ts = np.diff(t_test[peaks])
    avg_T = np.mean(Ts)
    avg_freq = 1 / avg_T

    print(f"The freq calculated is {avg_freq}")
    print(f"The freq found through eigenvalues is {freqs[i]}")
    print(f"Relative diffrence {abs(freqs[i] - avg_freq) / freqs[i]}")

    i += 1

