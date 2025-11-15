import numpy as np
import matplotlib.pyplot as plt

L = 1.0
d = 0.1
C = 1.0
sigma = 0.3
v = 100
h = 1e-6
N = 100

xs = np.linspace(0, L, N)
dx = xs[1] - xs[0]

phi = np.zeros(N)
psi = np.zeros(N)

def initialV(x):
    term1 = C * x * (L - x) / L**2
    term2 = np.exp(-(x - d)**2 / (2 * sigma**2))
    return term1 * term2

psi = initialV(xs)

save_times = [0.002, 0.004, 0.006, 0.012, 0.100]
save_tolerance = h * 10
saved_plots = {t: None for t in save_times}

t = 0
t_end = 0.1
frame_counter = 0
fig, axes = plt.subplots(3, 2, figsize=(12, 10))
axes = axes.flatten()

while t <= t_end:

    phi_old = phi.copy()
    psi_old = psi.copy()

    for i in range(1, N-1):
        d2phi_dx2 = (phi[i+1] - 2*phi[i] + phi[i-1]) / dx**2
        phi[i] = phi[i] + h * psi[i]
        psi[i] = psi[i] + h * v**2 * d2phi_dx2

    phi[0] = phi[-1] = 0
    psi[0] = psi[-1] = 0

    t += h
    frame_counter += 1

    for idx, save_time in enumerate(save_times):
        if saved_plots[save_time] is None and abs(t - save_time) < save_tolerance:
            saved_plots[save_time] = phi.copy()
            print(f"Saved frame at t = {t*1000:.3f} ms (target {save_time*1000:.1f} ms)")

            axes[idx].plot(xs, phi)
            axes[idx].set_xlabel("Position (m)")
            axes[idx].set_ylabel("Displacement")
            axes[idx].set_title(f"FTCS: t = {save_time*1000:.1f} ms")
            axes[idx].grid(True)
            fig_indiv = plt.figure(figsize=(6, 4))
            plt.plot(xs, phi)
            plt.xlabel("Position (m)")
            plt.ylabel("Displacement")
            plt.title(f"FTCS: t = {save_time*1000:.1f} ms")
            plt.grid(True)

            fname = f"FTCS_t_{save_time*1000:.1f}ms.png"
            plt.savefig(fname, dpi=300)
            plt.close(fig_indiv)

    if frame_counter % 10000 == 0:
        print(f"Progress: t = {t*1000:.2f} ms")

if len(save_times) % 2 == 1:
    fig.delaxes(axes[-1])

plt.tight_layout()
plt.show()

print("\nFTCS method complete!")
print(f"Total time steps: {frame_counter}")
