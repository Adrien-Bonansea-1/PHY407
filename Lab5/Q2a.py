import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import rfft, rfftfreq

m = 1.0
k = 12.0
c = 3e8
v0 = 0.0

xc = c * np.sqrt(m / k)
initial_pos = [1, xc, 10 * xc]
omega0 = np.sqrt(k / m)
period = 2 * np.pi / omega0


def spring(x, v, dt, t_end):
    current_t = 0

    t_list, x_list, v_list = [], [], []

    t_list.append(current_t)
    x_list.append(x)
    v_list.append(v)

    while current_t <= t_end:
        u2_dot = -(k / m) * x * ((1 - (v / c) ** 2) ** (3 / 2))

        v = v + u2_dot * dt
        x = x + v * dt

        current_t += dt

        t_list.append(current_t)
        x_list.append(x)
        v_list.append(v)

    return np.array(t_list), np.array(x_list), np.array(v_list)

results = [[], [], []]
for i, x in enumerate(initial_pos):

    # choose dt and t_end depending on amplitude
    if x == 1:
        dt = period / 1000
        t_end = 20 * period
    elif x == xc:
        dt = period / 20000
        t_end = 80 * period
    else:
        dt = period / 40000
        t_end = 100 * period

    t_l, x_l, v_l = spring(x, v0, dt, t_end)

    results[0].append(t_l)
    results[1].append(x_l)
    results[2].append(v_l)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(t_l, x_l)
    plt.xlabel('Time (s)')
    plt.ylabel('Position (m)')
    plt.title(f'x0 = {x:.2e} m')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(t_l, v_l)
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (m/s)')
    plt.title(f'x0 = {x:.2e} m')
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(f"signal x = {x}.png")
    plt.show()

#____________________
#Fourier for x
compare_results = []

plt.figure(figsize=(12, 6))
i = 0
while(i < len(initial_pos)):
    x0 = initial_pos[i]
    t_l = results[0][i]
    x_l = results[1][i]

    cur_dt = t_l[1] - t_l[0]

    x_fft = np.fft.rfft(x_l)
    freqs = np.fft.rfftfreq(len(x_l), cur_dt)
    omega = 2 * np.pi * freqs
    amp = np.abs(x_fft)
    norm_amp = amp / np.max(amp)

    peak_idx = np.argmax(norm_amp[1:]) + 1
    omega_fft = omega[peak_idx]

    plt.plot(omega[1:], norm_amp[1:], linewidth=2, label=f"x₀ = {x0:.2e} m")

    i += 1

plt.xlabel('ω (rad/s)')
plt.ylabel('|x̂(ω)| / |x̂(ω)|ₘₐₓ')
plt.title('Normalized Fourier Spectrum of Position')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim(0, 10)
plt.ylim(0, 1.05)
plt.tight_layout()
plt.savefig("x signals fourier")
plt.show()

plt.figure(figsize=(12, 6))

i = 0
while(i < len(initial_pos)):
    x0 = initial_pos[i]
    t_l = results[0][i]
    v_l = results[2][i]

    cur_dt = t_l[1] - t_l[0]

    v_fft = np.fft.rfft(v_l)
    freqs = np.fft.rfftfreq(len(v_l), cur_dt)
    omega = 2 * np.pi * freqs
    amp = np.abs(v_fft)
    norm_amp = amp / np.max(amp)

    peak_idx = np.argmax(norm_amp[1:]) + 1
    omega_fft = omega[peak_idx]

    plt.plot(omega[1:], norm_amp[1:], linewidth=2, label=f"x₀ = {x0:.2e} m")

    i += 1

plt.xlabel('ω (rad/s)')
plt.ylabel('|v̂(ω)| / |v̂(ω)|ₘₐₓ')
plt.title('Normalized Fourier Spectrum of Velocity')
#plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim(0, 10)
plt.ylim(0, 1.05)
plt.tight_layout()
plt.savefig("v signals fourier")
plt.show()

plt.figure(figsize=(12, 6))
