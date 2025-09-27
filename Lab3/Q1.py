#imports
from pylab import *
import numpy as np
from scipy.integrate import quad
import scipy.special as sc

# Set up
def integrand(x):
    #create integrand function for integral in question
    return 4/(1 + x**2)

#Use Scipy.integrate to calculate the reasult
sol, unc = quad(integrand, 0, 1)


def gaussxw(N):
    # Initial approximation to roots of the Legendre polynomial
    a = linspace(3, 4 * N - 1, N) / (4 * N + 2)
    x = cos(pi * a + 1 / (8 * N * N * tan(a)))
    epsilon = 1e-15
    delta = 1.0
    while delta > epsilon:
        p0 = ones(N, float)
        p1 = copy(x)
        for k in range(1, N):
            p0, p1 = p1, ((2 * k + 1) * x * p1 - k * p0) / (k + 1)
        dp = (N + 1) * (p0 - x * p1) / (1 - x * x)
        dx = p1 / dp
        x -= dx
        delta = max(abs(dx))
    # Calculate the weights
    w = 2 * (N + 1) * (N + 1) / (N * N * (1 - x * x) * dp * dp)
    return x, w


def gaussxwab(N, a, b):
    x, w = gaussxw(N)
    return 0.5 * (b - a) * x + 0.5 * (b + a), 0.5 * (b - a) * w


def plot_gaussxw(N):
    plt.figure(dpi=150)
    plt.bar(gaussxw(N)[0], gaussxw(N)[1], width=0.02)
    plt.grid()
    plt.title("For $N = {}$".format(N))
    plt.xlabel('position $x$')
    plt.ylabel('weight $w_k$')

##############
#Part A
##############
print("PART A i__________________________________")


def trap(N, a, b):
    h = (b - a) / N  # width of slice

    s = 0.5 * integrand(a) + 0.5 * integrand(b)  # the end bits
    for k in range(1, N):  # adding the interior bits
        s += integrand(a + k * h)
    return h * s


def simp(N, a, b):
    h = (b - a) / N  # width of slice
    odd_total = 0
    even_total = 0

    s = integrand(a) + integrand(b)  # the end bits
    for k in range(1, N, 2):  # adding the odd bits
        odd_total += integrand(a + k * h)
    for k in range(2, N, 2):  # adding the even bits
        even_total += integrand(a + k * h)
    s = s + 4 * odd_total + 2 * even_total

    return (h / 3) * s


def guass(N, a, b, int_func):
    x_vals, omega_vals = gaussxwab(N, a, b)
    i = 0
    total = 0
    while (i < len(x_vals)):
        total = total + (int_func(x_vals[i]) * omega_vals[i])
        i += 1
    return total


N = 8
a = 0
b = 1
Ns = []
traps = []
sims = []
guasses = []
while N <= 4096:
    Ns.append(N)
    print("____________________")
    print(f"for current N of: {N}")
    cur_trap = trap(N, a, b)
    traps.append(cur_trap)
    print(f"Trap = {cur_trap}")

    cur_sim = simp(N, a, b)
    sims.append(cur_sim)
    print(f"Simp = {cur_sim}")

    cur_guass = guass(N, a, b, integrand)
    guasses.append(cur_guass)
    print(f"Gauss = {cur_guass}")
    N = N * 2

print("PART A ii__________________________________")

trap_ers = []
sol = np.pi
for num in traps:
    trap_ers.append(np.abs((num - sol)/sol))
print(trap_ers)

print("_______________")

sims_ers = []
for num in sims:
    sims_ers.append(np.abs((num - sol)/sol))
print(sims_ers)

print("_______________")

guasss_ers = []
for num in guasses:
    guasss_ers.append(np.abs((num - sol)/sol))
print(guasss_ers)

plt.figure(figsize=(6,4))
plt.loglog(Ns[:-1], trap_ers[:-1], label="trap")
plt.loglog(Ns[:-1], sims_ers[:-1], label="sims")
plt.loglog(Ns[:-1], guasss_ers[:-1], label="guass")
plt.xlabel("N values")
plt.ylabel("errors")
plt.title("Log-Log Plot of different error methods")
plt.legend()
plt.grid(True, which="both", ls="--", linewidth=0.5)
plt.savefig("Q1a_errors.png")
plt.show()


def error_estimate(data):
    i = 0
    errors = []
    while i < len(data) - 1:
        errors.append(np.abs(data[i + 1] - data[i]))
        i += 1

    return errors


traps_estimate = error_estimate(traps)
sims_estimate = error_estimate(sims)
guasses_estimate = error_estimate(guasses)

print(len(traps_estimate))
print(len(trap_ers))

plt.figure(figsize=(6, 4))
plt.loglog(Ns[:-1], trap_ers[:-1], label="trap")
plt.loglog(Ns[:-1], sims_ers[:-1], label="sims")
plt.loglog(Ns[:-1], guasss_ers[:-1], label="guass")
plt.loglog(Ns[:-1], traps_estimate, '--', label="trap estimate")
plt.loglog(Ns[:-1], sims_estimate, '--', label="sims estimate")
plt.loglog(Ns[:-1], guasses_estimate, '--', label="guass estimate")

plt.xlabel("N")
plt.ylabel("errs")
plt.title("Log-Log Plot Example")
plt.legend()
plt.grid(True, which="both", ls="--", linewidth=0.5)

# Use log scale ticks
plt.xticks(Ns[:-1], labels=[str(n) for n in Ns[:-1]])  # places ticks at Ns values on the log axis

plt.savefig("Q1a_errors_and_estimates.png")
plt.show()

print("PART B i__________________________________")

#Scipy implementation
S, C = sc.fresnel(1)

#My implementation
def C_integral(u):
    return np.cos(0.5 * np.pi * u**2)

def S_integral(u):
    return np.sin(0.5 * np.pi * u**2)

def intensity(S, C):
    return (0.125 * ((0.5 + C)**2 + (0.5 + S)**2))


N = 50
x_vals = np.linspace(-5, 5, 100)
lmada = 1
z = 3

I_guass = []
I_scipy = []

for cur_x in x_vals:
    u = cur_x * np.sqrt(2 / (z * lmada))

    C2 = guass(N, 0, u, C_integral)
    S2 = guass(N, 0, u, S_integral)
    I_guass.append(intensity(C2, S2))

    S, C = sc.fresnel(u)
    I_scipy.append(intensity(S, C))

plt.figure(figsize=(6, 4))
plt.plot(x_vals, I_scipy, label="Scipy")
plt.xlabel("x values")
plt.ylabel("I")
plt.title("")
plt.legend()
plt.grid(True, which="both", ls="--", linewidth=0.5)
plt.savefig("Q1bi_intensities_scipy.png")
plt.show()

plt.figure(figsize=(6, 4))
plt.plot(x_vals, I_guass, label="Guass")
plt.xlabel("x values")
plt.ylabel("I")
plt.title("")
plt.legend()
plt.grid(True, which="both", ls="--", linewidth=0.5)
plt.savefig("Q1bi_intensities_guass.png")
plt.show()

plt.figure(figsize=(6, 4))
plt.plot(x_vals, I_scipy, label="SciPy")
plt.plot(x_vals, I_guass, '--', label="Gauss")
plt.xlabel("x values")
plt.ylabel("I")
plt.legend()
plt.grid(True, which="both", ls="--", linewidth=0.5)
plt.savefig("Q1bi_intensities_comparison.png")
plt.show()

i = 0
abs_ers = []

x_vals = np.linspace(-5, 5, 100)
lmada = 1
z = 3

while i <= len(I_guass) - 1:
    abs_ers.append(np.abs((I_scipy[i] - I_guass[i]) / I_scipy[i]))
    i += 1

plt.figure(figsize=(6,4))
plt.plot(x_vals, abs_ers, label="error")
plt.xlabel("x values")
plt.ylabel("error")
plt.title("")
plt.legend()
plt.grid(True, which="both", ls="--", linewidth=0.5)
plt.savefig("Q1a_intensities_err.png")
plt.show()

print("PART B ii__________________________________")

i = 3
N_list = []
while i <= 50:
    N_list.append(i)
    i += 1

x_vals = np.linspace(-5, 5, 100)
lmada = 1
z = 3
max_errors = []

for N in N_list:
    I_guass = []
    I_scipy = []

    for cur_x in x_vals:
        u = cur_x * np.sqrt(2 / (z * lmada))

        C2 = guass(N, 0, u, C_integral)
        S2 = guass(N, 0, u, S_integral)
        I_guass.append(intensity(C2, S2))

        S, C = sc.fresnel(u)
        I_scipy.append(intensity(S, C))
    i = 0
    abs_ers = []
    while i <= len(I_guass) - 1:
        abs_ers.append(np.abs((I_scipy[i] - I_guass[i]) / I_scipy[i]))
        i += 1

    max_errors.append(np.max(abs_ers))

plt.figure(figsize=(6,4))
plt.plot(N_list, max_errors, label="error")
plt.xlabel("N values")
plt.ylabel("max errors")
plt.title("")
plt.legend()
plt.grid(True, which="both", ls="--", linewidth=0.5)
plt.savefig("Q1b_intensities_max_error.png")
plt.show()

print("PART C __________________________________")

N = 12
x_list = np.linspace(-3, 10, 100)
z_list = np.linspace(1, 5, 100)
lmada = 2 #change y (wavelength) according toi the one we want to see

I_total = []
for z in z_list:
    I_x_guass = []
    for cur_x in x_list:
        u = cur_x * np.sqrt(2 / (z * lmada))

        C2 = guass(N, 0, u, C_integral)
        S2 = guass(N, 0, u, S_integral)
        I_x_guass.append(intensity(C2, S2))
    I_total.append(I_x_guass)

I_arr = np.array(I_total)

print(np.shape(I_arr))

plt.pcolormesh(x_list, z_list, I_arr, cmap='viridis', shading='auto')
plt.colorbar(label='Intensity values')
plt.xlabel('Position x (m)')
plt.ylabel('Distance z (m)')
plt.title(f'Diffraction Pattern (λ = {lmada} m)')
plt.savefig(f"Diffraction Pattern (λ = {lmada} m)")
plt.show()