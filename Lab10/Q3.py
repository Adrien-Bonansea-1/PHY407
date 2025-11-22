import numpy as np
import matplotlib.pyplot as plt


def integrand(x):
    return x ** (-0.5) / (1 + np.exp(x))


def mv_method(N):
    x = np.random.random(N)
    return np.mean(integrand(x))


def sample(N):
    x = np.random.random(N)
    w_x = x ** 2

    return np.mean(2 / (1 + np.exp(w_x)))


mean_res, sample_res = [], []
i = 0
N = 10000
while i <= 100:
    mean_res.append(mv_method(N))
    sample_res.append(sample(N))
    i += 1

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(mean_res, bins=10, range=[0.8, 0.88])
plt.title("Part A: Mean Value Method")
plt.xlabel("Integral Value")
plt.ylabel("Frequency")

plt.subplot(1, 2, 2)
plt.hist(sample_res, bins=10, range=[0.8, 0.88])
plt.title("Part A: Importance Sampling")
plt.xlabel("Integral Value")
plt.ylabel("Frequency")
plt.savefig('Part A: Importance Sampling')
plt.show()

def integrand2(x):
    return np.exp(-2 * np.abs(x - 5))

def mv_method2(N):
    x = np.random.uniform(0, 10, N)
    return 10 * np.mean(integrand2(x))

def sample2(N):
    x = np.random.normal(5, 1, N)
    w_x = (1/np.sqrt(2*np.pi)) * np.exp(-(x - 5)**2 / 2)

    return np.mean(integrand2(x) / w_x)

mean_res2, sample_res2 = [], []
N = 10000
i = 0
while i <= 100:
    mean_res2.append(mv_method2(N))
    sample_res2.append(sample2(N))
    i += 1

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(mean_res2, bins=10)
plt.title("Part A: Mean Value Method")
plt.xlabel("Integral Value")
plt.ylabel("Frequency")

plt.subplot(1, 2, 2)
plt.hist(sample_res2, bins=10)
plt.title("Part A: Importance Sampling")
plt.xlabel("Integral Value")
plt.ylabel("Frequency")
plt.savefig('Part B: Importance Sampling')

plt.tight_layout()
plt.show()

print(f"Mean Value Method: mean = {np.mean(mean_res2):.4f}, std = {np.std(mean_res2):.4f}")
print(f"Importance Sampling: mean = {np.mean(sample_res2):.4f}, std = {np.std(sample_res2):.4f}")


