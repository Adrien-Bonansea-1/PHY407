# SolveLinear.py
# Python module for PHY407
# Paul Kushner, 2015-09-26
# Modifications by Nicolas Grisouard, 2018-09-26
# This module contains useful routines for solving linear systems of equations.
# Based on gausselim.py from Newman
from numpy import empty
# The following will be useful for partial pivoting
from numpy import empty, copy
import numpy as np

def GaussElim(A_in, v_in):
    """Implement Gaussian Elimination. This should be non-destructive for input
    arrays, so we will copy A and v to
    temporary variables
    IN:
    A_in, the matrix to pivot and triangularize
    v_in, the RHS vector
    OUT:
    x, the vector solution of A_in x = v_in """
    # copy A and v to temporary variables using copy command
    A = copy(A_in)
    v = copy(v_in)
    N = len(v)

    for m in range(N):
        # Divide by the diagonal element
        div = A[m, m]
        A[m, :] /= div
        v[m] /= div

        # Now subtract from the lower rows
        for i in range( m +1, N):
            mult = A[i, m]
            A[i, :] -= mult *A[m, :]
            v[i] -= mult *v[m]

    # Backsubstitution
    # create an array of the same type as the input array
    x = empty(N, dtype=v.dtype)
    for m in range( N -1, -1, -1):
        x[m] = v[m]
        for i in range( m +1, N):
            x[m] -= A[m, i ] *x[i]
    return x


def PartialPivot(A_in, v_in):
    """ In this function, code the partial pivot (see Newman p. 222) """
    # copy A and v to temporary variables using copy command
    A = copy(A_in)
    v = copy(v_in)
    N = len(v)

    for m in range(N):
        pivot = np.argmax(np.abs(A[m:, m])) + m # add extra m
        if pivot != m:
            # switching
            A[m, :], A[pivot, :] = copy(A[pivot, :]), copy(A[m, :])
            v[m], v[pivot] = v[pivot], v[m]
        # Divide by the diagonal element
        div = A[m, m]
        A[m, :] /= div
        v[m] /= div

        # Now subtract from the lower rows
        for i in range( m +1, N):
            mult = A[i, m]
            A[i, :] -= mult *A[m, :]
            v[i] -= mult *v[m]

    # Backsubstitution
    # create an array of the same type as the input array
    x = empty(N, dtype=v.dtype)
    for m in range( N -1, -1, -1):
        x[m] = v[m]
        for i in range( m +1, N):
            x[m] -= A[m, i ] *x[i]
    return x

#Part A
print("Part A_____________________")

A = np.array([[2,  1,  4,  1], [3,  4, -1, -1], [1, -4,  1,  5], [2, -2,  1,  3]], dtype=float)
v = np.array([-4, 3, 9, 7], dtype=float)

print(PartialPivot(A, v))

#Part B
print("Part B_____________________")
from numpy.linalg import solve
import matplotlib.pyplot as plt
# example of how to use: x = solve(A, v)
import time
from numpy.random import rand

Ns = [5, 10, 25, 50, 75, 100, 150, 200, 250, 300, 400]

times_guass = []
times_pivot = []
times_LU = []

#error calc: mean of (v - Ax) as it should be v = Ax, so any deviations would be an error.
errs_gauss = []
errs_pivot = []
errs_LU = []

for N in Ns:
    # Pivot
    start1 = time.time()

    v = rand(N)
    A = rand(N, N)
    A = np.array(A)

    pivot_sol = PartialPivot(A, v)

    end1 = time.time()
    times_pivot.append(end1 - start1)
    errs_pivot.append(np.mean(np.abs(v - (A @ pivot_sol))))

    # Guass

    start2 = time.time()

    guass_sol = GaussElim(A, v)

    end2 = time.time()
    times_guass.append(end2 - start2)
    errs_gauss.append(np.mean(np.abs(v - (A @ guass_sol))))

    # LU
    start3 = time.time()

    LU_sol = solve(A, v)

    end3 = time.time()
    times_LU.append(end3 - start3)
    errs_LU.append(np.mean(np.abs(v - (A @ LU_sol))))


plt.figure()
plt.loglog(Ns, times_guass, 'o-', label="Gaussian elimination")
plt.loglog(Ns, times_pivot, 's-', label="Partial pivoting")
plt.loglog(Ns, times_LU, 'd-', label="LU")
plt.xlabel("Matrix size N")
plt.ylabel("Time taken(s)")
plt.legend()
plt.title("Time comparison")
plt.savefig("Q1b_times.png")
plt.show()


plt.figure()
plt.loglog(Ns, errs_gauss, 'o-', label="Gaussian elimination")
plt.loglog(Ns, errs_pivot, 's-', label="Partial Pivoting")
plt.loglog(Ns, errs_LU, 'd-', label="LU")
plt.xlabel("N")
plt.ylabel("Error")
plt.legend()
plt.title("Error comparison")
plt.savefig("Q1b_errors.png")
plt.show()

#Part C
print("PART C_____________________")
#for graph 1

#set up eq
R1 = 1e3
R2 = 2e3
R3 = 1e3
R4 = 2e3
R5 = 1e3
R6 = 2e3
C1 = 1e-6
C2 = 0.5e-6
x_plus = 3
omega = 1000


A_text = np.array([
    [1/R1 + 1/R4 + 1j*omega*C1,  -1j*omega*C1,  0],
    [-1j*omega*C1,  1/R2 + 1/R5 + 1j*omega*C1 + 1j*omega*C2,  -1j*omega*C2],
    [0,  -1j*omega*C2,  1/R3 + 1/R6 + 1j*omega*C2]], dtype=complex) #can be complex so we cant put float


v_text = np.array([x_plus/R1, x_plus/R2, x_plus/R3], dtype=complex) #can be complex so we cant put float

xs = PartialPivot(A_text, v_text)

V_amps = []
V_phase = []
for x in xs:
    V_amps.append(np.abs(x))
    x_rad = np.angle(x)
    V_phase.append(np.degrees(x_rad))

T = 2 * (2*np.pi/omega)
t = np.linspace(0, T, 5000)

Vs = []
for x in xs:
    Vs.append(np.real(x * np.exp(1j * omega * t)))

plt.plot(t, Vs[0], label='V1(t)')
plt.plot(t, Vs[1], label='V2(t)')
plt.plot(t, Vs[2], label='V3(t)')
plt.xlabel('Time (s)')
plt.ylabel('Voltage (V)')
plt.title('Voltage comparison')
plt.legend()
plt.savefig("Q1c_Vs.png")
plt.show()

#for graph 2:
#set up eq
R1 = 1e3
R2 = 2e3
R3 = 1e3
R4 = 2e3
R5 = 1e3

R6 = 2e3
C1 = 1e-6
C2 = 0.5e-6
x_plus = 3
omega = 1000

# L = R6/omega
# In the equation we have 1/i * omega * L
# We can cancel this to 1/i * R6

A_text_2 = np.array([
    [1/R1 + 1/R4 + 1j*omega*C1,  -1j*omega*C1,  0],
    [-1j*omega*C1,  1/R2 + 1/R5 + 1j*omega*C1 + 1j*omega*C2,  -1j*omega*C2],
    [0,  -1j*omega*C2,  1/R3 + 1/(1j * R6) + 1j*omega*C2]], dtype=complex) #can be complex so we cant put float


v_text_2 = np.array([x_plus/R1, x_plus/R2, x_plus/R3], dtype=complex) #can be complex so we cant put float

xs = PartialPivot(A_text_2, v_text_2)

V_amps = []
V_phase = []
for x in xs:
    V_amps.append(np.abs(x))
    x_rad = np.angle(x)
    V_phase.append(np.degrees(x_rad))

T = 2 * (2*np.pi/omega)
t = np.linspace(0, T, 5000)

Vs = []
for x in xs:
    Vs.append(np.real(x * np.exp(1j * omega * t)))

plt.plot(t, Vs[0], label='V1(t)')
plt.plot(t, Vs[1], label='V2(t)')
plt.plot(t, Vs[2], label='V3(t)')
plt.xlabel('Time (s)')
plt.ylabel('Voltage (V)')
plt.title('Voltage comparison with inductor')
plt.legend()
plt.legend()
plt.savefig("Q1c_Vs_2.png")
plt.show()