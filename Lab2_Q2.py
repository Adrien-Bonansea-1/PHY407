import numpy as np
from scipy.integrate import quad
import time

#Part A
#Part i
print("Part i ______________________")
def integrand(x):
    #create integrand function for integral in question
    return 4/(1 + x**2)

#Use Scipy.integrate to calculate the result
sol, unc = quad(integrand, 0, 1)
#we should get pi as a sol
print(sol)

print("Part ii ______________________")

# Let's now evaluate the integral with Numpy
a = 0
b = 1
N = 4  # number of slices; try and increase it
h = (b-a)/N  # width of slice

s = 0.5*integrand(a) + 0.5*integrand(b)  # the end bits
for k in range(1,N):  # adding the interior bits
    s += integrand(a+k*h)

print("Trapezoidal rule =>", h*s)
print("relative dif = ", (np.abs((h*s) - sol)/sol))

# Let's now evaluate the integral with Numpy
N = 4  # number of slices; try and increase it
h = (b-a)/N  # width of slice
odd_total = 0
even_total = 0

s = integrand(a) + integrand(b) # the end bits
for k in range(1,N,2):  # adding the odd bits
    odd_total += integrand(a+k*h)
for k in range(2,N,2):  # adding the even bits
    even_total += integrand(a+k*h)
s = s + 4*odd_total + 2*even_total

print("Simpson's rule =>", (h/3)*s)
print("relative dif = ", (np.abs((h/3)*s - np.pi))/sol)

print("Part iii ______________________")

start_time = time.time()
smaller = False
x = 2
while (smaller == False):
    # error order calculation
    b = 1
    a = 0
    N = 2 ** x
    h = (b - a) / N
    error_order = h ** 2 / 12
    if 10 ** -9 >= error_order:  #increment and check until we have the error order we want
        smaller = True
    x += 1

print("We end up with an N of ", N, " and an x of ", x)
end_time = time.time()
print("time taken is = ", (end_time - start_time))

# Finding the exact n
# reverse the error order calculation equations to find the N and time it
b = 1
a = 0
h = (10 ** (-9) * 12) ** (1 / 2)
N = (b - a) / h
print("Trapezoidal's N =>", N)

print("----------")

start_time = time.time()
smaller = False
x = 2
while (smaller == False):
    # error order calculation
    b = 1
    a = 0
    N = 2 ** x
    h = (b - a) / N
    error_order = h ** 4 / 180
    if 10 ** -9 >= error_order: #increment and check until we have the error order we want
        smaller = True
    x += 1

print("We end up with an N of ", N, " and an x of ", x)
end_time = time.time()
print("time taken is = ", (end_time - start_time))

# finding exact N
# reverse the error order calculation equations to find the N
b = 1
a = 0
h = (1e-9 * 180) ** (1 / 4)
N = (b - a) / h
print("Sympson's N =>", N)

print("Part iv ______________________")

# N = 16
a = 0
b = 1
N = 16  # number of slices; try and increase it
h = (b - a) / N  # width of slice

s = 0.5 * integrand(a) + 0.5 * integrand(b)  # the end bits
for k in range(1, N):  # adding the interior bits
    s += integrand(a + k * h)

errN16 = h * s
print("Trapezoidal rule N = 16: =>", errN16)
print("abs error N = 16: => ", (np.abs(errN16 - sol)))

print("------------")

# N = 32
a = 0
b = 1
N = 32  # number of slices; try and increase it
h = (b - a) / N  # width of slice

s = 0.5 * integrand(a) + 0.5 * integrand(b)  # the end bits
for k in range(1, N):  # adding the interior bits
    s += integrand(a + k * h)

errN32 = h * s
print("Trapezoidal rule N = 32: =>", errN32)
print("abs error N = 32: => ", (np.abs(errN32 - sol)))

print("------------")

print('Practical estimation error')
print((errN32 - errN16) / 3)

print("------------")

# Check that the absolute error and estimation are the same
print(np.abs(((errN32 - errN16) / 3) - np.abs(errN32 - sol)))
print("this is so close to 0 that it can be considered negligible.")

