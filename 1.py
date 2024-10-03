import numpy as np

# Defining f(x) and f'(x)


def f_x(x):
    return x*np.sin(3*x) - np.exp(x)


def fd_x(x):
    return np.sin(3*x) + 3*x*np.cos(3*x) - np.exp(x)


iters = []

# Newton-Raphson method

x_vals = [-1.6]   # Initial guess
i = 0
while True:
    cur_x = x_vals[i]
    i += 1
    f_c = f_x(cur_x)
    x_vals.append(cur_x - (f_x(cur_x) / fd_x(cur_x)))
    if np.abs(f_c) <= 10**(-6):
        break

print("Newton-Raphson method:")
print(f"Number of iterations: {i}")
iters.append(i)
print(f"Solution: {x_vals[-1]}")
A1 = np.array(x_vals)
np.save("A1.npy", A1)

# Bisection method

x_r = -0.4
x_l = -0.7
x_c_list = []

i = 0
while True:
    i += 1
    x_c = (x_l + x_r)/2
    x_c_list.append(x_c)
    f_c = f_x(x_c)
    if f_c > 0:
        x_l = x_c
    else:
        x_r = x_c
    if np.abs(f_c) <= 10 ** (-6):
        break

iters.append(i)
print("Bisection method:")
print(f"Number of iterations: {i}")
print(i)
print(f"Solution: {x_c_list[-1]}")
print(x_c_list[-1])
A2 = np.array(x_c_list)
np.save("A2.npy", A2)
A3 = np.array(iters)
np.save("A3.npy", A3)


A = np.array([[1, 2], [-1, 1]])
B = np.array([[2, 0], [0, 2]])
C = np.array([[2, 0, -3], [0, 0, -1]])
D = np.array([[1, 2], [2, 3], [-1, 0]])
x = np.array([1, 0])
y = np.array([0, 1])
z = np.array([1, 2, -1])

A4 = A + B
np.save("A4.npy", A4)
print(A4)
A5 = 3 * x - 4 * y
np.save("A5.npy", A5)
print(A5)
A6 = A.dot(x)
np.save("A6.npy", A6)
print(A6)
A7 = B.dot(x - y)
np.save("A7.npy", A7)
print(A7)
A8 = D.dot(x)
np.save("A8.npy", A8)
print(A8)
A9 = D.dot(y) + z
np.save("A9.npy", A9)
print(A9)
A10 = A.dot(B)
np.save("A10.npy", A10)
print(A10)
A11 = B.dot(C)
np.save("A11.npy", A11)
print(A11)
A12 = C.dot(D)
np.save("A12.npy", A12)
print(A12)
