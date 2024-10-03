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
    if np.abs(f_c) <= 10**(-6):
        break
    x_vals.append(cur_x - (f_x(cur_x)/fd_x(cur_x)))

print("Newton-Raphson method:")
print(f"Number of iterations: {i}")
iters.append(i)
print(f"Solution: {x_vals[-1]}")
np.save("A1.npy", np.array(x_vals))

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
np.save("A2.npy", np.array(x_c_list))
np.save("A3.npy", np.array([iters]))


A = np.array([[1, 2], [-1, 1]])
B = np.array([[2, 0], [0, 2]])
C = np.array([[2, 0, -3], [0, 0, -1]])
D = np.array([[1, 2], [2, 3], [-1, 0]])
x = np.array([[1], [0]])
y = np.array([[0], [1]])
z = np.array([[1], [2], [-1]])

np.save("A4.npy", A + B)
print(A + B)
np.save("A5.npy", 3 * x - 4 * y)
print(3*x - 4*y)
np.save("A6.npy", A.dot(x))
print(A.dot(x))
np.save("A7.npy", B.dot(x - y))
print(B.dot(x - y))
np.save("A8.npy", D.dot(x))
print(D.dot(x))
np.save("A9.npy", D.dot(y) + z)
print(D.dot(y) + z)
np.save("A10.npy", A.dot(B))
print(A.dot(B))
np.save("A11.npy", B.dot(C))
print(B.dot(C))
np.save("A12.npy", C.dot(D))
print(C.dot(D))