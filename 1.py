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
