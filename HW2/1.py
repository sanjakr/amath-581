import numpy as np
from scipy.integrate import odeint
from scipy.integrate import RK45
import matplotlib.pyplot as plt


def shoot2(y, x, beta):
    return [y[1], (x**2 - beta) * y[0]]


tol = 1e-6  # define a tolerance level
col = ['r', 'b', 'g', 'c', 'm', 'k']  # eigenfunc colors
n0 = 1
A = -1
x0 = [0, A]
L = 4
xp = [-L, L]
xshoot = np.linspace(xp[0], xp[1], 100)
ys = []
evs = []

beta_start = n0  # beginning value of beta
for modes in range(1, 6):  # begin mode loop
    beta = beta_start  # initial value of eigenvalue beta
    dbeta = n0 / 100  # default step size in beta
    for i in range(1000):  # begin convergence loop for beta
        y_sol = odeint(shoot2, x0, xshoot, args=(beta,))
        # y = RK45(shoot2, xp[0], x0, xp[1], args=(n0,beta))

        if abs(y_sol[-1, 0] - 0) < tol:  # check for convergence
            print(i)
            print(beta)  # write out eigenvalue
            evs.append(beta)
            break  # get out of convergence loop

        if (-1) ** (modes + 1) * y_sol[-1, 0] > 0:
            beta -= dbeta
        else:
            beta += dbeta / 2
            dbeta /= 2

    beta_start = beta + 2  # after finding eigenvalue, pick new start
    norm = np.trapezoid(y_sol[:, 0] * y_sol[:, 0], xshoot)  # calculate the normalization
    y_norm = np.abs(y_sol[:, 0] / np.sqrt(norm))
    ys.append(y_norm)
    plt.plot(xshoot, y_norm, col[modes - 1])  # plot modes

plt.show()  # end mode loop

A1 = np.vstack(ys)
A2 = np.transpose(np.vstack(evs))
print(A1.shape)
print(A2)