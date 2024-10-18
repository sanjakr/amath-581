import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


def shoot2(y, x, beta):
    return [y[1], (x**2 - beta) * y[0]]


tol = 1e-6  # define a tolerance level
col = ['r', 'b', 'g', 'c', 'm', 'k']  # eigenfunc colors
L = 4
xp = [-L, L]
xshoot = np.linspace(xp[0], xp[1], L*2*10 + 1)
ys = []
evs = []

beta_start = 1  # beginning value of beta
for modes in range(1, 6):  # begin mode loop
    beta = beta_start  # initial value of eigenvalue beta
    boundary_cond_left = [1, np.sqrt(L ** 2 - beta)]
    dbeta = beta / 100  # default step size in beta
    for i in range(1000):  # begin convergence loop for beta
        y_sol = odeint(shoot2, boundary_cond_left, xshoot, args=(beta,))
        boundary_cond_right = y_sol[-1, 1] + np.sqrt(L ** 2 - beta) * y_sol[-1, 0]

        if abs(boundary_cond_right) < tol:  # check for convergence
            evs.append(beta)
            break  # get out of convergence loop

        if (-1) ** (modes + 1) * boundary_cond_right < 0:
            beta -= dbeta
        else:
            beta += dbeta / 2
            dbeta /= 2

    beta_start = beta + 2  # after finding eigenvalue, pick new start
    norm = np.trapezoid(y_sol[:, 0] * y_sol[:, 0], xshoot)  # calculate the normalization
    y_norm = np.abs(y_sol[:, 0]) / np.sqrt(norm)
    ys.append(y_norm)
    plt.plot(xshoot, y_norm, col[modes - 1])  # plot modes

plt.show()  # end mode loop

A1 = np.transpose(np.array(ys))
A2 = np.array(evs)
print(A1[0], A1[-1])
print(A2)
