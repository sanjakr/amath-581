import numpy as np
from scipy.integrate import odeint
from scipy.integrate import RK45
import matplotlib.pyplot as plt


def shoot2(y, x, beta):
    return [y[1], (x**2 - beta) * y[0]]


tol = 1e-4  # define a tolerance level
col = ['r', 'b', 'g', 'c', 'm', 'k']  # eigenfunc colors
n0 = 1
A = -1
L = 4
# x0 = [1, np.sqrt(L*L - n0)]
xp = [-L, L]
xshoot = np.linspace(xp[0], xp[1], 81)
ys = []
evs = []

beta_start = n0  # beginning value of beta
for modes in range(1, 6):  # begin mode loop
    beta = beta_start  # initial value of eigenvalue beta
    x0 = [1, np.sqrt(L**2 - beta)]
    dbeta = beta / 100  # default step size in beta
    for i in range(1000):  # begin convergence loop for beta
        y_sol = odeint(shoot2, x0, xshoot, args=(beta,))
        # y = RK45(shoot2, xp[0], x0, xp[1], args=(n0,beta))
        cond = y_sol[-1, 1] + np.sqrt(L**2 - beta)*y_sol[-1, 0]

        if i == 1 or i == 999:
            print(cond)

        if abs(cond) < tol:  # check for convergence
            print(i)
            print(beta)  # write out eigenvalue
            evs.append(beta)
            break  # get out of convergence loop

        if (-1) ** (modes + 1) * cond > 0:
            beta -= dbeta
        else:
            beta += dbeta / 10
            dbeta /= 10

    print(beta)
    beta_start = beta + 2  # after finding eigenvalue, pick new start
    norm = np.trapezoid(y_sol[:, 0] * y_sol[:, 0], xshoot)  # calculate the normalization
    y_norm = np.abs(y_sol[:, 0]) / np.sqrt(norm)
    ys.append(y_norm)
    plt.plot(xshoot, y_norm, col[modes - 1])  # plot modes

plt.show()  # end mode loop

A1 = np.transpose(np.array(ys))
A2 = np.array(evs)
# print(A1)
print(A2)