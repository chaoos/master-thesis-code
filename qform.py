#!/usr/bin/env python3

import matplotlib
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt

import sys
sys.path.append('../python/libs/')
import latex as lt

#print(lt.COLORS)

A = np.array([[3, 2],
              [2, 6]])
b = np.array([2, -8])
c = 0

def f(x):
    return 0.5*x.dot(A.dot(x)) - b.dot(x) + c

def F(X, Y):
    Z = np.zeros((len(X), len(X[0])))
    for i in range(len(X)):
        for j in range(len(X[i])):
            Z[i][j] = f(np.array([X[i][j], Y[i][j]]))
    return Z

# Steepest descent
def sd(x0, N, A, b):
    x = np.zeros((N,2))
    x[0] = x0
    for i in range(0, N-1):
        r = b - A.dot(x[i])
        alpha = r.dot(r)/r.dot(A.dot(r))
        x[i+1] = x[i] + alpha*r
    return x

# Conjugate Gradient
def cg(x0, N, A, b):
    x = np.zeros((N,2))
    x[0] = x0
    p = r = b - A.dot(x[0])
    for i in range(0, N-1):
        #print(i)
        r_old = r
        p_old = p
        alpha = r_old.dot(r_old)/p_old.dot(A.dot(p_old))
        r = r_old - alpha*A.dot(p_old)
        beta = r.dot(r)/r_old.dot(r_old)
        p = r + beta*p_old
        x[i+1] = x[i] + alpha*p_old
    return x

analytic_zero = np.array([2.0, -2.0])
minimum = f(analytic_zero)
levels = [-9.9, -9, -7, 0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140]
delta = 0.1
xi = np.arange(-4.0, 6.1, delta)
yi = np.arange(-6.0, 4.1, delta)
X, Y = np.meshgrid(xi, yi)
Z = F(X, Y)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
CS = ax1.contour(X, Y, Z, levels=levels, cmap='coolwarm')
ax1.clabel(CS, inline=1, fmt='%1.0f', fontsize=10)
ax1.grid(c='k', ls='-', alpha=0.3)
ax1.plot(analytic_zero[0], analytic_zero[1], '.', color=[0, 0, 1]) 
CS = ax2.contour(X, Y, Z, levels=levels, cmap='coolwarm')
ax2.clabel(CS, inline=1, fmt='%1.0f', fontsize=10)
ax2.grid(c='k', ls='-', alpha=0.3)
ax2.plot(analytic_zero[0], analytic_zero[1], '.', color=[0, 0, 1]) 

config = {
    0: {
        'N': 10,
        'col': 'r-',
        'label': r'SD: "good" initial guess $\vec{x}_0 = (-3, 0.5)^T$',
        'x0': np.array([-3, 0.5]),
        'solv': sd,
        'ax': ax1,
    },
    1: {
        'N': 10,
        'col': 'g-',
        'label': r'SD: "bad" initial guess $\vec{x}_0 = (-3, -3)^T$',
        'x0': np.array([-3, -3]),
        'solv': sd,
        'ax': ax1,
    },
    2: {
        'N': 10,
        'col': 'm-',
        'label': r'SD: "bad" initial guess $\vec{x}_0 = (-3, 3)^T$',
        'x0': np.array([-3, 3]),
        'solv': sd,
        'ax': ax1,
    },
    3: {
        'N': 10,
        'col': 'y-',
        'label': r'SD: "bad" initial guess $\vec{x}_0 = (5, -5)^T$',
        'x0': np.array([5, -5]),
        'solv': sd,
        'ax': ax1,
    },
    5: {
        'N': 3,
        'col': 'r-',
        'label': r'CG: initial guess $\vec{x}_0 = (-3, 0.5)^T$',
        'x0': np.array([-3, 0.5]),
        'solv': cg,
        'ax': ax2,
    },
    6: {
        'N': 3,
        'col': 'g-',
        'label': r'CG: initial guess $\vec{x}_0 = (-3, -3)^T$',
        'x0': np.array([-3, -3]),
        'solv': cg,
        'ax': ax2,
    },
    7: {
        'N': 3,
        'col': 'm-',
        'label': r'CG: initial guess $\vec{x}_0 = (-3, 3)^T$',
        'x0': np.array([-3, 3]),
        'solv': cg,
        'ax': ax2,
    },
    8: {
        'N': 3,
        'col': 'y-',
        'label': r'CG: initial guess $\vec{x}_0 = (5, -5)^T$',
        'x0': np.array([5, -5]),
        'solv': cg,
        'ax': ax2,
    },
}

for i in config:
    solv = config[i]['solv']
    x0 = config[i]['x0']
    col = config[i]['col']
    label = config[i]['label']
    ax = config[i]['ax']
    N = config[i]['N']
    x = solv(x0, N, A, b)

    for i in range(0, N-1):
        if i == 0:
            ax.plot(x[i:i+2,0], x[i:i+2,1], col, label=label)
        else:
            ax.plot(x[i:i+2,0], x[i:i+2,1], col)


ax1.set_xlabel(r'$x_1$')
ax2.set_xlabel(r'$x_1$')
ax1.set_ylabel(r'$x_2$')
ax2.set_ylabel(r'$x_2$')
ax1.legend()
ax2.legend()

ax1.axhline(0, color='black')
ax2.axhline(0, color='black')
ax1.axvline(0, color='black')
ax2.axvline(0, color='black')

fig.tight_layout()

plt.savefig("../document/master-thesis/plots/qform_contour.pdf")
#plt.show()
