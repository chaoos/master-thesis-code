#!/usr/bin/env python3

import matplotlib.pyplot as plt
import matplotlib as matplotlib
import matplotlib.colors as colors
import matplotlib.patches as patches
from copy import copy
import numpy as np
from scipy import stats
import sys # sys.stdin
import json # json.dump(), json.load()
import glob
import os
import re
sys.path.append('../python/libs/')
import latex as lt
import collections

directories = [
    'dfl_sap_post_processing/*conf6_0-8x8-30_8x8x8x8_k_0.15649*batch1*.json',
    'dfl_sap_post_processing/*conf6_0-8x8-30_8x8x8x8_k_0.05*batch1*.json'
]

matrices = [
    'conf6_0-8x8-30_8x8x8x8_k_0.15649',
    'conf6_0-8x8-30_8x8x8x8_k_0.05'
]

# no longer titles than the below ones
titles = [
    r'conf6_0-8x8-30, CPU, $k_c=0.15649$, $k=0.15649$, $vol=8^4$, $res=10^{-9}$, $n_{kv}=7$, $bs=4^4$',
    r'conf6_0-8x8-30, CPU, $k_c=0.15649$, $k=0.05$, $vol=8^4$, $res=10^{-9}$, $n_{kv}=7$, $bs=4^4$'
]

plot_path = "../document/master-thesis/plots/dfl_sap_gcr_{0}.pdf"

def get_setup(j):
    j2 = j.copy()
    del j2['rn'], j2['reset'], j2['duration'], j2['i'], j2['status']
    return json.dumps(j2)

def get_xlabel(j, fname):
    m = re.search('Ns([0-9]+)_nb([0-9]+)', fname) # Ns10_nb4
    Ns = int(m.group(1))
    nb = int(m.group(2))
    dim = Ns*nb
    bs = 0
    if nb == 2:
        bs = r'$8^3 4$'
    elif nb == 4:
        bs = r'$8^2 4^2$'
    elif nb == 8:
        bs = r'$8^2 4^2$'
    elif nb == 16:
        bs = r'$8 4^3$'
    elif nb == 32:
        bs = r'$4^4$'
    elif nb == 64:
        bs = r'$4^3 2$'
    elif nb == 128:
        bs = r'$4^2 2^2$'
        #({Ns},{nb})
    return '{dim}\n'.format(dim=dim) + r'({Ns},{nb})'.format(dim = dim, Ns = Ns, nb = nb, bs = bs)

def get_x(j, fname):
    m = re.search('Ns([0-9]+)_nb([0-9]+)', fname) # Ns10_nb4
    return int(m.group(1))*int(m.group(2))

def get_ylabel(j):
    return '({ncy},{nmr})'.format(**j)

def get_y(j):
    return j['ncy']*j['nmr']

for mat, directory, title in zip(matrices, directories, titles):

    x,y,z = {},{},{}
    xlabel,ylabel = {},{}
    for file in glob.glob(directory):
        with open(file) as fh:
            j = json.load(fh)
            setup = get_setup(j)
            if setup in z:
                z[setup].entend(j['duration'])
            else:
                z[setup] = j['duration']

            xlabel[setup] = {'label': get_xlabel(j, file), 'value': get_x(j, file)}
            ylabel[setup] = {'label': get_ylabel(j), 'value': get_y(j)}
            x[setup] = get_x(j, file)
            y[setup] = get_y(j)

    for setup in z: z[setup] = np.mean(z[setup])

    # sort the thing according to 'value'
    xlabel = {k: v for k, v in sorted(xlabel.items(), key=lambda item: item[1]['value'])}
    ylabel = {k: v for k, v in sorted(ylabel.items(), key=lambda item: item[1]['value'])}
    # extract only the labels/value (keep order)
    X_labels = [i['label'] for i in xlabel.values()]
    X_values = [i['value'] for i in xlabel.values()]
    Y_labels = [i['label'] for i in ylabel.values()]
    Y_values = [i['value'] for i in ylabel.values()]
    # unique the list while keeping order
    X_labels = list(dict.fromkeys(X_labels))
    X_values = list(dict.fromkeys(X_values))
    Y_labels = list(dict.fromkeys(Y_labels))
    Y_values = list(dict.fromkeys(Y_values))

    #Y_labels = ['(1,1)', '(1,2)', '(1,4)', '(2,4)', '(4,4)', '(4,6)', '(6,6)', '(4,20)', '(10,10)', '(20,20)']
    #Y_values = [1, 2, 4, 8, 16, 24, 36, 80, 100, 400]

    zmin = np.min(list(z.values()))
    zdefault = np.nan # values not set
    Z = np.empty((len(Y_values), len(X_values)))
    Z[:] = zdefault # fill with NaNs
    J = np.empty((len(Y_values), len(X_values)), dtype=dict)
    for file in glob.glob(directory):
        with open(file) as fh:
            j = json.load(fh)
            setup = get_setup(j)
            xi = X_values.index(get_x(j, file))
            yi = Y_values.index(get_y(j))
            Z[yi,xi] = z[setup]
            J[yi,xi] = j

    #for setup in z: print(x[setup], y[setup], z[setup])
    X = np.arange(0,len(X_values)+1)
    Y = np.arange(0,len(Y_values)+1)

    plt.figure()

    # marks the NaNs in color 'w' (white)
    cmap = plt.get_cmap('plasma') # viridis or plasma
    cmap.set_bad(color = 'w', alpha = 1.)
    
    Z = np.ma.masked_invalid(Z) # such that nan are no problem in the LogNorm

    #fig, ax = plt.subplots()
    plt.pcolormesh(X, Y, Z, cmap=cmap, norm=colors.LogNorm(vmin=zmin, vmax=Z.max()))
    plt.colorbar(label=r'runtime [s]')
    ax = plt.gca()
    ax.set_title(title, fontsize=10)

    # Loop over data dimensions and create text annotations.
    for i in range(len(X_values)):
        for j in range(len(Y_values)):
            if not np.isnan(Z[j,i]) and Z[j,i] != zdefault:
                status = J[j,i]['status']
                time = float(Z[j,i])
                annotation = '{0:.2g}s\n{1}'.format(time, status) if time < 1 else '{0:.3g}s\n{1}'.format(time, status)
                text = plt.text(i+0.5, j+0.5, annotation, ha="center", va="center", color="w")

    # put X_labels here if the labels should be (Ns,nb) instead of Ns*nb
    #plt.xticks(X[:-1]+0.5, X_values) # remove the last element and shift by a half element to the right
    plt.xticks(X[:-1]+0.5, X_labels)
    plt.yticks(Y[:-1]+0.5, Y_labels)
    plt.setp(ax.get_xticklabels(), fontsize=7, rotation=0) # rotates only the xticks
    plt.setp(ax.get_yticklabels(), fontsize=7, rotation=0) # rotates only the xticks
    plt.xlabel(r'dimension of deflation subspace, $\dim \Omega, (N_s, n_b)$', fontsize=10)
    plt.ylabel(r'amount of SAP-preconditioning, $(n_{cy}, n_{mr})$', fontsize=10)

    plt.tight_layout()
    plt.savefig(plot_path.format(mat))
    #plt.show()
