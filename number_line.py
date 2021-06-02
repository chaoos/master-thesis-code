#!/usr/bin/env python3
#
# plots the number line of representable numbers

import numpy as np
import softposit as sp
import softfloat as sf
import floats as fl
import bitstring
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import sys # sys.path
sys.path.append('../python/libs/')
import latex as lt

N = 2**16

config = {
    0: {
        'dtype': fl.tfloat32,
        'n': 2**19,
        'label': 'tensorfloat32',
        'color': lt.PYCOLORS['cpurple'],
    },
    1: {
        'dtype': fl.bfloat16,
        'n': 2**16,
        'label': 'bfloat16',
        'color': lt.PYCOLORS['cturquois'],
    },
    2: {
        'dtype': sf.float16,
        'n': 2**16,
        'label': 'binary16',
        'color': lt.PYCOLORS['cblue'],
    },
    3: {
        'dtype': sp.posit16,
        'n': 2**16,
        'label': 'posit16',
        'color': lt.PYCOLORS['cpink'],
    },
}


def calcX(dtype, n):
    zero = dtype(0.0)
    #X = np.zeros(n, dtype=dtype)
    X = np.array([zero for i in range(0, n)])
    if dtype == fl.bfloat16:
        for i in range(0, n):
            ba = bitstring.BitArray(format(i*(2**16), '#034b'), length=32)
            ba.overwrite('0000000000000000', 16)
            X[i] = dtype(ba.float)
            #print(format(i*(2**16), '#034b'), X[i].v)
    elif dtype == fl.tfloat32:
        for i in range(0, n):
            ba = bitstring.BitArray(format(i*(2**13), '#034b'), length=32)
            ba.overwrite('0000000000000', 19)
            #print(format(i*(2**13), '#034b'), X[i].v)
            X[i] = dtype(ba.float)
    else:
        for i in range(0, n):
            X[i] = dtype(bits=i)
    return X

def issafe(X):
    return np.array([(not (x != x) and x != np.float64('inf') and x != np.float64('-inf')) for x in X], dtype=bool)

own_bins = np.logspace(-41,38, 1024)

#fig = plt.gcf()
fig, ax = plt.subplots()
fig.set_figheight(4)
fig.set_figwidth(16)
#fig.set_size_inches(8, 3)
#plt.figure(figsize=(8, 2))

for i in config:
    X = calcX(config[i]['dtype'], config[i]['n'])
    X = np.array([x for x in X[issafe(X)]], dtype=np.float64)
    print(config[i]['label'], len(X))
    n, bins, patches = plt.hist(X, bins=own_bins, density=False, facecolor=config[i]['color'], alpha=0.5, label=config[i]['label'])

#plt.yscale('log')
#plt.xscale('log')
ax.set_xscale('log', basex=2)
ax.set_yscale('log', basey=10)
ticks = np.array([2**(-126), 4**(-14), 2**(-15), 1, 2**16, 4**14, 2**127], dtype=np.float64)
#labels = np.array([int(np.log2(x)) for x in ticks])
plt.xticks(ticks)
plt.xlabel('value range')
plt.ylabel('# numbers')
plt.legend(loc='upper right')
fig.savefig("../document/master-thesis/plots/number_line.pdf")

