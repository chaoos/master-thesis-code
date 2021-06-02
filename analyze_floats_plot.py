#!/usr/bin/env python3
#
# plot script

import matplotlib.pyplot as plt
import numpy as np
import math
import sys # sys.stdin
import json # json.dump(), json.load()
import argparse
sys.path.append('../python/libs/')
import latex as lt

print(lt.PYCOLORS)

parser = argparse.ArgumentParser(description='Plot float stats.')
parser.add_argument('outfile', help='the output plot file name')
args = parser.parse_args()

j = json.load(sys.stdin)
exp = j["exp"]
num = j["num"]
zeros = j["zeros"]
n = j["n"]

percent = [100*v/n for v in num]
elower, eupper = min(exp), max(exp)
xmin, xmax = min(exp), max(exp)
while xmin % 5 != 0: xmin -= 1
while xmax % 5 != 0: xmax += 1

# mantissa * 2 ^exponent
print("highest and lowest apprearing expoenents:", elower, eupper)
print("should be 100.0:", sum(percent))

lowest_exp = -24
lowest_exp2 = -14
highest_exp = 16
ymax = math.ceil(max(percent))+1.5
a, b, c = 4.0, 3.5, 3.0

plt.plot(exp, percent, '+', c=lt.PYCOLORS['cred'], label='non-zero float exponents and their' "\n" r'occurences in % of all non-zero floats')

plt.axvline(x=lowest_exp, linestyle='-', label='lowest subnormal fp16 exp (x = {})'.format(lowest_exp), c=lt.PYCOLORS['cblue'])
plt.axvline(x=lowest_exp/2, linestyle='dashed', ymax=a/ymax, label='lowest subnormal fp16 exp before sq. (x = {})'.format(lowest_exp/2), c=lt.PYCOLORS['cblue'])

plt.axvline(x=lowest_exp2, linestyle='-', ymax=a/ymax, label='lowest regular fp16 exp (x = {})'.format(lowest_exp2), c=lt.PYCOLORS['cgreen'])
plt.axvline(x=lowest_exp2/2, linestyle='dashed', ymax=a/ymax, label='lowest regular fp16 exp before sq. (x = {})'.format(lowest_exp2/2), c=lt.PYCOLORS['cgreen'])

plt.axvline(x=highest_exp, linestyle='-', label='highest fp16 exp (x = {})'.format(highest_exp), c=lt.PYCOLORS['cred'])
plt.axvline(x=highest_exp/2, linestyle='dashed', ymax=a/ymax, label='highest fp16 exp before sq. (x = {})'.format(highest_exp/2), c=lt.PYCOLORS['cred'])

plt.axvspan(highest_exp/2, highest_exp+1, label='unsave region', color=lt.PYCOLORS['cyellow'], alpha=0.3, lw=0)
plt.xlim(left=xmin, right=highest_exp+1)
plt.ylim(top=ymax)

plt.annotate('', xy=(lowest_exp, a), xytext=(highest_exp, a), arrowprops={'arrowstyle': '<->'})
plt.annotate('', xy=(lowest_exp, c), xytext=(lowest_exp2, c), arrowprops={'arrowstyle': '<->'})
plt.annotate('', xy=(lowest_exp, c), xytext=(-60, c), arrowprops={'arrowstyle': '->'})
plt.annotate('', xy=(17, c), xytext=(highest_exp/2, c), arrowprops={'arrowstyle': '<-'})
plt.annotate('', xy=(lowest_exp2/2, c), xytext=(highest_exp/2, c), arrowprops={'arrowstyle': '<->'})

plt.annotate('fp16 representable range', xy=((highest_exp+lowest_exp)/2, a+0.05), ha='center', fontsize=6)
plt.annotate("fp16\nsubnormals", xy=((lowest_exp2+lowest_exp)/2, c+0.05), ha='center', fontsize=6)
plt.annotate("zero in fp16", xy=((-60+lowest_exp)/2, c+0.05), ha='center', fontsize=6)
plt.annotate("inf in fp16", xy=((highest_exp+highest_exp/2)/2, c+0.05), ha='center', fontsize=6)
plt.annotate("save in fp16", xy=((lowest_exp2/2+highest_exp/2)/2, c+0.05), ha='center', fontsize=6)

plt.xticks(np.arange(xmin, highest_exp+1, 5), fontsize=8)
plt.yticks(np.arange(0, ymax, 0.5), fontsize=8)

plt.xlabel('exponents')
plt.ylabel('occurence in %')

plt.title(r'$f = (-1)^s \cdot m \cdot 2^{e}$; ' + r'$e \in \{{{a}, ..., {b}\}}$; $m \in [1, 2)$; $s \in \{{0, 1\}}$' "\n" r'{z:d}/{n:d} ({p:2.2f}%) zeros'.format(
    a = elower,
    b = eupper,
    z = zeros,
    n = n,
    p = 100*zeros/n
), fontsize=8)
plt.legend(loc='upper left', prop={'size': 5})
plt.savefig(args.outfile)
#plt.show()
