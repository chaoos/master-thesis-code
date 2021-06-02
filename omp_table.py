#!/usr/bin/env python3

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
from uncertainties import ufloat

files = '../master-thesis-data/omp_data/*ault21*.log'

#time5_16x16x16x16_r128_th1_native_it10_i10_ault18
#Absolute time of 100 invocations of Dw():
#47912.434 micro sec
def get_json_from_log(data, fname):
    j = {}
    m = re.search('([0-9]+)x([0-9]+)x([0-9]+)x([0-9]+)', fname) # 16x16x16x16
    j['lattice'] = str(m.group(1) + "x" + m.group(2) + "x" + m.group(3) + "x" + m.group(4))
    m = re.search('it([0-9]+)', fname) # it100
    j['iterations'] = int(m.group(1))
    j['operator'] = 'omp' if "_omp_" in fname else 'native'
    m = re.search('r([0-9]+)', fname) # r64
    j['ranks'] = int(m.group(1))
    m = re.search('th([0-9]+)', fname) # th8
    j['threads'] = int(m.group(1))
    m = re.search('([0-9\.\-\+e]+) micro sec', data) # 47912.434 micro sec
    j['duration'] = float(m.group(1))
    return j

def get_setup(j):
    j2 = j.copy()
    del j2['duration']
    del j2['lattice']
    return json.dumps(j2)

z = {}
for file in glob.glob(files):
    with open(file) as fh:
        j = get_json_from_log(fh.read(), file)
        #print(setup)
        if j['lattice'] == '64x64x64x64':
            k = '64'
        elif j['lattice'] == '32x32x32x32':
            k = '32'
        elif j['lattice'] == '16x16x16x16':
            k = '16'

        setup = get_setup(j)
        if setup in z:
            if k in z[setup]:
                z[setup][k].append(j['duration'])
            else:
                z[setup][k] = [j['duration']]
        else:
            z[setup] = {}
            z[setup][k] = [j['duration']]

# sort according to setup
z = collections.OrderedDict(sorted(z.items()))

print(r"""\begin{tabular}{ |p{1.5cm}||p{1cm}|p{1cm}|p{2cm}|p{2cm}|p{2cm}| }
    \hline
    & & & $64^4$ & $32^4$ & $16^4$ \\
    operator & $n_r$ & $n_{th}$ & time [s] & time [ms] & time [ms] \\
    \hline""")

for key in z:
    setup = json.loads(key)
    if '64' in z[key]:
        z[key]['64'] = np.array(z[key]['64'])/1000000 # now its seconds
    if '32' in z[key]:
        z[key]['32'] = np.array(z[key]['32'])/1000 # now its milli seconds
    if '16' in z[key]:
        z[key]['16'] = np.array(z[key]['16'])/1000 # now its milli seconds

    setup['time64'] = np.mean(z[key]['64']) if '64' in z[key] else 0.0
    setup['std64'] = np.std(z[key]['64']) if '64' in z[key] else 0.0
    setup['result64'] = ufloat(setup['time64'], setup['std64']) if '64' in z[key] else ufloat(0,0)

    setup['time32'] = np.mean(z[key]['32']) if '32' in z[key] else 0.0
    setup['std32'] = np.std(z[key]['32']) if '32' in z[key] else 0.0
    setup['result32'] = ufloat(setup['time32'], setup['std32']) if '32' in z[key] else ufloat(0,0)

    setup['time16'] = np.mean(z[key]['16']) if '16' in z[key] else 0.0
    setup['std16'] = np.std(z[key]['16']) if '16' in z[key] else 0.0
    setup['result16'] = ufloat(setup['time16'], setup['std16']) if '16' in z[key] else ufloat(0,0)

    if setup['operator'] == 'omp':
        print(r"    {operator} & {ranks} & {threads} & ${result64:.3L}$ & ${result32:.3L}$ & ${result16:.3L}$ \\".format(**setup))
    else:
        print(r"    {operator} & {ranks} & N/A & ${result64:.3L}$ & ${result32:.3L}$ & ${result16:.3L}$ \\".format(**setup))

print(r"""    \hline
\end{tabular}
""")
