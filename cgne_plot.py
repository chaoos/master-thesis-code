#!/usr/bin/env python3

import matplotlib.pyplot as plt
import matplotlib as matplotlib
import numpy as np
import sys # sys.stdin
import json # json.dump(), json.load()
import glob
import os
import re
import math
sys.path.append('../python/libs/')
import latex as lt

# rows are (last row is the float in decimal representation)
# sign binary_exp binary_mantissa float_in_%.10e
# examples:
# 1 10000000 01100110000110001101110 -2.7976336479e+00
# 0 01111100 01010001000001010100100 1.6456085443e-01

# 4x4x4x4 lattice and periodic bc'c
plot_path = "../document/master-thesis/plots/cgne_final_{0}.pdf"
nrows = 3#len(config)
ncols = 2 #4
special = 80
middle = 800
mk_amounts = True
ext = 0
xticks = [0, 20, 40, 60, 80, 165, 250]
config = {
    0: {
        'name': r'naive, $4^4$',
        'xmax': 270,
        'res': 1e-6,
        'tol': 7.838367176906168e-05,
        'markers': ['x', '+', '+', 'x', '+', '+', '+', 'x'],
        'reset_markers': [3, 3, 3, 3, 3, 3, 3, 3, 3],
        'markersize': [6, 8, 8, 6, 8, 8, 8, 6],
        'files': [
            'pre_processing/cgne_final_4x4x4x4_1rank_ctype_p8_res_1e-6_posit8.json',
            'pre_processing/cgne_final_4x4x4x4_1rank_ctype_f64_res_1e-6_binary64.json',
            'pre_processing/cgne_final_4x4x4x4_1rank_ctype_f32_res_1e-6_binary32.json',
            'pre_processing/cgne_final_4x4x4x4_1rank_ctype_p32_res_1e-6_posit32.json',
            'pre_processing/cgne_final_4x4x4x4_1rank_ctype_tf32_res_1e-6_tfloat32.json',
            'pre_processing/cgne_final_4x4x4x4_1rank_ctype_f16_res_1e-6_binary16.json',
            'pre_processing/cgne_final_4x4x4x4_1rank_ctype_bf16_res_1e-6_bfloat16.json',
            'pre_processing/cgne_final_4x4x4x4_1rank_ctype_p16_res_1e-6_posit16.json',
        ],
        'names': [
            'posit8',
            'binary64',
            'binary32',
            'posit32',
            'tfloat32',
            'binary16',
            'bfloat16',
            'posit16',
        ],
        'colors': [
            lt.PYCOLORS['cyellow'],
            lt.PYCOLORS['cbrown'],
            lt.PYCOLORS['cred'],
            lt.PYCOLORS['cgreen'],
            lt.PYCOLORS['cpurple'],
            lt.PYCOLORS['cblue'],
            lt.PYCOLORS['cturquois'],
            lt.PYCOLORS['cpink'],
        ],
        'labels': [
            r'posit8, s={status} (+{reset})',
            r'binary64, s={status} (+{reset})',
            r'binary32, s={status} (+{reset})',
            r'posit32, s={status} (+{reset})',
            r'tensorfloat32, s={status} (+{reset})',
            r'binary16, s={status} (+{reset})',
            r'bfloat16, s={status} (+{reset})',
            r'posit16, s={status} (+{reset})',
        ],
        'legend_order': [1, 2, 3, 4, 5, 6, 7, 0, 8],
        'legend_loc': 'lower right',
        'inline_legend': [],
    },
    1: {
        'name': r'naive+quire, $4^4$',
        'xmax': 270,
        'res': 1e-6,
        'tol': 7.838367176906168e-05,
        'markers': ['+', '+', 'x', '+', '+', '+', 'x', '4', 'x'],
        'reset_markers': [3, 3, 3, 3, 3, 3, 3, 3, 3],
        'markersize': [8, 8, 6, 8, 8, 8, 6, 12, 6],
        'files': [
            'pre_processing/cgne_final_4x4x4x4_1rank_ctype_f64_res_1e-6_binary64.json',
            'pre_processing/cgne_final_4x4x4x4_1rank_ctype_f32_res_1e-6_binary32.json',
            'pre_processing/cgne_final_4x4x4x4_1rank_ctype_q32_res_1e-6_posit32.json',
            'pre_processing/cgne_final_4x4x4x4_1rank_ctype_tf32_res_1e-6_tfloat32.json',
            'pre_processing/cgne_final_4x4x4x4_1rank_ctype_f16_res_1e-6_binary16.json',
            'pre_processing/cgne_final_4x4x4x4_1rank_ctype_bf16_res_1e-6_bfloat16.json',
            'pre_processing/cgne_final_4x4x4x4_1rank_ctype_q16_res_1e-6_posit16.json',
            'pre_processing/cgne_final_4x4x4x4_1rank_ctype_q16_res_1e-5_posit16.json',
            'pre_processing/cgne_final_4x4x4x4_1rank_ctype_q8_res_1e-6_posit8.json',
        ],
        'names': [
            'binary64',
            'binary32',
            'posit32',
            'tfloat32',
            'binary16',
            'bfloat16',
            'posit16',
            'posit16',
            'posit8',
        ],
        'colors': [
            lt.PYCOLORS['cbrown'],
            lt.PYCOLORS['cred'],
            lt.PYCOLORS['cgreen'],
            lt.PYCOLORS['cpurple'],
            lt.PYCOLORS['cblue'],
            lt.PYCOLORS['cturquois'],
            lt.PYCOLORS['cpink'],
            lt.PYCOLORS['clightpink'],
            lt.PYCOLORS['cyellow'],
        ],
        'labels': [
            r'binary64, s={status} (+{reset})',
            r'binary32, s={status} (+{reset})',
            r'posit32, s={status} (+{reset})',
            r'tensorfloat32, s={status} (+{reset})',
            r'binary16, s={status} (+{reset})',
            r'bfloat16, s={status} (+{reset})',
            r'posit16, s={status} (+{reset})',
            r'posit16, res=1e-5, s={status} (+{reset})',
            r'posit8, s={status} (+{reset})',
        ],
        'legend_loc': 'upper right',
        'inline_legend': [8],
    },
    2: {
        'name': r'col=binary64, $4^4$',
        'xmax': 270,
        'res': 1e-6,
        'tol': 7.838367176906168e-05,
        'markers': ['x', '+', '+', 'x', '+', '+', '+', 'x'],
        'reset_markers': [3, 3, 3, 3, 3, 3, 3, 3, 3],
        'markersize': [6, 8, 8, 6, 8, 8, 8, 6],
        'files': [
            'pre_processing/cgne_final_4x4x4x4_1rank_ctype_f64_res_1e-6_posit8.json',
            'pre_processing/cgne_final_4x4x4x4_1rank_ctype_f64_res_1e-6_binary64.json',
            'pre_processing/cgne_final_4x4x4x4_1rank_ctype_f64_res_1e-6_binary32.json',
            'pre_processing/cgne_final_4x4x4x4_1rank_ctype_f64_res_1e-6_posit32.json',
            'pre_processing/cgne_final_4x4x4x4_1rank_ctype_f64_res_1e-6_tfloat32.json',
            'pre_processing/cgne_final_4x4x4x4_1rank_ctype_f64_res_1e-6_binary16.json',
            'pre_processing/cgne_final_4x4x4x4_1rank_ctype_f64_res_1e-6_bfloat16.json',
            'pre_processing/cgne_final_4x4x4x4_1rank_ctype_f64_res_1e-6_posit16.json',
        ],
        'names': [
            'posit8',
            'binary64',
            'binary32',
            'posit32',
            'tfloat32',
            'binary16',
            'bfloat16',
            'posit16',
        ],
        'colors': [
            lt.PYCOLORS['cyellow'],
            lt.PYCOLORS['cbrown'],
            lt.PYCOLORS['cred'],
            lt.PYCOLORS['cgreen'],
            lt.PYCOLORS['cpurple'],
            lt.PYCOLORS['cblue'],
            lt.PYCOLORS['cturquois'],
            lt.PYCOLORS['cpink'],
        ],
        'labels': [
            r'posit8, s={status} (+{reset})',
            r'binary64, s={status} (+{reset})',
            r'binary32, s={status} (+{reset})',
            r'posit32, s={status} (+{reset})',
            r'tensorfloat32, s={status} (+{reset})',
            r'binary16, s={status} (+{reset})',
            r'bfloat16, s={status} (+{reset})',
            r'posit16, s={status} (+{reset})',
        ],
        'legend_order': [1, 2, 3, 4, 5, 6, 7, 0, 8],
        'legend_loc': 'lower right',
        'inline_legend': [],
    },
    3: {
        'name': r'col=binary64, $4^4$',
        'xmax': 270,
        'res': 1e-12,
        'tol': 7.838367176906168e-11,
        'markers': ['+', '+', 'x', '4', '+', '+', '+', 'x', 'x'],
        'markersize': [8, 8, 6, 12, 8, 8, 8, 6, 6],
        'reset_markers': [3, 3, 3, 3, 3, 3, 3, 3, 3],
        'files': [
            'pre_processing/cgne_final_4x4x4x4_1rank_ctype_f64_res_1e-12_posit8.json',
            'pre_processing/cgne_final_4x4x4x4_1rank_ctype_f64_res_1e-12_binary64.json',
            'pre_processing/cgne_final_4x4x4x4_1rank_ctype_f64_res_1e-12_binary32.json',
            'pre_processing/cgne_final_4x4x4x4_1rank_ctype_f64_res_1e-12_posit32.json',
            'pre_processing/cgne_final_4x4x4x4_1rank_ctype_q32_res_1e-12_posit32.json',
            'pre_processing/cgne_final_4x4x4x4_1rank_ctype_f64_res_1e-12_tfloat32.json',
            'pre_processing/cgne_final_4x4x4x4_1rank_ctype_f64_res_1e-12_binary16.json',
            'pre_processing/cgne_final_4x4x4x4_1rank_ctype_f64_res_1e-12_bfloat16.json',
            'pre_processing/cgne_final_4x4x4x4_1rank_ctype_f64_res_1e-12_posit16.json',
        ],
        'names': [
            'posit8',
            'binary64',
            'binary32',
            'posit32',
            'posit32',
            'tfloat32',
            'binary16',
            'bfloat16',
            'posit16',
        ],
        'colors': [
            lt.PYCOLORS['cyellow'],
            lt.PYCOLORS['cbrown'],
            lt.PYCOLORS['cred'],
            lt.PYCOLORS['cgreen'],
            lt.PYCOLORS['clightgreen'],
            lt.PYCOLORS['cpurple'],
            lt.PYCOLORS['cblue'],
            lt.PYCOLORS['cturquois'],
            lt.PYCOLORS['cpink'],
        ],
        'labels': [
            r'posit8, s={status} (+{reset})',
            r'binary64, s={status} (+{reset})',
            r'binary32, s={status} (+{reset})',
            r'posit32, s={status} (+{reset})',
            r'posit32, quire32, s={status} (+{reset})',
            r'tensorfloat32, s={status} (+{reset})',
            r'binary16, s={status} (+{reset})',
            r'bfloat16, s={status} (+{reset})',
            r'posit16, s={status} (+{reset})',
        ],
        'legend_order': [1, 2, 3, 4, 5, 6, 7, 8, 0, 9],
        'legend_loc': ['upper right', 'lower left', 'upper right', 'lower left'],
        'inline_legend': [],
    },
}

"""
# 8x8x8x8 lattice and sf bc'c
plot_path = "../document/master-thesis/plots/cgne_8x8x8x8_new.pdf"
nrows = 4 #len(config)
ncols = 3 #4
special = 80
middle = 800
mk_amounts = True
xticks = [0, 20, 40, 60, 80, 165, 250]
ext = 0
config = {
    0: {
        'name': r'col=binary64, $8^4$',
        'xmax': 270,
        'res': 1e-6,
        'tol': 0.0003135349005339143,
        'markers': ['+', 'x', 'x', '+', 'x', 'x'],
        'reset_markers': [3, 3, 3, 3, 3, 3],
        'markersize': [8, 8, 8, 8, 8, 8],
        'files': [
            'pre_processing/cgne_final_8x8x8x8_1rank_ctype_f64_res_1e-6_binary64.json',
            'pre_processing/cgne_final_8x8x8x8_1rank_ctype_f64_res_1e-6_binary32.json',
            'pre_processing/cgne_final_8x8x8x8_1rank_ctype_f64_res_1e-6_binary16.json',
        ],
        'names': [
            'binary64',
            'binary32',
            'binary16',
        ],
        'colors': [
            lt.PYCOLORS['cbrown'],
            lt.PYCOLORS['cred'],
            lt.PYCOLORS['cblue'],
        ],
        'labels': [
            r'binary64, s={status} (+{reset})',
            r'binary32, s={status} (+{reset})',
            r'binary16, s={status} (+{reset})',
        ],
        'inline_legend': [],
    },
    1: {
        'name': r'col=binary64, $8^4$',
        'xmax': 270,
        'res': 1e-12,
        'tol': 3.135345084825413e-10,
        'markers': ['+', 'x', 'x', '+', 'x', 'x'],
        'reset_markers': [3, 3, 3, 3, 3, 3],
        'markersize': [8, 8, 8, 8, 8, 8],
        'files': [
            'pre_processing/cgne_final_8x8x8x8_1rank_ctype_f64_res_1e-12_binary64.json',
            'pre_processing/cgne_final_8x8x8x8_1rank_ctype_f64_res_1e-12_binary32.json',
            'pre_processing/cgne_final_8x8x8x8_1rank_ctype_f64_res_1e-12_binary16.json',
        ],
        'names': [
            'binary64',
            'binary32',
            'binary16',
        ],
        'colors': [
            lt.PYCOLORS['cbrown'],
            lt.PYCOLORS['cred'],
            lt.PYCOLORS['cblue'],
        ],
        'labels': [
            r'binary64, s={status} (+{reset})',
            r'binary32, s={status} (+{reset})',
            r'binary16, s={status} (+{reset})',
        ],
        'inline_legend': [],
    },
}


# 4x4x4x4 lattice and sf bc'c, e=1
plot_path = "../document/master-thesis/plots/cgne_4x4x4x4_e1.pdf"
nrows = 4 #len(config)
ncols = 3 #4
special = 80
middle = 800
mk_amounts = True
ext = 1
xticks = [0, 20, 40, 60, 80, 165, 250]
config = {
    0: {
        'name': r'col=binary64, $4^4$',
        'xmax': 270,
        'res': 1e-6,
        'tol': 7.838367176906168e-05,
        'markers': ['x', 'x', '+', 'x', 'x', 'x', '+'],
        'reset_markers': [3, 3, 3, 3, 3, 3, 3],
        'markersize': [6, 6, 8, 6, 6, 6, 8],
        'files': [
            'pre_processing/cgne_final_4x4x4x4_1rank_ctype_f64_res_1e-6_binary64_e1.json',
            'pre_processing/cgne_final_4x4x4x4_1rank_ctype_f64_res_1e-6_binary32_e1.json',
            'pre_processing/cgne_final_4x4x4x4_1rank_ctype_f64_res_1e-6_posit32_e1.json',
            'pre_processing/cgne_final_4x4x4x4_1rank_ctype_f64_res_1e-6_tfloat32_e1.json',
            'pre_processing/cgne_final_4x4x4x4_1rank_ctype_f64_res_1e-6_binary16_e1.json',
            'pre_processing/cgne_final_4x4x4x4_1rank_ctype_f64_res_1e-6_bfloat16_e1.json',
            'pre_processing/cgne_final_4x4x4x4_1rank_ctype_f64_res_1e-6_posit16_e1.json',
        ],
        'names': [
            'binary64',
            'binary32',
            'posit32',
            'tfloat32',
            'binary16',
            'bfloat16',
            'posit16',
        ],
        'colors': [
            lt.PYCOLORS['cbrown'],
            lt.PYCOLORS['cred'],
            lt.PYCOLORS['cgreen'],
            lt.PYCOLORS['cpurple'],
            lt.PYCOLORS['cblue'],
            lt.PYCOLORS['cturquois'],
            lt.PYCOLORS['cpink'],
        ],
        'labels': [
            r'binary64, s={status} (+{reset})',
            r'binary32, s={status} (+{reset})',
            r'posit32, s={status} (+{reset})',
            r'tfloat32, s={status} (+{reset})',
            r'binary16, s={status} (+{reset})',
            r'bfloat16, s={status} (+{reset})',
            r'posit16, s={status} (+{reset})',
        ],
        'inline_legend': [],
    },
    1: {
        'name': r'col=binary64, $4^4$',
        'xmax': 270,
        'res': 1e-12,
        'tol': 7.838367176906167e-11,
        'markers': ['x', 'x', '+', 'x', 'x', 'x', '+'],
        'reset_markers': [3, 3, 3, 3, 3, 3, 3],
        'markersize': [6, 6, 8, 6, 6, 6, 8],
        'files': [
            'pre_processing/cgne_final_4x4x4x4_1rank_ctype_f64_res_1e-12_binary64_e1.json',
            'pre_processing/cgne_final_4x4x4x4_1rank_ctype_f64_res_1e-12_binary32_e1.json',
            'pre_processing/cgne_final_4x4x4x4_1rank_ctype_f64_res_1e-12_posit32_e1.json',
            'pre_processing/cgne_final_4x4x4x4_1rank_ctype_f64_res_1e-12_tfloat32_e1.json',
            'pre_processing/cgne_final_4x4x4x4_1rank_ctype_f64_res_1e-12_binary16_e1.json',
            'pre_processing/cgne_final_4x4x4x4_1rank_ctype_f64_res_1e-12_bfloat16_e1.json',
            'pre_processing/cgne_final_4x4x4x4_1rank_ctype_f64_res_1e-12_posit16_e1.json',
        ],
        'names': [
            'binary64',
            'binary32',
            'posit32',
            'tfloat32',
            'binary16',
            'bfloat16',
            'posit16',
        ],
        'colors': [
            lt.PYCOLORS['cbrown'],
            lt.PYCOLORS['cred'],
            lt.PYCOLORS['cgreen'],
            lt.PYCOLORS['cpurple'],
            lt.PYCOLORS['cblue'],
            lt.PYCOLORS['cturquois'],
            lt.PYCOLORS['cpink'],
        ],
        'labels': [
            r'binary64, s={status} (+{reset})',
            r'binary32, s={status} (+{reset})',
            r'posit32, s={status} (+{reset})',
            r'tfloat32, s={status} (+{reset})',
            r'binary16, s={status} (+{reset})',
            r'bfloat16, s={status} (+{reset})',
            r'posit16, s={status} (+{reset})',
        ],
        'inline_legend': [],
    },
}


# 8x8x8x8 lattice and sf bc'c, e=1
plot_path = "../document/master-thesis/plots/cgne_8x8x8x8_e1.pdf"
nrows = 4 #len(config)
ncols = 3 #4
special = 80
middle = 800
mk_amounts = True
ext = 1
xticks = [0, 20, 40, 60, 80, 165, 250]
config = {
    0: {
        'name': r'col=binary64, $8^4$',
        'xmax': 270,
        'res': 1e-6,
        'tol': 0.000313534687076247,
        'markers': ['+', 'x', 'x', '+', 'x', 'x'],
        'reset_markers': [3, 3, 3, 3, 3, 3],
        'markersize': [8, 8, 8, 8, 8, 8],
        'files': [
            'pre_processing/cgne_final_8x8x8x8_1rank_ctype_f64_res_1e-6_binary64_e1.json',
            'pre_processing/cgne_final_8x8x8x8_1rank_ctype_f64_res_1e-6_binary32_e1.json',
            'pre_processing/cgne_final_8x8x8x8_1rank_ctype_f64_res_1e-6_binary16_e1.json',
        ],
        'names': [
            'binary64',
            'binary32',
            'binary16',
        ],
        'colors': [
            lt.PYCOLORS['cbrown'],
            lt.PYCOLORS['cred'],
            lt.PYCOLORS['cblue'],
        ],
        'labels': [
            r'binary64, s={status} (+{reset})',
            r'binary32, s={status} (+{reset})',
            r'binary16, s={status} (+{reset})',
        ],
        'inline_legend': [],
    },
    1: {
        'name': r'col=binary64, $8^4$',
        'xmax': 270,
        'res': 1e-12,
        'tol': 3.13534687076247e-10,
        'markers': ['+', 'x', 'x', '+', 'x', 'x'],
        'reset_markers': [3, 3, 3, 3, 3, 3],
        'markersize': [8, 8, 8, 8, 8, 8],
        'files': [
            'pre_processing/cgne_final_8x8x8x8_1rank_ctype_f64_res_1e-12_binary64_e1.json',
            'pre_processing/cgne_final_8x8x8x8_1rank_ctype_f64_res_1e-12_binary32_e1.json',
            'pre_processing/cgne_final_8x8x8x8_1rank_ctype_f64_res_1e-12_binary16_e1.json',
        ],
        'names': [
            'binary64',
            'binary32',
            'binary16',
        ],
        'colors': [
            lt.PYCOLORS['cbrown'],
            lt.PYCOLORS['cred'],
            lt.PYCOLORS['cblue'],
        ],
        'labels': [
            r'binary64, s={status} (+{reset})',
            r'binary32, s={status} (+{reset})',
            r'binary16, s={status} (+{reset})',
        ],
        'inline_legend': [],
    },
}
"""

def format10tothepower(x):
    y = abs(x)
    if y == 0:
        return r'0'
    else:
        y = int(math.floor(math.log10(y)))
        return r'10^{{{0:+02}}}'.format(y)

def frexp10(x):
    exp = int(math.floor(math.log10(abs(x))))
    return x / 10**exp, exp

def convert(x):
    return np.nan if x == "unknown" else x

def mklegend(config_row, ax, plt, tol, idx):
    if 'legend_loc' in config_row:
        if isinstance(config_row['legend_loc'], list):
            loc = config_row['legend_loc'][idx]
        else:
            loc = config_row['legend_loc']
    else:
        loc = 'best'
    if 'legend_order' in config_row:
        handles, labels = ax.get_legend_handles_labels()
        if tol:
            plt.legend([handles[i] for i in config_row['legend_order']],[labels[i] for i in config_row['legend_order']], loc=loc)
        else:
            plt.legend([handles[i] for i in config_row['legend_order'][:-1]],[labels[i] for i in config_row['legend_order'][:-1]], loc=loc)
    else:
        plt.legend(loc=loc)

nmx = 256
fig = plt.figure()
plots, legend_plots, annots = [], [], []
plt_nr = 1
file_nr = 0

def forward(X):
    return np.array([((middle/special)*x) if x<=special else (x+(middle-special)) for x in X])                
def inverse(X):
    return np.array([((special/middle)*x) if x<=middle else (x-(middle-special)) for x in X])

for series in range(0,len(config)):
    plots, legend_plots, annots = [], [], []
    ax = plt.subplot(nrows, ncols, plt_nr)
    plt_nr = plt_nr+1
    ax.set_yscale('log')
    ax.set_xscale('function', functions=(forward, inverse))
    plt.xticks(xticks)
    ax.set_title('plot {0}{1}, {2}, nmx={3}, res={4}'.format(series+1, 'a', config[series]['name'], nmx, config[series]['res']))

    i=0
    for path in config[series]['files']:
        with open(path) as fh:
            j = json.load(fh)
            iterations = j["i"]
            y = np.array(j["Axmb"])
            status = j["status"]
            reset = j["reset"]

            label = config[series]['labels'][i].format(status = status, reset = len(j["reset"]))
            if i in config[series]['inline_legend']:
                legend_plots.append(plt.plot(iterations, y, config[series]['markers'][i], c=config[series]['colors'][i], mew=1, markersize=config[series]['markersize'][i], label=label)[0])
            else:
                plots.append(plt.plot(iterations, y, config[series]['markers'][i], c=config[series]['colors'][i], mew=1, markersize=config[series]['markersize'][i], label=label)[0])

            plots.append(plt.plot(reset, [config[series]['tol'] for i in reset], linestyle='None', marker=config[series]['reset_markers'][i], c=config[series]['colors'][i], mew=1, markersize=20)[0])
            i = i+1

    legend_plots.append(plt.axhline(y=config[series]['tol'], linestyle='dashed', lw=1, label='tol={0:.1e}'.format(config[series]['tol']), c='black'))
    plt.xlim(left=0, right=config[series]['xmax'])
    plt.xlabel(r'iteration $i$')
    plt.ylabel(r'norm of residue')
    loc = config[series]['legend_loc'] if 'legend_loc' in config[series] else 'upper right'
    mklegend(config[series], ax, plt, True, 0)

    ax = plt.subplot(nrows, ncols, plt_nr)
    plt_nr = plt_nr+1
    ax.set_yscale('log')
    ax.set_xscale('function', functions=(forward, inverse))
    plt.xticks(xticks)
    ax.set_title('plot {0}{1}, {2}, nmx={3}, res={4}'.format(series+1, 'b', config[series]['name'], nmx, config[series]['res']))

    i=0
    for path in config[series]['files']:
        with open(path) as fh:
            j = json.load(fh)
            iterations = j["i"]
            y = np.array(j["rn"])
            status = j["status"]
            reset = j["reset"]

            label = config[series]['labels'][i].format(status = status, reset = len(j["reset"]))
            plots.append(plt.plot(iterations, y, config[series]['markers'][i], c=config[series]['colors'][i], mew=1, markersize=config[series]['markersize'][i], label=label)[0])
            plots.append(plt.plot(reset, [config[series]['tol'] for i in reset], linestyle='None', marker=config[series]['reset_markers'][i], c=config[series]['colors'][i], mew=1, markersize=20)[0])
            i = i+1

    annots.append(plt.axhline(y=config[series]['tol'], linestyle='dashed', lw=1, label='tol={0:.1e}'.format(config[series]['tol']), c='black'))
    plt.xlim(left=0, right=config[series]['xmax'])
    plt.xlabel(r'iteration $i$')
    plt.ylabel(r'norm of recursive residue')
    if nrows == 2  and ncols == 2:
        mklegend(config[series], ax, plt, True, 1)

    ax = plt.subplot(nrows, ncols, plt_nr)
    plt_nr = plt_nr+1
    ax.set_yscale('log')
    ax.set_xscale('function', functions=(forward, inverse))
    plt.xticks(xticks)
    ax.set_title('plot {0}{1}, {2}, nmx={3}, res={4}'.format(series+1, 'c', config[series]['name'], nmx, config[series]['res']))

    i=0
    for path in config[series]['files']:
        with open(path) as fh:
            j = json.load(fh)
            iterations = j["i"][:-1]
            y = abs(np.array(j["Axmb"]) - np.array(j["rn"]))[:-1]
            status = j["status"]

            label = config[series]['labels'][i].format(status = status, reset = len(j["reset"]))
            plots.append(plt.plot(iterations, y, config[series]['markers'][i], c=config[series]['colors'][i], mew=1, markersize=config[series]['markersize'][i], label=label)[0])
            i = i+1

    plt.xlim(left=0, right=config[series]['xmax'])
    plt.ylim(bottom=10**(-18), top=10**14)
    plt.xlabel(r'iteration $i$')
    plt.ylabel(r'roundoff accumulation of the residual')
    if nrows == 2  and ncols == 2:
        mklegend(config[series], ax, plt, False, 2)

    ax = plt.subplot(nrows, ncols, plt_nr)
    plt_nr = plt_nr+1
    ax.set_yscale('log')
    ax.set_xscale('function', functions=(forward, inverse))
    plt.xticks(xticks)
    ax.set_title('plot {0}{1}, {2}, nmx={3}, res={4}'.format(series+1, 'd', config[series]['name'], nmx, config[series]['res']))

    i=0
    for path in config[series]['files']:
        with open(path) as fh:
            j = json.load(fh)
            iterations = j["i"]
            y = np.array([convert(x) for x in j["diAdip1"]], dtype=np.float64)

            status = j["status"]

            label = config[series]['labels'][i].format(status = status, reset = len(j["reset"]))
            plots.append(plt.plot(iterations, y, config[series]['markers'][i], c=config[series]['colors'][i], mew=1, markersize=config[series]['markersize'][i], label=label)[0])
            i = i+1

    plt.xlim(left=0, right=config[series]['xmax'])
    plt.xlabel(r'iteration $i$')
    plt.ylabel(r'$A$-orthogonality of directions')
    if nrows == 2  and ncols == 2:
        mklegend(config[series], ax, plt, False, 3)

    if mk_amounts:
        ax = plt.subplot(nrows, ncols, plt_nr)
        plt_nr = plt_nr+1
        #if ext == 1: ax.set_yscale('log')
        ax.set_xscale('function', functions=(forward, inverse))
        plt.xticks(xticks)
        ax.set_title('plot {0}{1}, {2}, nmx={3}, res={4}'.format(series+1, 'e', config[series]['name'], nmx, config[series]['res']))

        i=0
        for path in config[series]['files']:
            print(path)
            with open(path) as fh:
                j = json.load(fh)
                iterations = j["i"]
                for x in j["reset"]: iterations.remove(x)
                y = np.array(j["ai"]) if ext == 0 else np.abs(100*(np.array(j["approx_ai"]) - np.array(j["ai"]))/np.array(j["ai"]))
                status = j["status"]
                reset = j["reset"]
                if ext == 1:
                    maxai, minai = max(np.array(j["ai"])), min(np.array(j["ai"]))
                    minaiten = format10tothepower(minai)
                    maxaiten = format10tothepower(maxai)
                    if minaiten == maxaiten:
                        mainlabel = r'{0}, $\alpha \in O({1})$'.format(config[series]['names'][i], maxaiten)
                    else:
                        mainlabel = r'{0}, $\alpha \in \left[{1} - {2}\right]$'.format(config[series]['names'][i], minaiten, maxaiten)
                else:
                    mainlabel = None

                label = config[series]['labels'][i].format(status = status, reset = len(j["reset"]))
                if status > 0:
                    l, a, lines = 0, 0, 0
                    for r in reset:
                        if len(iterations[l+3-a:r-1-a]) > 2 and y[l+3-a:r-1-a].size != 0:
                            lines+=1
                        l, a = r, a+1
                    l, a, p = 0, 0, 0
                    if ext == 0:
                        for r in reset:
                            length = len(iterations[l+3-a:r-1-a])
                            if length > 2 and y[l+3-a:r-1-a].size != 0:
                                p+=1
                                m, b = np.polyfit(list(range(0, length)), y[l+3-a:r-1-a], 1)
                                if p == 1: maxm, minm = m, m
                                maxm = m if m>maxm else maxm
                                minm = m if m<minm else minm
                                maxexpm = int(math.floor(math.log10(abs(maxm))))
                                minexpm = int(math.floor(math.log10(abs(minm))))
                                if p == lines:
                                    if maxexpm == minexpm:
                                        coll_label = r'{0}, $f(x) = mx+b, m \in O(10^{{{1:+02}}})$'.format(config[series]['names'][i], maxexpm)
                                    else:
                                        coll_label = r'{0}, $f(x) = mx+b, m \in \left[10^{{{1:+02}}} - 10^{{{2:+02}}}\right]$'.format(config[series]['names'][i], maxexpm, minexpm)
                                else:
                                    coll_label = None
                                plots.append(plt.plot(iterations[l+3-a:r-1-a], [m*x+b for x in list(range(0, length))], c=config[series]['colors'][i], linewidth=1, label=coll_label)[0])
                            l, a = r, a+1

                    plots.append(plt.plot(iterations, y, config[series]['markers'][i], c=config[series]['colors'][i], mew=1, markersize=config[series]['markersize'][i], label=mainlabel)[0])
                    plt.legend()

                i = i+1

        i=0
        ymin = plt.gca().get_ylim()[0]
        for path in config[series]['files']:
            with open(path) as fh:
                j = json.load(fh)
                if j["status"] > 0:
                    plots.append(plt.plot(j["reset"], [ymin for x in j["reset"]], linestyle='None', marker=3, c=config[series]['colors'][i], mew=1, markersize=20)[0])
                i = i+1

        plt.xlim(left=0, right=config[series]['xmax'])
        #plt.ylim(bottom=0.04, top=0.065)
        plt.xlabel(r'iteration $i$')
        ylabel = r'amounts $\alpha_i$' if ext == 0 else r'relative error in amount $\alpha_i$ [%]'
        plt.ylabel(ylabel)

        ax = plt.subplot(nrows, ncols, plt_nr)
        plt_nr = plt_nr+1
        #ax.set_yscale('log')
        ax.set_xscale('function', functions=(forward, inverse))
        plt.xticks(xticks)
        ax.set_title('plot {0}{1}, {2}, nmx={3}, res={4}'.format(series+1, 'f', config[series]['name'], nmx, config[series]['res']))

        i=0
        for path in config[series]['files']:
            with open(path) as fh:
                j = json.load(fh)
                iterations = j["i"]
                for x in j["reset"]: iterations.remove(x)
                y = np.array(j["bi"])
                status = j["status"]

                label = config[series]['labels'][i].format(status = status, reset = len(j["reset"]))
                if status > 0:
                    plots.append(plt.plot(iterations, y, config[series]['markers'][i], c=config[series]['colors'][i], mew=1, markersize=config[series]['markersize'][i], label=label)[0])
                i = i+1

        plt.xlim(left=0, right=config[series]['xmax'])
        #plt.ylim(bottom=0.3, top=0.6)
        plt.xlabel(r'iteration $i$')
        plt.ylabel(r'amounts $\beta_i$')

    if plt_nr >= nrows*ncols:
        if nrows == 2 and ncols == 2:
            fig.set_size_inches(ncols*10, nrows*4.5, forward=True)
        elif nrows == 3 and ncols == 2:
            fig.set_size_inches(ncols*6, nrows*4.5, forward=True)
        else:
            fig.set_size_inches(20, nrows*4.5, forward=True)

        fig.tight_layout()
        plt.savefig(plot_path.format(file_nr), dpi=fig.dpi)
        fig = plt.figure()
        file_nr += 1
        plt_nr = 1
