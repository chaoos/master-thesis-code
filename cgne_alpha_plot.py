#!/usr/bin/env python3

import matplotlib.pyplot as plt
import matplotlib as matplotlib
import numpy as np
import sys # sys.stdin
import json # json.dump(), json.load()
import glob
import os
import re
sys.path.append('../python/libs/')
import latex as lt

# rows are (last row is the float in decimal representation)
# sign binary_exp binary_mantissa float_in_%.10e
# examples:
# 1 10000000 01100110000110001101110 -2.7976336479e+00
# 0 01111100 01010001000001010100100 1.6456085443e-01

# 4x4x4x4 lattice and periodic bc'c
plot_path = "../document/master-thesis/plots/cgne_alpha.pdf"
nrows = 1
ncols = 4
config = {
    0: {
        'name': r'naive, $4^4$',
        'xmax': 270,
        'res': 1e-6,
        'tol': 7.838367176906168e-05,
        'files': [
            'pre_processing/cgne_4x4x4x4_1rank_ctype_f64_res_1e-6_binary64.json',
            'pre_processing/cgne_4x4x4x4_1rank_ctype_f32_res_1e-6_binary32_pl.json',
            'pre_processing/cgne_4x4x4x4_1rank_ctype_tf32_res_1e-6_tfloat32_pl.json',
            'pre_processing/cgne_4x4x4x4_1rank_ctype_bf16_res_1e-6_bfloat16_pl.json',
            'pre_processing/cgne_4x4x4x4_1rank_ctype_f16_res_1e-6_binary16_pl.json',
        ],
        'colors': [
            lt.PYCOLORS['cbrown'],
            lt.PYCOLORS['cred'],
            lt.PYCOLORS['cpurple'],
            lt.PYCOLORS['cturquois'],
            lt.PYCOLORS['cblue'],
        ],
        'labels': [
            r'binary64, s={status} (+{reset})',
            r'binary32, s={status} (+{reset})',
            r'tensorfloat32, s={status} (+{reset})',
            r'bfloat16, s={status} (+{reset})',
            r'binary16, s={status} (+{reset})',
        ],
    },
    1: {
        'name': r'naive+quire, $4^4$',
        'xmax': 270,
        'res': 1e-6,
        'tol': 7.838367176906168e-05,
        'files': [
            'pre_processing/cgne_4x4x4x4_1rank_ctype_f64_res_1e-6_binary64.json',
            'pre_processing/cgne_4x4x4x4_1rank_ctype_f32_res_1e-6_binary32_pl.json',
            'pre_processing/cgne_4x4x4x4_1rank_ctype_tf32_res_1e-6_tfloat32_pl.json',
            'pre_processing/cgne_4x4x4x4_1rank_ctype_bf16_res_1e-6_bfloat16_pl.json',
            'pre_processing/cgne_4x4x4x4_1rank_ctype_f16_res_1e-6_binary16_pl.json',
        ],
        'colors': [
            lt.PYCOLORS['cbrown'],
            lt.PYCOLORS['cred'],
            lt.PYCOLORS['cpurple'],
            lt.PYCOLORS['cturquois'],
            lt.PYCOLORS['cblue'],
        ],
        'labels': [
            r'binary64, s={status} (+{reset})',
            r'binary32, s={status} (+{reset})',
            r'tensorfloat32, s={status} (+{reset})',
            r'bfloat16, s={status} (+{reset})',
            r'binary16, s={status} (+{reset})',
        ],
    },
    2: {
        'name': r'col=binary64, $4^4$',
        'xmax': 270,
        'res': 1e-6,
        'tol': 7.838367176906168e-05,
        'files': [
            'pre_processing/cgne_4x4x4x4_1rank_ctype_f64_res_1e-6_binary64_pl.json',
            'pre_processing/cgne_4x4x4x4_1rank_ctype_f64_res_1e-6_binary32_pl.json',
            'pre_processing/cgne_4x4x4x4_1rank_ctype_f64_res_1e-6_tfloat32_pl.json',
            'pre_processing/cgne_4x4x4x4_1rank_ctype_f64_res_1e-6_bfloat16_pl.json',
            'pre_processing/cgne_4x4x4x4_1rank_ctype_f64_res_1e-6_binary16_pl.json',
        ],
        'colors': [
            lt.PYCOLORS['cbrown'],
            lt.PYCOLORS['cred'],
            lt.PYCOLORS['cpurple'],
            lt.PYCOLORS['cturquois'],
            lt.PYCOLORS['cblue'],
        ],
        'labels': [
            r'binary64, s={status} (+{reset})',
            r'binary32, s={status} (+{reset})',
            r'tensorfloat32, s={status} (+{reset})',
            r'bfloat16, s={status} (+{reset})',
            r'binary16, s={status} (+{reset})',
        ],
    },
    3: {
        'name': r'col=binary64, $4^4$',
        'xmax': 270,
        'res': 1e-12,
        'tol': 7.838367176906168e-11,
        'files': [
            'pre_processing/cgne_4x4x4x4_1rank_ctype_f64_res_1e-12_binary64.json',
            'pre_processing/cgne_4x4x4x4_1rank_ctype_f64_res_1e-12_binary32_pl.json',
            'pre_processing/cgne_4x4x4x4_1rank_ctype_f64_res_1e-12_tfloat32_pl.json',
            'pre_processing/cgne_4x4x4x4_1rank_ctype_f64_res_1e-12_bfloat16_pl.json',
            'pre_processing/cgne_4x4x4x4_1rank_ctype_f64_res_1e-12_binary16_pl.json',
        ],
        'colors': [
            lt.PYCOLORS['cbrown'],
            lt.PYCOLORS['cred'],
            lt.PYCOLORS['cpurple'],
            lt.PYCOLORS['cturquois'],
            lt.PYCOLORS['cblue'],
        ],
        'labels': [
            r'binary64, s={status} (+{reset})',
            r'binary32, s={status} (+{reset})',
            r'tensorfloat32, s={status} (+{reset})',
            r'bfloat16, s={status} (+{reset})',
            r'binary16, s={status} (+{reset})',
        ],
    },
}

"""
# 8x8x8x8 lattice and sf bc'c
plot_path = "../document/master-thesis/plots/cgne_alpha_8x8x8x8.pdf"
nrows = 4 #len(config)
ncols = 3 #4
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
            #'pre_processing/cgne_sf_8x8x8x8_1rank_ctype_f64_res_1e-6_binary64.json',
            #'pre_processing/cgne_sf_8x8x8x8_1rank_ctype_f64_res_1e-6_binary32_pl.json',
            #'pre_processing/cgne_sf_8x8x8x8_1rank_ctype_f64_res_1e-6_binary16_pl.json',
            'pre_processing/cgne_10_sf_8x8x8x8_1rank_ctype_f64_res_1e-6_binary64_pl.json',
            'pre_processing/cgne_10_sf_8x8x8x8_1rank_ctype_f64_res_1e-6_binary32_pl.json',
            'pre_processing/cgne_10_sf_8x8x8x8_1rank_ctype_f64_res_1e-6_binary16_pl.json',
        ],
        'colors': [
            lt.PYCOLORS['cbrown'],
            lt.PYCOLORS['cred'],
            lt.PYCOLORS['cblue'],
            #lt.PYCOLORS['cgreen'],
            #lt.PYCOLORS['cpink'],
            #lt.PYCOLORS['cyellow'],
        ],
        'labels': [
            r'binary64, s={status} (+{reset})',
            r'binary32, s={status} (+{reset})',
            r'binary16, s={status} (+{reset})',
            #r'binary64, s={status} (+{reset})',
            #r'binary32, s={status} (+{reset})',
            #r'binary16, s={status} (+{reset})',
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
            #'pre_processing/cgne_sf_8x8x8x8_1rank_ctype_f64_res_1e-12_binary64.json',
            #'pre_processing/cgne_sf_8x8x8x8_1rank_ctype_f64_res_1e-12_binary32_pl.json',
            #'pre_processing/cgne_sf_8x8x8x8_1rank_ctype_f64_res_1e-12_binary16_pl.json',
            'pre_processing/cgne_10_sf_8x8x8x8_1rank_ctype_f64_res_1e-12_binary64_pl.json',
            'pre_processing/cgne_10_sf_8x8x8x8_1rank_ctype_f64_res_1e-12_binary32_pl.json',
            'pre_processing/cgne_10_sf_8x8x8x8_1rank_ctype_f64_res_1e-12_binary16_pl.json',
        ],
        'colors': [
            lt.PYCOLORS['cbrown'],
            lt.PYCOLORS['cred'],
            lt.PYCOLORS['cblue'],
            #lt.PYCOLORS['cgreen'],
            #lt.PYCOLORS['cpink'],
            #lt.PYCOLORS['cyellow'],
        ],
        'labels': [
            r'binary64, s={status} (+{reset})',
            r'binary32, s={status} (+{reset})',
            r'binary16, s={status} (+{reset})',
            #r'binary64, s={status} (+{reset})',
            #r'binary32, s={status} (+{reset})',
            #r'binary16, s={status} (+{reset})',
        ],
        'inline_legend': [],
    },
}
"""

nmx = 256
fig = plt.figure()
plt_nr = 1
file_nr = 0
bins = np.linspace(0.045, 0.06, 25)

for series in range(0,len(config)):
    ax = plt.subplot(nrows, ncols, plt_nr)
    plt_nr = plt_nr+1
    #ax.set_yscale('log')
    ax.set_title('plot {0}{1}, {2}, nmx={3}, res={4}'.format(series+1, 'a', config[series]['name'], nmx, config[series]['res']))

    i=0
    for path in config[series]['files']:
        print(path)
        with open(path) as fh:
            j = json.load(fh)
            y = j["ai"]
            for x in y:
                if np.isnan(x) or np.isinf(x):
                    y.remove(x) 
            status = j["status"]
            reset = j["reset"]

            label = config[series]['labels'][i].format(status = status, reset = len(j["reset"]))
            if status > 0:
                plt.hist(y, bins, histtype='bar', stacked=True, facecolor=config[series]['colors'][i], alpha=0.5, label=label)
            i = i+1

    plt.xlabel(r'value of amounts $\alpha$')
    plt.ylabel(r'occurence')
    plt.legend()


fig.set_size_inches(20, nrows*4.5, forward=True)
fig.tight_layout()
plt.savefig(plot_path.format(file_nr), dpi=fig.dpi)
