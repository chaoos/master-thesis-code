#!/usr/bin/env python3
#
# pre processing script analizing the raw data
# produces a json output file
# call by:
# ./exponents_pp.py cgne.dat [-d] >out.json

#import float_utils as fl
import sys # sys.stdin, sys.stdout
import argparse
import json # json.dump(), json.load()
import struct # struct.unpack()
import numpy as np # np.asarray, np.zeros(), np.float32
import math # math.frexp()
from collections import Counter # Counter().keys(), Counter().values()

# The binary file contains all needed numbers of cgne()
# vol nmx res
# eta[0] ... eta[vol*24]
# psi[0] ... psi[vol*24]
# D[0,0]        D[0,1]      ... D[0,vol*24]
# D[1,0]        D[1,1]      ... D[1,vol*24]
# ...
# D[vol*24,0]   D[24*vol,1] ... D[vol*24,vol*24]

parser = argparse.ArgumentParser(description='Analize cgne() .dat.')
parser.add_argument('file', help='the input file')
parser.add_argument('-d', '--double', help='the input file contains Dop_dble() instead of Dop().', default=False, action="store_true")
args = parser.parse_args()

int_size, float_size, double_size = 4, 4, 8
byteorder='little'

size = float_size
dtype = np.float32
fmt = '<{0}f'
if args.double:
    size = double_size
    dtype = np.float64
    fmt = '<{0}d'

exponents = []
zeros, n = 0, 0
with open(args.file, "rb") as f:
    vol = struct.unpack('<1I', f.read(int_size))[0] # 1 unsigned int I, < little endian
    nmx = struct.unpack('<1I', f.read(int_size))[0]
    res = struct.unpack('<1d', f.read(double_size))[0] # 1 double
    eta = struct.unpack('<{0}d'.format(24*vol), f.read(double_size*24*vol)) # 24*vol doubles
    psi = struct.unpack('<{0}d'.format(24*vol), f.read(double_size*24*vol)) # 24*vol doubles

    # initialize the Dirac matrix with all zeros and datatype binary32
    # F means column major instead of row major, because the matrix is already
    # saved that way (this is cheaper)
    Dop = np.zeros((24*vol,24*vol), dtype=dtype, order='F')
    for j in range(0,24*vol):
        Dop[j] = np.asarray(struct.unpack(fmt.format(24*vol), f.read(size*24*vol)))
        
    #print(vol, nmx, re
    for flt in np.nditer(Dop):
        n = n + 1
        if flt != 0.0:
            exponents.append(math.frexp(flt)[1] -1)
        else:
            zeros = zeros + 1

# https://stackoverflow.com/a/12288109/2768341
exp = list(Counter(exponents).keys())
num = list(Counter(exponents).values())
norm = np.linalg.norm(Dop)

json_data = {
    'vol': vol,
    'norm': norm,
    'exp': exp,     # list of exponents
    'num': num,     # number of occurences of the exponents
    'zeros': zeros, # number of zeros 0.0
    'n': n          # total number of valid entries in raw data file 
}
json.dump(json_data, sys.stdout)

