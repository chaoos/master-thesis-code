#!/usr/bin/env python3
#
# pre processing script analizing the raw data
# produces a json output file
# call by:
# ./analyze_floats_pp.py <debug.log >output.json

from collections import Counter
import float_utils as fl
import sys # sys.stdin, sys.stdout
#import pickle # pickle.dump(), pickle.load()
import json # json.dump(), json.load()

# rows are (last row is the float in decimal representation)
# sign binary_exp binary_mantissa float_in_%.10e
# examples:
# 1 10000000 01100110000110001101110 -2.7976336479e+00
# 0 01111100 01010001000001010100100 1.6456085443e-01

exponents = []
#mantissas = []
zeros, n = 0, 0

for line in sys.stdin:
    row = line.split(" ", 5)
    if len(row) > 1:
        n = n + 1
        if float(row[3]) != 0.0:
            exponents.append(fl.binary2exponent(row[1]))
            #mantissas.append(fl.binary2mantissa(row[2], row[1]))
        else:
            zeros = zeros +1

# https://stackoverflow.com/a/12288109/2768341
exp = list(Counter(exponents).keys())
num = list(Counter(exponents).values())

#with open("pre_processing/analyze_floats_pp.json", "w") as f:
    #pickle.dump(exp, f)             # list of exponents
    #pickle.dump(num, f)             # number of occurences of the exponents
    #pickle.dump(zeros, f)           # number of zeros 0.0
    #pickle.dump(n, f)               # total number of valid entries in raw data file 

json_data = {
    'exp': exp,     # list of exponents
    'num': num,     # number of occurences of the exponents
    'zeros': zeros, # number of zeros 0.0
    'n': n          # total number of valid entries in raw data file 
}
json.dump(json_data, sys.stdout)
