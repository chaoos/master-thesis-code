import sys # sys.stdin
import argparse # argparse.ArgumentParser()
import bitstring # BitArray()
import math # floor()
import numpy as np # np.float16
import random

"""
The exponent is converted by 2 components:
 * the highest bit is taken as it is
 * the remaining bits are taken from the end
Example:
    10010111    8-bit exponent of the IEEE754 float single precision datatype
    │┌──┘│││
    ││┌──┘││
    │││┌──┘│
    ││││┌──┘
    10111       5-bit exponent of the IEEE754 half precision datatype

    * the other 3 bits ("001" in the example) are lost due to lower precision
"""
def convert_exponent(exp, exp_bits, man = "0"):
    if (exp == "1"*(len(exp)-1) + "0") and (man == "1"*(len(man))) :
        # infinity (exponent is 1...10 and mantissa is 1...1)
        return infinity_exp(exp_bits)
    if exp_bits > len(exp):
        # if the desired exponent is bigger than the given one, fill it with zeros
        return exp[0] + "0"*(exp_bits - len(exp)) + exp[1:]
    elif exp_bits == len(exp):
        return exp
    else:
        # down conversion
        return exp[0] + exp[-exp_bits+1:]

def convert_exponent_subnormal(exp_bits):
    return "0"*exp_bits

"""
Implements roundTiesToEven from IEEE754_2008
The mantissa is converted as follows (roundTiesToEven)
 if what is left is <1/2:
    round down by truncating
 if what is left is >1/2:
    round up by adding 1 to the lsb of the mantissa
 if what is left is =1/2:
    if lsb==1:
        round up by adding 1 to the lsb of the mantissa
    if lsb==0:
        round down by truncating
Example:
                    := rem=1010000010110 --> cut away
                ┌──────┴────┐ 
old = 00111010001010000010110
      ││││││││││└─────────┐
      │││││││││└──┐     carry
new = 0011101001  lsb  
      └───┬───┘
        copy
    if rem == 0*...*:
        cut aray rem
    elif rem == 1*...*1*...*:
        new++
    elif rem == 10...0:
        if lsb == 1:
            new++
        else:
            cut away rem

"""
def convert_mantissa(man, man_bits, exp = "10"):
    if (exp == "1"*(len(exp)-1) + "0") and (man == "1"*(len(man))):
        # infinity
        return infinity_man(man_bits)
    if man_bits > len(man):
        # if the desired mantissa is bigger than the given one, fill it with zeros
        return man + "0"*(man_bits-len(man))
    elif man_bits == len(man):
        return man
    else:
        # down conversion
        carry = man[man_bits]
        lsb = man[man_bits-1]
        truncated_mantissa = man[0:man_bits]

        if carry == "0":
            return truncated_mantissa
        elif carry == "1":
            ones = 0
            for i in range(man_bits+1,len(man)):
                if (man[i] == "1"):
                    ones += 1

            if (ones > 0):
                return increment_binary(truncated_mantissa)
            else:
                if lsb == "1":
                    return increment_binary(truncated_mantissa)
                else:
                    return truncated_mantissa


"""
                                               := rem=1001010011001101000000011011110100111101 --> cut away
                             ┌──────────────────┴───────────────────┐
old =            0111100111001001010011001101000000011011110100111101
                 │││││││││││ └─────┐
new = 00000000001011110011101      │
      └───┬────┘│└────┬────┘│      │
       z zeros  │    copy  lsb   carry
            fixed one

z = -true exponent -bias_of_new_exponent

if carry == 1
    if rem == 1*...*:
        new++
    elif rem == 0*...*:
        cut away rem

0 01101110110            1011111000001011110100001011111100100010010111000111
0    00000000 00000000001101111100001


"""
def convert_mantissa_subnormal(man, man_bits, exp, exp_bits):
    zeros = int(-binary2exponent(exp)-exponent_bias(exp_bits))
    from_mantissa = man_bits -zeros -1
    tail = man[0:from_mantissa]
    carry_bit = man[from_mantissa]
    mantissa = "0"*zeros + "1" + tail
    remainder = man[from_mantissa+1:]
    if carry_bit == "1":
        ones = 0
        for i in range(0,len(remainder)):
            if (remainder[i] == "1"):
                ones += 1
        if (ones > 0):
            return increment_binary(mantissa)
    
    return mantissa

def increment_binary(b):
    has_zeros = False
    for i in range(0,len(b)):
        if b[i] == "0":
            has_zeros = True

    if has_zeros:
        if (b[-1] == "0"):
            l = list(b)
            l[-1] = "1"
            b = ''.join(l)
        elif (b[-1] == "1"):
            return increment_binary(b[0:-1]) + "0"
    return b

def cast_binary(input, exp_bits, man_bits):
    sign, exp, man = input.split(" ", 2)
    a = abs(binary2double(input))
    if a >= lowest_inf(exp_bits, man_bits):
        # out off range -> ±inf
        return signed_binfinity(sign, exp_bits, man_bits)
    elif a < lowest_inf(exp_bits, man_bits) and a >= highest_number(exp_bits, man_bits):
        # round to -> ±highest number
        return sign + " " + "1"*(exp_bits-1) + "0 " + "1"*man_bits
    elif a <= highest_zero(exp_bits, man_bits):
        # out of range -> ±zero
        return signed_bzero(sign, exp_bits, man_bits)
    elif a > highest_zero(exp_bits, man_bits) and a <= lowest_subnormal_number(exp_bits, man_bits):
        # round to -> ±lowest subnormal number
        return sign + " " + "1"*exp_bits + "0"*(man_bits-1) + "1"
    elif a >= lowest_subnormal_number(exp_bits, man_bits) and a < lowest_regular_number(exp_bits, man_bits):
        # in the subnormal numbers regime
        return sign + " " + convert_exponent_subnormal(exp_bits) + " " + convert_mantissa_subnormal(man, man_bits, exp, exp_bits)
    else:
        # within bounds -> regular down conversion
        return sign + " " + convert_exponent(exp, exp_bits, man) + " " + convert_mantissa(man, man_bits, exp)

def infinity_exp(exp_bits):
    return "1"*exp_bits

def infinity_man(man_bits):
    return "0"*man_bits

def signed_binfinity(sign, exp_bits, man_bits):
    return sign + " " + infinity_exp(exp_bits) + " " + infinity_man(man_bits)

def signed_bzero(sign, exp_bits, man_bits):
    return sign + " " + "0"*exp_bits + " " + "0"*man_bits

def binary2exponent(exp):
    exp_bits = len(exp)
    exponent = 0.0

    for i in range(0, exp_bits):
        if (exp[exp_bits-1 -i] == "1"):
            exponent += 2**i
    exponent -= exponent_bias(exp_bits)
    return int(exponent)

def binary2mantissa(man, exp):
    exp_bits, man_bits = len(exp), len(man)

    if (exp == "0"*exp_bits):
        mantissa = 0.0 # IEEE754_2008 3.4 d)
    else:
        mantissa = 1.0 # IEEE754_2008 3.4 d)

    for i in range(0, man_bits):
        if (man[i] == "1"):
            mantissa += 2**(-i-1)
    return mantissa

def binary2signum(sign):
    if sign == "0":
        return 1.0
    else:
        return -1.0

def binary2double(input):
    sign, exp, man = input.split(" ", 2)
    exp_bits, man_bits = len(exp), len(man)

    if (exp == "0"*exp_bits) and (man == "0"*man_bits):
        # IEEE754_2008 3.4 e)
        if sign == "0":
            return 0.0
        else:
            return -0.0
    elif (exp == "1"*exp_bits):
        if (man == "0"*man_bits):
            # IEEE754_2008 3.4 b)
            if sign == "0":
                return float("inf")
            else:
                return float("-inf")
        else:
            # IEEE754_2008 3.4 a)
            return float("NaN")

    exponent = binary2exponent(exp)
    mantissa = binary2mantissa(man, exp)
    signum   = binary2signum(sign)

    return signum*mantissa*(2**exponent)

def double2binary(input):
    f = bitstring.BitArray(float=input, length=64)
    return str(f.bin[0]) +" "+ str(f.bin[1:12]) +" "+ str(f.bin[12:])

def cast(input, exp_bits, man_bits):
    return binary2double(cast_binary(double2binary(input), exp_bits, man_bits))

def exponent_bias(exp_bits):
    return 2**(exp_bits -1) -1

"""
Returns the highest number representable by the float type
@param int exp_bits number of exponent bits
@param int man_bits number of mantissa bits
@return float number
"""
def highest_number(exp_bits, man_bits):
    return (+1)*(2 - 2**(-man_bits))*2**(2**exp_bits -2**(exp_bits -1) -1)

"""
Returns the lowest (subnormal) number above zero representable by the float type.
This is the lowest number that is strictly positive.
@param int exp_bits number of exponent bits
@param int man_bits number of mantissa bits
@return float number
"""
def lowest_subnormal_number(exp_bits, man_bits):
    return (+1)*(0.0 + 2**(-man_bits))*2**(-exponent_bias(exp_bits) +1)

"""
Returns the lowest (non subnormal) number above zero representable by the float type
@param int exp_bits number of exponent bits
@param int man_bits number of mantissa bits
@return float number
"""
def lowest_regular_number(exp_bits, man_bits):
    return (+1)*(1.0 + 2**(-man_bits))*2**(-exponent_bias(exp_bits) +1)

"""
Returns the lowest number that is rounded to infinity
@param int exp_bits number of exponent bits
@param int man_bits number of mantissa bits
@return float number
"""
def lowest_inf(exp_bits, man_bits):
    return highest_number(exp_bits, man_bits) + 2**(exponent_bias(exp_bits) - man_bits -1)

"""
TODO
Returns the highest positive number that is rounded to zero
@param int exp_bits number of exponent bits
@param int man_bits number of mantissa bits
@return float number
"""
def highest_zero(exp_bits, man_bits):
    return lowest_subnormal_number(exp_bits, man_bits)


def random_subnormal(exp_bits, man_bits):
    lower = lowest_subnormal_number(exp_bits, man_bits)
    upper = lowest_regular_number(exp_bits, man_bits)
    return random.uniform(lower, upper)

"""
Return the difference between 1.0 and the least value greater than 1.0 that is representable as a float
Also known as the machine epsilon
@param int man_bits number of mantissa bits
@return float number
"""
def machine_epsilon(man_bits):
    return 2**(-man_bits-1)
