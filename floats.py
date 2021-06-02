#!/usr/bin/env python3
#
# TODO

import bitstring
import numpy as np

class bfloat16:
    def __init__(self, f):
        self.f32 = np.float32(f)
        self.ebits = 8 # only 8 bit exponents can be simulated with this datatype!!!
        self.mbits = 7
        self.len = 16 # ebits + mbits + 1
        self.v = self.roundTiesToEven() # the actual simulated value

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
    normal numbers (mantissa):
                        := rem=1010000010110 --> cut away
                    ┌──────┴────┐ 
    old = 00111010001010000010110
          ││││││││││└─────────┐
          │││││││││└──┐     carry
    new = 0011101001  lsb  
          └───┬───┘
            copy
        if rem == 10...0:
            if lsb == 1:
                new++
            else:
                cut away rem
        elif carry == 0:
            cut aray rem
        elif carry == 1:
            new++

    subnormal numbers (mantissa):
                              := rem=1001....1101 --> cut away
                    ┌──────────┴──────────┐
    old =  01101110010010100110011010000001
           ││││││││ └─────┐
    new = 1011011101      │
     ┌───┘│└──┬───┘│      │
     │    │ copy  lsb   carry
     │ fixed one
     └── no leading zeros, special case when both exponents are equal

    if carry == 1:
        new++
    elif carry == 0:
        cut away rem

    """
    def roundTiesToEven(self):
        if np.isnan(self.f32):
            return float("nan")
        ba = bitstring.BitArray(float=self.f32, length=32)
        exp = ba.bin[1:self.ebits+1] # excluding end, 8 bits

        if exp == '0'*self.ebits: # number is subnormal
            man = '1' + ba.bin[self.ebits+1:self.len-1] # insert the fixed one
            carry = ba.bin[self.len-1]
            ba.overwrite('0b'+man, self.ebits+1)
            ba.overwrite('0b'+'0'*(32-self.len), self.len) # truncate
            if carry == '1':
                return self.increment(man, exp, ba)
            else:
                return ba.float
        else: # number is normal
            rem = ba.bin[self.len:]
            lsb = ba.bin[self.len-1]
            man = ba.bin[self.ebits+1:self.len] # excluding end, end-start bits
            carry = ba.bin[self.len]
            # truncate, overwrites the leftmost len (16) bits with 0s
            ba.overwrite('0b'+'0'*(32-self.len), self.len)
            if rem == '1'+'0'*(32-self.len-1): # =1/2
                if lsb == '1':
                    return self.increment(man, exp, ba)
                else:
                    return ba.float
            elif carry == '1': # >1/2
                return self.increment(man, exp, ba)
            else: # <1/2
                return ba.float

    def increment(self, man, exp, ba):
        man, carry = self.incr_bits(man)
        if carry:
            exp, carry = self.incr_bits(exp)
            if carry:
                return np.float32('inf')
            else:
                ba.overwrite('0b'+exp, 1)
                ba.overwrite('0b'+man, self.ebits+1)
                return ba.float
        else:
            ba.overwrite('0b'+man, self.ebits+1)
            return ba.float

    def incr_bits(self, b):
        carry = False
        bits = list(b[::-1])
        for i, bit in enumerate(bits):
            carry = bit != "0"
            if bit == "0":
                bits[i] = "1"
                break
            else:
                bits[i] = "0"
        return "".join(bits[::-1]), carry

    def __str__(self):
        return str(self.v)

    # if casted via float(obj)
    def __float__(self):
        return self.v

    # +=, should return self
    """
    def __iadd__(self, other):
        rtype, oval = self.rtype(other)
        self.f32 = self.v + oval
        self.v = self.roundTiesToEven()
        return self
    """

    def __add__(self, other):
        rtype, oval = self.rtype(other)
        return rtype(self.v + oval)

    def __radd__(self, other):
        rtype, oval = self.rtype(other)
        return rtype(oval + self.v)

    def __sub__(self, other):
        rtype, oval = self.rtype(other)
        return rtype(self.v - oval)

    def __rsub__(self, other):
        rtype, oval = self.rtype(other)
        return rtype(oval - self.v)

    def __mul__(self, other):
        rtype, oval = self.rtype(other)
        return rtype(self.v * oval)

    def __rmul__(self, other):
        rtype, oval = self.rtype(other)
        return rtype(oval * self.v)

    def __truediv__(self, other):
        rtype, oval = self.rtype(other)
        return rtype(self.v / oval)

    def __rtruediv__(self, other):
        rtype, oval = self.rtype(other)
        return rtype(oval / self.v)

    def __neg__(self):
        return type(self)(-self.f32)

    def __lt__(self, other):
        rtype, oval = self.rtype(other)
        return self.v < oval

    def __le__(self, other):
        rtype, oval = self.rtype(other)
        return self.v <= oval

    def __eq__(self, other):
        rtype, oval = self.rtype(other)
        return self.v == oval

    def __ne__(self, other):
        rtype, oval = self.rtype(other)
        return self.v != oval

    def __ge__(self, other):
        rtype, oval = self.rtype(other)
        return self.v >= oval

    def __gt__(self, other):
        rtype, oval = self.rtype(other)
        return self.v > oval

    def sqrt(self):
        return type(self)(np.sqrt(self.v))

    def rtype(self, other):
        metype = type(self)
        if isinstance(other, metype):
            return metype, other.v
        elif isinstance(other, np.float64) or isinstance(other, np.float32):
            return type(other), other
        elif isinstance(other, np.float16):
            return type(self), other
        elif isinstance(other, type(1.0)):
            return np.float64, other
        else:
            return None, other

class tfloat32(bfloat16):
    def __init__(self, f):
        self.f32 = np.float32(f)
        self.ebits = 8 # only 8 bit exponents can be simulated with this datatype!!!
        self.mbits = 10
        self.len = 19 # ebits + mbits + 1
        self.v = self.roundTiesToEven() # the actual simulated value

"""
        ba = bitstring.BitArray(float=self.f32, length=32)
        exp = ba.bin[1:self.ebits+1] # excluding end, 8 bits
        old = bitstring.BitArray(float=self.f32, length=32)
        new = bitstring.BitArray(float=self.v, length=32)
        if exp == '0'*self.ebits: # number is subnormal
            print("fl32", old.bin[:1], old.bin[1:self.ebits+1], " "+old.bin[self.ebits+1:self.len-1], old.bin[self.len-1:], self.f32)
            print("bf16", new.bin[:1], new.bin[1:self.ebits+1], new.bin[self.ebits+1:self.len], new.bin[self.len:], self.v)
        else:
            print("fl32", old.bin[:1], old.bin[1:self.ebits+1], old.bin[self.ebits+1:self.len], old.bin[self.len:], self.f32)
            print("bf16", new.bin[:1], new.bin[1:self.ebits+1], new.bin[self.ebits+1:self.len], new.bin[self.len:], self.v)
        print("------------------------------")
"""

"""
        self.f = np.float32(f)
        self.ebits = 8
        self.mbits = 10
        self.len = self.ebits + self.mbits + 1 # 19
        ba = bitstring.BitArray(float=self.f, length=32)
        ba.overwrite('0b0000000000000', 19) # truncate, overwrites the leftmost 13 bits with 0s (stating from bit 19, counted from the left)
        self.v = ba.float
        self.metype = tfloat32
"""

"""
print("normals")
flt = bfloat16(np.float32('inf')) # inf
flt = bfloat16(3.4028235E38) # inf
flt = bfloat16(1.7014117E38) # biggest nr
flt = bfloat16(1.7014086E38) # same
flt = bfloat16(8.5070587E37) # same
flt = bfloat16(1.6582119E38) # =1/2, roundup
flt = bfloat16(1.6515658E38) # =1/2, cutoff
flt = bfloat16(1.6515788E38) # >1/2, roundup
flt = bfloat16(1.6482557E38) # <1/2, cutoff
"""

"""
print("normals")
flt = tfloat32(np.float32('inf')) # inf
flt = tfloat32(np.float32('nan')) # nan
flt = tfloat32(bitstring.BitArray('0b11111111111111111111000100000111', length=32).float) # nan
flt = tfloat32(3.4028235E38) # inf
flt = tfloat32(1.7014117E38) # biggest nr
flt = tfloat32(1.7014086E38) # same
flt = tfloat32(8.5070587E37) # same
flt = tfloat32(1.6893657E38) # =1/2, roundup
flt = tfloat32(1.6885349E38) # =1/2, cutoff
flt = tfloat32(1.6893795E38) # >1/2, roundup
flt = tfloat32(1.6889641E38) # <1/2, cutoff
print("subnormals")
flt = tfloat32(5.957828E-39) # subnormal, roundup
flt = tfloat32(5.957873E-39) # subnormal, roundup
flt = tfloat32(1.1743464E-38) # subnormal, roundup
flt = tfloat32(1.1743554E-38) # subnormal, roundup
flt = tfloat32(5.946348E-39) # subnormal, cutoff
flt = tfloat32(5.946393E-39) # subnormal, cutoff
flt = tfloat32(1.173203E-38) # subnormal, cutoff
flt = tfloat32(1.1731985E-38) # subnormal, cutoff

b16 = np.float16(1)
b32 = np.float32(2)
b64 = np.float64(3)
tf32 = bfloat16(4)
tf32_2 = bfloat16(5)
pi = 3.1415

x = np.sqrt(tf32)
y = -tf32
print("sqrt", type(x), x)
print("neg", type(y), y)

print("tf32")
x = tf32_2+tf32
y = tf32+tf32_2
print("add", type(y), y)
print("add", type(x), x)
print("done")

print("float")
x = pi+tf32
y = tf32+pi
print("add", type(y), y)
print("add", type(x), x)
print("done")

print("b64")
x = b64+tf32
y = tf32+b64
print("add", type(y), y)
print("add", type(x), x)
print("done")

print("b32")
x = b32+tf32
y = tf32+b32
print("add", type(y), y)
print("add", type(x), x)
print("done")

print("b16")
x = b16+tf32
y = tf32+b16
print("add", type(y), y)
print("add", type(x), x)
print("done")
"""
