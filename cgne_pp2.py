#!/usr/bin/env python3
#
# pre processing script analizing the raw data
# calculates the cgne() using the dirac matrix ||D|| and eta

import argparse
import struct # struct.unpack(), struct.pack()
import numpy as np # np.asarray, np.zeros(), np.float16, np.float32, np.float64, np.linalg.norm()
import softposit as sp # sp.posit8, sp.posit16, sp.posit32
import os # os.remove(), os.system()
import sys
import time
from random import randrange
from scipy import sparse
import json # json.dump(), json.load()
import floats as fl

# The binary file contains all needed numbers of cgne()
# vol nmx res
# eta[0] ... eta[vol*24]
# psi[0] ... psi[vol*24]
# D[0,0]        D[0,1]      ... D[0,vol*24]
# D[1,0]        D[1,1]      ... D[1,vol*24]
# ...
# D[vol*24,0]   D[24*vol,1] ... D[vol*24,vol*24]

# for IEEE floats: 2^(-mantissa_bits)
# for posits: determined empirical see https://en.wikipedia.org/wiki/Machine_epsilon#Approximation
def get_epsilon(dtype):
    if dtype == np.float64:
        return np.float64(2.2204460492503131e-16) # 2^-52
    elif dtype == np.float32:
        return np.float64(1.1920928955078125e-07) # 2^-23
    elif dtype == np.float16:
        return np.float64(0.0009765625) # 2^-10
    elif dtype == sp.posit32:
        return np.float64(7.450580596923828e-09)
    elif dtype == sp.posit16:
        return np.float64(0.000244140625)
    elif dtype == sp.posit8:
        return np.float64(0.03125)
    elif dtype == fl.tfloat32:
        return np.float64(0.0009765625) # 2^-10
    elif dtype == fl.bfloat16:
        return np.float64(0.0078125) # 2^-7

# number of significant digits in decimal
# for binary64 -> 16
# for binary32 -> 7
# for binary16 -> 3-4
# *100 because he wants to abort 2 orders of magitude before the real precision limit (epsilon) is reached
#
def get_precision_limit(dtype):
    return np.float64(100.0*get_epsilon(dtype))

FLT_EPSILON = np.float64(1.1920928955078125e-07)
DBL_EPSILON = np.float64(2.2204460492503131e-16)
PRECISION_LIMIT = np.float64(100.0*FLT_EPSILON)

def convert(x, dtype):
    if type(x) == dtype:
        return x
    if dtype == sp.posit32 or dtype == sp.posit16 or dtype == sp.posit8:
        return dtype(np.float64(x))
    else:
        return dtype(x)

def cgne(vol, nmx, res, Dop_Re, Dop_Im, Dop_dble_Re, Dop_dble_Im, eta_Re, eta_Im, psi_Re, psi_Im, sdtype, ddtype, cdtype, extension = 0):

    N = eta_Re.size
    # BEGIN cg_init()
    print('cg_init()'.format())
    sdtype_zero = sdtype(0.0)
    updb_Re, updb_Im = eta_Re, eta_Im # unscaled eta
    eta_Re, eta_Im = np.array([x for x in eta_Re]), np.array([x for x in eta_Im])
    pdb_Re, pdb_Im = eta_Re, eta_Im
    pdx_Re, pdx_Im = psi_Re, psi_Im
    psb_Re, psb_Im = np.array([sdtype(x) for x in pdb_Re]), np.array([sdtype(y) for y in pdb_Im]) # ext=1
    #psx_Re, psx_Im = np.zeros(N, dtype=sdtype), np.zeros(N, dtype=sdtype)
    psx_Re, psx_Im = np.array([sdtype_zero for x in pdb_Re]), np.array([sdtype_zero for y in pdb_Im])
    psr_Re, psr_Im = np.array([sdtype(x) for x in pdb_Re]), np.array([sdtype(y) for y in pdb_Im])
    psp_Re, psp_Im = psr_Re, psr_Im
    pdx_Re, pdx_Im = np.zeros(N, dtype=ddtype), np.zeros(N, dtype=ddtype)
    pdr_Re, pdr_Im = pdb_Re, pdb_Im # for extension=1
    rsq = complex_norm_sq(psr_Re, psr_Im, cdtype) # cdtype
    # END cg_init()

    rn = np.sqrt(convert(rsq, ddtype)) # ddtype
    adaptive = 1
    ttol = res*rn
    tol = res*rn*adaptive**4
    status = 0
    i = 0

    json_data = {}
    json_data['res'] = res
    json_data['tol'] = tol
    json_data['vol'] = vol
    json_data['nmx'] = nmx
    json_data['extension'] = extension
    json_data['sdtype'] = type(sdtype(0.0)).__name__
    json_data['ddtype'] = type(ddtype(0.0)).__name__
    json_data['cdtype'] = type(cdtype(0.0)).__name__
    json_data['i'] = []
    json_data['extension_steps'] = []
    json_data['ai'] = [] # the real calculated value of ai
    json_data['approx_ai'] = [] # the approximated one
    json_data['bi'] = []
    json_data['reset'] = []
    json_data['rn'] = []
    json_data['Axmb'] = []
    json_data['diAdip1'] = []


    xn = np.sqrt(complex_norm_sq(psx_Re, psx_Im, ddtype)) # ddtype

    while rn > tol or status == 0:
        ncg = 0
        while True:
            # BEGIN cg_step()
            print('cg_step() ncg={0}, status={1}, rn({2}), tol({3}), rn={4} > tol={5}'.format(ncg, status, type(rn), type(tol), rn, tol))
            psw_Re, psw_Im = complex_matrix_vector_prod(Dop_Re, Dop_Im, psp_Re, psp_Im, sdtype)
            psap_Re, psap_Im = complex_matrix_dagger_vector_prod(Dop_Re, Dop_Im, psw_Re, psw_Im, sdtype)

            ai = rsq/complex_norm_sq(psw_Re, psw_Im, cdtype) # cdtype
            if extension == 1: # ext=1
                if ncg == 4:
                    # average the last 2 elements + the current one
                    cai = (convert(json_data['ai'][-1], cdtype) + convert(json_data['ai'][-2], cdtype) + ai)/convert(3.0, cdtype)
                elif ncg < 4:
                    cai = ai
            else:
                cai = ai

            # update_g()
            psr_Re, psr_Im = axpy(-cai, psap_Re, psap_Im, psr_Re, psr_Im, sdtype)
            print("norm recu", np.sqrt(complex_norm_sq(psr_Re, psr_Im, cdtype)))

            # update_xp()
            psx_Re, psx_Im = axpy(cai, psp_Re, psp_Im, psx_Re, psx_Im, sdtype)

            if extension == 1 and ncg >= 4: # ext=1
                # b - A x_current = pdr - A xi
                # because x_current = xi + x_d
                # pdr = b - A x_d (in double precision from the reset step)
                re, im = np.array([sdtype(x) for x in pdr_Re]), np.array([sdtype(y) for y in pdr_Im])
                psr_Re, psr_Im = Axmb_vec(Dop_Re, Dop_Im, psx_Re, psx_Im, re, im, sdtype)
                psr_Re, psr_Im = -psr_Re, -psr_Im
                print("norm bmAx", np.sqrt(complex_norm_sq(psr_Re, psr_Im, cdtype)))
                json_data['extension_steps'].append(i)

            rsq_old = rsq # cdtype

            rsq = complex_norm_sq(psr_Re, psr_Im, cdtype) # cdtype
            #print("rsq", type(rsq), rsq)

            bi = rsq/rsq_old # cdtype

            # update_xp()
            psp_Re, psp_Im = axpy(bi, psp_Re, psp_Im, psr_Re, psr_Im, sdtype)
            # END cg_step()

            re, im = complex_scalar_prod(psp_Re, psp_Im, psap_Re, psap_Im, cdtype)
            json_data['diAdip1'].append(convert(complex_abs(re, im, cdtype), ddtype))

            print(type(psx_Re[0]), type(psx_Im[0]))

            ncg += 1
            status += 1
            i += 1

            xn = np.sqrt(convert(complex_norm_sq(psx_Re, psx_Im, cdtype), ddtype))
            rn = np.sqrt(convert(rsq, ddtype))

            json_data['i'].append(i)
            json_data['ai'].append(convert(ai, ddtype))
            json_data['approx_ai'].append(convert(cai, ddtype))
            json_data['bi'].append(convert(bi, ddtype))
            json_data['rn'].append(rn)
            uxd_Re = np.array([(convert(x, ddtype)+y) for x, y in zip(psx_Re, pdx_Re)], dtype=ddtype)
            uxd_Im = np.array([(convert(x, ddtype)+y) for x, y in zip(psx_Im, pdx_Im)], dtype=ddtype)
            json_data['Axmb'].append(Axmb(Dop_dble_Re, Dop_dble_Im, uxd_Re, uxd_Im, updb_Re, updb_Im, ddtype))
            print("i", json_data['i'][-1])
            print("ai", json_data['ai'][-1])
            print("approx_ai", json_data['approx_ai'][-1])
            print("ai_error {0:.2g}%".format(abs(100*(json_data['approx_ai'][-1] - json_data['ai'][-1])/json_data['ai'][-1])))
            print("bi", json_data['bi'][-1])
            print("rn", json_data['rn'][-1])
            print("Axmb", json_data['Axmb'][-1])
            print("diAdip1", json_data['diAdip1'][-1])

            nreset = 100
            if rn <= tol or rn <= (get_precision_limit(sdtype)*xn) or ncg >= nreset or status >= nmx:
                tol = tol/adaptive if ttol < tol else ttol
                if rn <= tol:
                    print('cg_step() rn <= tol rn({0}), tol({1}), xn({2}), rn={3}, tol={4} xn={5}'.format(type(rn), type(tol), type(xn), rn, tol, xn))
                if rn <= (get_precision_limit(sdtype)*xn):
                    print('cg_step() rn <= PRECISION_LIMIT*xn rn({0}), xn({1}), rn={2} <= xn={3}'.format(type(rn), type(xn), rn, xn))
                if ncg >= nreset:
                    print('cg_step() ncg >= nreset rn({0}), xn({1}), rn={2}, xn={3}'.format(type(rn), type(xn), rn, xn))
                if status >= nmx:
                    print('cg_step() status >= nmx rn({0}), xn({1}), rn={2}, xn={3}'.format(type(rn), type(xn), rn, xn))
                break

        # pdx(ddtype) = pdx(ddtype) + (ddtype) psx(single)
        pdx_Re = np.array([convert(x, ddtype)+y for x, y in zip(psx_Re, pdx_Re)], dtype=ddtype)
        pdx_Im = np.array([convert(x, ddtype)+y for x, y in zip(psx_Im, pdx_Im)], dtype=ddtype)

        xn = np.sqrt(complex_norm_sq(pdx_Re, pdx_Im, ddtype)) # ddtype

        # BEGIN cg_reset()
        print('cg_reset()'.format())

        pdw_Re, pdw_Im = complex_matrix_vector_prod(Dop_dble_Re, Dop_dble_Im, pdx_Re, pdx_Im, ddtype)
        pdv_Re, pdv_Im = complex_matrix_dagger_vector_prod(Dop_dble_Re, Dop_dble_Im, pdw_Re, pdw_Im, ddtype)

        # single = double - double
        # (single) psr = b - Ax
        pdr_Re, pdr_Im = axpy(-1.0, pdv_Re, pdv_Im, pdb_Re, pdb_Im, ddtype) # ddtype
        psr_Re = np.array([convert(x, sdtype) for x in pdr_Re], dtype=sdtype)
        psr_Im = np.array([convert(x, sdtype) for x in pdr_Im], dtype=sdtype)

        rsq = complex_norm_sq(psr_Re, psr_Im, cdtype) # cdtype

        psw_Re, psw_Im = psp_Re, psp_Im
        psp_Re, psp_Im = psr_Re, psr_Im

        z_Re, z_Im = complex_scalar_prod(psr_Re, psr_Im, psw_Re, psw_Im, cdtype) # cdtype
        z_Re = -z_Re/rsq # cdtype
        z_Im = -z_Im/rsq # cdtype
        psw_Re, psw_Im = complex_axpy(z_Re, z_Im, psr_Re, psr_Im, psw_Re, psw_Im, sdtype)

        psx_Re, psx_Im = complex_matrix_vector_prod(Dop_Re, Dop_Im, psw_Re, psw_Im, sdtype)
        psap_Re, psap_Im = complex_matrix_dagger_vector_prod(Dop_Re, Dop_Im, psx_Re, psx_Im, sdtype)

        r = complex_norm_sq(psx_Re, psx_Im, cdtype) # cdtype
        z_Re, z_Im = complex_scalar_prod(psap_Re, psap_Im, psr_Re, psr_Im, cdtype) # cdtype

        if (z_Re*z_Re+z_Im*z_Im) < cdtype(2.0)*r*r: # cdtype
            z_Re = -z_Re/r # cdtype
            z_Im = -z_Im/r # cdtype
            psp_Re, psp_Im = complex_axpy(z_Re, z_Im, psw_Re, psw_Im, psp_Re, psp_Im, sdtype)

        # psx = (sdtype)0
        psx_Re, psx_Im = np.array([sdtype_zero for x in psx_Re]), np.array([sdtype_zero for y in psx_Im])
        # END cg_reset()
        i += 1

        rn = np.sqrt(convert(rsq, ddtype)) # ddtype
        json_data['i'].append(i)
        json_data['reset'].append(i)
        json_data['rn'].append(rn)
        updx_Re = np.array([x for x in pdx_Re])
        updx_Im = np.array([x for x in pdx_Im])
        json_data['Axmb'].append(Axmb(Dop_dble_Re, Dop_dble_Im, updx_Re, updx_Im, updb_Re, updb_Im, ddtype))
        json_data['diAdip1'].append("unknown")

        print('cg_reset() rn({0}), tol({1}), xn({2}), rn={3}, tol={4} xn={5}'.format(type(rn), type(tol), type(xn), rn, tol, xn))
        print("i", json_data['i'][-1])
        print("reset", json_data['reset'][-1])
        print("rn", json_data['rn'][-1])
        print("Axmb", json_data['Axmb'][-1])
        print("diAdip1", json_data['diAdip1'][-1])

        if status >= nmx and rn > tol:
            status = -1
            break

        if (100.0*get_epsilon(ddtype)*xn) > tol:
            status = -2
            break

    json_data['status'] = status
    json_data['x_Re'] = pdx_Re.tolist()
    json_data['x_Im'] = pdx_Im.tolist()
    return pdx_Re, pdx_Im, rn**2, json_data

def Ax(Dop_Re, Dop_Im, x_Re, x_Im, dtype):
    Dx_Re, Dx_Im = complex_matrix_vector_prod(Dop_Re, Dop_Im, x_Re, x_Im, dtype)
    return complex_matrix_dagger_vector_prod(Dop_Re, Dop_Im, Dx_Re, Dx_Im, dtype)

def complex_abs(re, im, dtype):
    return np.sqrt(re*re + im*im)

def Axmb_vec(Dop_Re, Dop_Im, x_Re, x_Im, b_Re, b_Im, dtype):
    DDx_Re, DDx_Im = Ax(Dop_Re, Dop_Im, x_Re, x_Im, dtype)
    return axpy(-1.0, b_Re, b_Im, DDx_Re, DDx_Im, dtype)

def Axmb(Dop_Re, Dop_Im, x_Re, x_Im, b_Re, b_Im, dtype):
    DDx_Re, DDx_Im = Ax(Dop_Re, Dop_Im, x_Re, x_Im, dtype)
    res_Re, res_Im = axpy(-1.0, DDx_Re, DDx_Im, b_Re, b_Im, dtype)
    return np.sqrt(complex_norm_sq(res_Re, res_Im, dtype))

def complex_matrix_vector_prod(A, B, x, y, dtype):
    return A.dot(x) - B.dot(y), B.dot(x) + A.dot(y)

def complex_matrix_dagger_vector_prod(A, B, x, y, dtype):
    A = A.transpose()
    B = B.transpose()
    x, y = A.dot(x) + B.dot(y), A.dot(y) - B.dot(x)
    A = A.transpose()
    B = B.transpose()
    return x, y

def complex_scalar_prod(re1, im1, re2, im2, dtype=np.float64):
    return real_scalar_prod(re1, re2, dtype) + real_scalar_prod(im1, im2, dtype), real_scalar_prod(re1, im2, dtype) - real_scalar_prod(im1, re2, dtype)

def real_scalar_prod(x, y, dtype=np.float64):
    #if False: # for posits in naive calculation
    if dtype == sp.posit32 or dtype == sp.posit16 or dtype == sp.posit8:
        if dtype == sp.posit32:
            quire = sp.quire32()
        elif dtype == sp.posit16:
            quire = sp.quire16()
        elif dtype == sp.posit8:
            quire = sp.quire8()

        for i in range(0,len(x)):
            # quire.qma(a, b)
            # quire = quire + a*b
            # rounding deferred to the last step
            quire.qma(convert(x[i], dtype), convert(y[i], dtype))
        return quire.toPosit()
    else:
        n = dtype(0.0)
        for i in range(0,len(x)):
            n += convert(x[i], dtype)*convert(y[i], dtype)
        return convert(n, dtype)

def complex_norm_sq(re, im, dtype=np.float64):
    return real_norm_sq(re, dtype) + real_norm_sq(im, dtype)

def real_norm_sq(x, dtype=np.float64):
    return real_scalar_prod(x, x, dtype)

def complex_axpy(a_Re, a_Im, x_Re, x_Im, y_Re, y_Im, dtype):
    a_Re = convert(a_Re, dtype)
    a_Im = convert(a_Im, dtype)
    return np.array([a_Re*xr-a_Im*xi+yr for xr, xi, yr in zip(x_Re, x_Im, y_Re)]), np.array([a_Re*xi+a_Im*xr+yi for xr, xi, yi in zip(x_Re, x_Im, y_Im)])

# a*x + y
def axpy(a, x1, x2, y1, y2, dtype):
    a = convert(a, dtype)
    return np.array([a*xi+yi for xi, yi in zip(x1, y1)]), np.array([a*xi+yi for xi, yi in zip(x2, y2)])

class csc_matrix:
    def __init__(self, arrs, dtype):
        self.dtype = dtype
        self.cdata = np.array(arrs[0], dtype=self.dtype)
        self.row_index = np.array(arrs[1])
        self.col_ptr = np.array(arrs[2])
        self.N = self.col_ptr.size -1 # it's a NxN matrix
        self.zero = dtype(0.0)

    # i = row, j = column
    def getij(self, i, j):
        col_start = self.col_ptr[j]
        col_end =   self.col_ptr[j+1]
        #non_zero_row_indices = self.row_index[col_start:col_end] # all row indices of the nonzero values in the j-th column
        #non_zero_column_data = self.cdata[col_start:col_end] # all values in the j-th column that are nonzero
        if i in self.row_index[col_start:col_end]:
            # return nonzero value
            idx = np.where(self.row_index[col_start:col_end] == i)[0][0]
            return np.float64(self.cdata[col_start:col_end][idx])
        else:
            return np.float64(0.0) #self.zero

    # i = row, j = column
    def getrowi(self, i):
        row = np.zeros(self.N, dtype=self.dtype)
        for j in range(0,self.N):
            row[j] = self.getij(i, j)
        return row

    def asarray(self):
        A = np.zeros((self.N,self.N), dtype=self.dtype)
        for i in range(0,self.N):
            A[i] = self.getrowi(i)
        return A

    def getcsc(self):
        return [self.cdata, self.row_index, self.col_ptr]

    def getcsr(self):
        tr_data = np.array([], dtype=self.dtype)
        tr_row_index = np.array([], dtype=int)
        tr_col_ptr = np.array([int(0)], dtype=int) # ptr always starts with 0
        nz = 0

        k = int(self.N/100) if int(self.N/100) != 0 else 2
        # iterate row-wise
        for i in range(0,self.N):
            if i % k == 0:
                print("{0}%, {1}/{2}".format(int(round(i*100/self.N)), i, self.N), end="\r")
            row = self.getrowi(i)
            idx = np.nonzero(row) # all indices of nonzero elements in row
            tr_data = np.append(tr_data, [self.dtype(x) for x in row[idx]])
            tr_row_index = np.append(tr_row_index, idx[0])
            nz = tr_col_ptr[-1] + idx[0].size
            tr_col_ptr = np.append(tr_col_ptr, nz)

        tr_data = np.array(tr_data, dtype=self.dtype)
        tr_row_index = np.array(tr_row_index)
        tr_col_ptr = np.array(tr_col_ptr)
        return [tr_data, tr_row_index, tr_col_ptr]

    def transpose(self):
        return csc_matrix(self.getcsr(), self.dtype)

class csr_matrix:
    def __init__(self, arrs, dtype):
        self.dtype = dtype
        self.rdata = np.array(arrs[0], dtype=self.dtype)
        self.col_index = np.array(arrs[1])
        self.row_ptr = np.array(arrs[2])
        self.N = self.row_ptr.size -1 # it's a NxN matrix
        self.zero = dtype(0.0)
        self.transposed = False
        # transposed arrays
        self.tr_rdata = np.array([], dtype=self.dtype)
        self.tr_col_index = np.array([])
        self.tr_row_ptr = np.array([])
        self.tr_calculated = False

    def transpose(self):
        # calc transposed array only if not yet calculated
        if self.tr_calculated == False:
            self.tr_rdata, self.tr_col_index, self.tr_row_ptr = csc_matrix([self.rdata, self.col_index, self.row_ptr], self.dtype).getcsr()
            self.tr_calculated = True

        self.transposed = not self.transposed
        return self

    def dot(self, x):
        #y = np.zeros(self.N, dtype=self.dtype)
        dtype_zero = self.dtype(0.0)
        y = np.array([dtype_zero for x in np.zeros(self.N, dtype=self.dtype)])
        if self.transposed:
            ptr = self.tr_row_ptr
            data = self.tr_rdata
            index = self.tr_col_index
        else:
            ptr = self.row_ptr
            data = self.rdata
            index = self.col_index

        #if False: # for naive calculation
        if self.dtype == sp.posit32 or self.dtype == sp.posit16 or self.dtype == sp.posit8:
            if self.dtype == sp.posit32:
                quire = sp.quire32()
            elif self.dtype == sp.posit16:
                quire = sp.quire16()
            elif self.dtype == sp.posit8:
                quire = sp.quire8()

            for i in range(0,self.N):
                quire.clr()
                start = ptr[i]
                end = ptr[i+1]
                for j in range(start,end):
                    quire.qma(data[j], x[index[j]])
                y[i] = quire.toPosit()

        # regular operator overloading, slow with fl.tfloat32 and fl.bfloat16
        else:
            for i in range(0,self.N):
                start = ptr[i]
                end = ptr[i+1]
                s = self.dtype(0.0)
                for j in range(start,end):
                    s += data[j]*x[index[j]]
                y[i] = s

        return y


    def asarray(self):
        A = np.zeros((self.N,self.N), dtype=self.dtype)
        if self.transposed:
            for i in range(0,self.N):
                start = self.tr_row_ptr[i]
                end =   self.tr_row_ptr[i+1]
                A[i][self.tr_col_index[start:end]] = self.tr_rdata[start:end]
        else:
            for i in range(0,self.N):
                start = self.row_ptr[i]
                end =   self.row_ptr[i+1]
                A[i][self.col_index[start:end]] = self.rdata[start:end]
        return A

def get_size(obj, seen=None):
    # Recursively finds size of objects
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size

"""
p = sp.posit32(1.0)
bf = fl.bfloat16(1.0)
print(get_size(p))
print(get_size(bf))
exit()
"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Ax = b, with A = D^dagger D.')
    parser.add_argument('file', help='the input file containing the parameters and Dop() in CSR format.')
    parser.add_argument('-d', '--double', help='the input file contains Dop_dble() instead of Dop().', default=False, action="store_true")
    parser.add_argument('-s', '--simulate', help='Simulate the calculation in the given datatype.', default="binary64", choices=["binary16", "binary32", "binary64", "posit8", "posit16", "posit32", "bfloat16", "tfloat32"])
    parser.add_argument('-v', '--vol', help='spacetime volume.', type=int)
    parser.add_argument('-e', '--extension', help='extension level (0: no enabled extension, 1:constant alphas).', default=0, type=int)
    parser.add_argument('-n', '--nmx', help='maximal number of iterations.', type=int)
    parser.add_argument('-c', '--cdtype', help='datatype for the colletive variables.', default="binary64", choices=["binary16", "binary32", "binary64", "posit8", "posit16", "posit32", "bfloat16", "tfloat32"])
    parser.add_argument('-r', '--res', help='relative residue.', type=float)
    required = parser.add_argument_group('required named arguments')
    required.add_argument('-o', '--out', help='output json file.', required=True)
    args = parser.parse_args()

    int_size, float_size, double_size = 4, 4, 8 # bytes
    byteorder='little'

    entry_size = float_size
    dtype = np.float32
    itype = np.float32
    fmt = '<{0}f'
    if args.double:
        entry_size = double_size
        itype = np.float64
        dtype = np.float64
        fmt = '<{0}d'

    ddtype = np.float64 # "big" datatype

    # "small" datatype
    if args.simulate == "binary16":
        dtype = np.float16
    elif args.simulate == "binary32":
        dtype = np.float32
    elif args.simulate == "binary64":
        dtype = np.float64
    elif args.simulate == "posit8":
        dtype = sp.posit8
    elif args.simulate == "posit16":
        dtype = sp.posit16
    elif args.simulate == "posit32":
        dtype = sp.posit32
    elif args.simulate == "bfloat16":
        dtype = fl.bfloat16
    elif args.simulate == "tfloat32":
        dtype = fl.tfloat32

    #cdtype = np.float64 # datatype for collective variables
    if args.cdtype == "binary16":
        cdtype = np.float16
    elif args.cdtype == "binary32":
        cdtype = np.float32
    elif args.cdtype == "binary64":
        cdtype = np.float64
    elif args.cdtype == "posit8":
        cdtype = sp.posit8
    elif args.cdtype == "posit16":
        cdtype = sp.posit16
    elif args.cdtype == "posit32":
        cdtype = sp.posit32
    elif args.cdtype == "bfloat16":
        cdtype = fl.bfloat16
    elif args.cdtype == "tfloat32":
        cdtype = fl.tfloat32


    with open(args.file, "rb") as f:
        lastime = time.time()
        print("retrieving parameters ...")
        vol = struct.unpack('<1I', f.read(int_size))[0] # 1 unsigned int I, < little endian
        N = 12*vol
        nmx = struct.unpack('<1I', f.read(int_size))[0]
        res = struct.unpack('<1d', f.read(double_size))[0] # 1 double
        eta = struct.unpack('<{0}d'.format(2*N), f.read(double_size*2*N)) # 24*vol doubles
        psi = struct.unpack('<{0}d'.format(2*N), f.read(double_size*2*N)) # 24*vol doubles

        if args.nmx != None:
            nmx = args.nmx
        if args.vol != None:
            vol = args.vol
            N = 12*vol

        if args.res != None:
            res = args.res

        print('vol={0}, nmx={1}, res={2}, N={3}'.format(vol, nmx, res, N))

        # b = b_Re + i*b_Im
        b_Re = np.array([dtype(x) for x in eta[:2*N:2]], dtype=dtype)
        b_Im = np.array([dtype(x) for x in eta[1:2*N:2]], dtype=dtype)
        b_dble_Re = np.array([ddtype(x) for x in eta[:2*N:2]], dtype=ddtype)
        b_dble_Im = np.array([ddtype(x) for x in eta[1:2*N:2]], dtype=ddtype)

        # x0 = 0
        x0_Re = np.zeros(N, dtype=dtype)
        x0_Im = np.zeros(N, dtype=dtype)
        x0_dble_Re = np.zeros(N, dtype=ddtype)
        x0_dble_Im = np.zeros(N, dtype=ddtype)

        print("time", time.time(), time.time() - lastime); lastime = time.time();
        print("retrieving Dirac matrix in csr format (Re,dble) ...")

        # length numbers, data, index and ptr of Dop (Re)
        length = struct.unpack('<1I', f.read(int_size))[0]
        data = np.array(struct.unpack(fmt.format(length), f.read(entry_size*length)))
        length = struct.unpack('<1I', f.read(int_size))[0]
        index = struct.unpack('<{0}I'.format(length), f.read(int_size*length))
        length = struct.unpack('<1I', f.read(int_size))[0]
        ptr = struct.unpack('<{0}I'.format(length), f.read(int_size*length))

        # length numbers, tr_data, tr_index and tr_ptr of Dop (Re)
        length = struct.unpack('<1I', f.read(int_size))[0]
        tr_data = np.array([ddtype(x) for x in struct.unpack(fmt.format(length), f.read(entry_size*length))], dtype=ddtype)
        length = struct.unpack('<1I', f.read(int_size))[0]
        tr_index = np.array(struct.unpack('<{0}I'.format(length), f.read(int_size*length)))
        length = struct.unpack('<1I', f.read(int_size))[0]
        tr_ptr = np.array(struct.unpack('<{0}I'.format(length), f.read(int_size*length)))

        # creating the ddtype object
        csr_Dop_dble_Re = csr_matrix([data, index, ptr], ddtype)
        csr_Dop_dble_Re.tr_rdata, csr_Dop_dble_Re.tr_col_index, csr_Dop_dble_Re.tr_row_ptr = tr_data, tr_index, tr_ptr
        csr_Dop_dble_Re.tr_calculated = True

        print("time", time.time(), time.time() - lastime); lastime = time.time();
        print("converting Dirac matrix in csr format (Re) to sdtype ...")

        # creating the dtype object
        data2 = np.array([dtype(x) for x in data])
        tr_data2 = np.array([dtype(x) for x in tr_data])
        csr_Dop_Re = csr_matrix([data2, index, ptr], dtype)
        csr_Dop_Re.tr_rdata, csr_Dop_Re.tr_col_index, csr_Dop_Re.tr_row_ptr = tr_data2, tr_index, tr_ptr
        csr_Dop_Re.tr_calculated = True

        print("time", time.time(), time.time() - lastime); lastime = time.time();
        print("retrieving Dirac matrix in csr format (Im,dble) ...")

        # length numbers, data, index and ptr of Dop (Re)
        length = struct.unpack('<1I', f.read(int_size))[0]
        data = struct.unpack(fmt.format(length), f.read(entry_size*length))
        length = struct.unpack('<1I', f.read(int_size))[0]
        index = struct.unpack('<{0}I'.format(length), f.read(int_size*length))
        length = struct.unpack('<1I', f.read(int_size))[0]
        ptr = struct.unpack('<{0}I'.format(length), f.read(int_size*length))

        # length numbers, tr_data, tr_index and tr_ptr of Dop (Re)
        length = struct.unpack('<1I', f.read(int_size))[0]
        tr_data = np.array([ddtype(x) for x in struct.unpack(fmt.format(length), f.read(entry_size*length))], dtype=ddtype)
        length = struct.unpack('<1I', f.read(int_size))[0]
        tr_index = np.array(struct.unpack('<{0}I'.format(length), f.read(int_size*length)))
        length = struct.unpack('<1I', f.read(int_size))[0]
        tr_ptr = np.array(struct.unpack('<{0}I'.format(length), f.read(int_size*length)))

        # creating the object
        csr_Dop_dble_Im = csr_matrix([data, index, ptr], ddtype)
        csr_Dop_dble_Im.tr_rdata, csr_Dop_dble_Im.tr_col_index, csr_Dop_dble_Im.tr_row_ptr = tr_data, tr_index, tr_ptr
        csr_Dop_dble_Im.tr_calculated = True

        print("time", time.time(), time.time() - lastime); lastime = time.time();
        print("converting Dirac matrix in csr format (Im) to sdtype ...")

        # creating the dtype object
        data2 = np.array([dtype(x) for x in data])
        tr_data2 = np.array([dtype(x) for x in tr_data])
        csr_Dop_Im = csr_matrix([data2, index, ptr], dtype)
        csr_Dop_Im.tr_rdata, csr_Dop_Im.tr_col_index, csr_Dop_Im.tr_row_ptr = tr_data2, tr_index, tr_ptr
        csr_Dop_Im.tr_calculated = True
        print("time", time.time(), time.time() - lastime); lastime = time.time();

        print("transposing Dop (Re) (1/2) ...")
        csr_Dop_Re.transpose()
        csr_Dop_Re.transpose()
        print("time", time.time(), time.time() - lastime); lastime = time.time();
        print("transposing Dop (Im) (1/2) ...")
        csr_Dop_Im.transpose()
        csr_Dop_Im.transpose()
        print("time", time.time(), time.time() - lastime); lastime = time.time();
        print("transposing Dop_dble (Re) (2/2) ...")
        csr_Dop_dble_Re.transpose()
        csr_Dop_dble_Re.transpose()
        print("time", time.time(), time.time() - lastime); lastime = time.time();
        print("transposing Dop_dble (Im) (2/2) ...")
        csr_Dop_dble_Im.transpose()
        csr_Dop_dble_Im.transpose()
        print("time", time.time(), time.time() - lastime); lastime = time.time();

        #print(Dop_Re)
        #print(csr_Dop_Re.asarray())
        #print(Dop_Re.shape)
        #print(csr_Dop_Re.asarray().shape)
        #print(np.array_equal(Dop_Re.transpose(), csr_Dop_dble_Re.transpose().asarray()))
        #print(np.array_equal(Dop_Re, csr_Dop_Re.asarray()))
        #print(np.array_equal(Dop_Re, csr_Dop_Re.transpose().transpose().asarray()))
        #print(np.array_equal(csr_Dop_Re.asarray(), csr_Dop_Re.transpose().transpose().asarray()))
        #exit()

        print("time", time.time(), time.time() - lastime); lastime = time.time();
        print("calculating |b|^2 ...")
        norm_sq = complex_norm_sq(b_Re, b_Im, dtype)
        print(type(norm_sq), norm_sq)
        print("time", time.time(), time.time() - lastime); lastime = time.time();

        print("calculating axpy ...")
        r_Re, r_Im = axpy(0.123123, b_Re, b_Im, b_Re, b_Im, dtype)
        print(type(r_Re), type(r_Re[0]))
        print("time", time.time(), time.time() - lastime); lastime = time.time();

        print("calculating r = D b (1/2)...")
        r_Re, r_Im = complex_matrix_vector_prod(csr_Dop_Re, csr_Dop_Im, b_Re, b_Im, dtype)
        print(type(r_Re), type(r_Re[0]))
        print("time", time.time(), time.time() - lastime); lastime = time.time();

        print("calculating r = D^dagger b (1/2)...")
        r_Re, r_Im = complex_matrix_dagger_vector_prod(csr_Dop_Re, csr_Dop_Im, b_Re, b_Im, dtype)
        print(type(r_Re), type(r_Re[0]))
        print("time", time.time(), time.time() - lastime); lastime = time.time();

        print("calculating r = D r (2/2)...")
        r_Re, r_Im = complex_matrix_vector_prod(csr_Dop_Re, csr_Dop_Im, r_Re, r_Im, dtype)
        print(type(r_Re), type(r_Re[0]))
        print("time", time.time(), time.time() - lastime); lastime = time.time();

        print("calculating r = D^dagger r (2/2)...")
        r_Re, r_Im = complex_matrix_dagger_vector_prod(csr_Dop_Re, csr_Dop_Im, r_Re, r_Im, dtype)
        print(type(r_Re), type(r_Re[0]))
        print("time", time.time(), time.time() - lastime); lastime = time.time();

        print("calculating Ax=b ...")
        x_Re, x_Im, rn, json_data = cgne(vol, nmx, res, csr_Dop_Re, csr_Dop_Im, csr_Dop_dble_Re, csr_Dop_dble_Im, b_dble_Re, b_dble_Im, x0_dble_Re, x0_dble_Im, dtype, ddtype, cdtype, args.extension)
        print(rn, type(x_Re[0]), type(x_Im[0]))
        print("status", json_data['status'])
        print("time", time.time(), time.time() - lastime); lastime = time.time();

        print("calculating |A_dble x - b| ...")
        res = Axmb(csr_Dop_dble_Re, csr_Dop_dble_Im, x_Re, x_Im, b_dble_Re, b_dble_Im, ddtype)
        print("--> res", type(res), res)
        print("time", time.time(), time.time() - lastime); lastime = time.time();

        if args.out == "-":
            json.dump(json_data, sys.stdout, indent=2)
            print("\n")
        else:
            with open(args.out, 'w') as outfile:
                json.dump(json_data, outfile, indent=2)

    os.system('speaker-test -t pink -f 1000 -p1 -l1')
    os.system('beep')
