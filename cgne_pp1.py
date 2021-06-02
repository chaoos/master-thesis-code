#!/usr/bin/env python3
#
# convert Dop_dble() from raw to csr format

import argparse
import struct # struct.unpack(), struct.pack()
import numpy as np # np.asarray, np.zeros(), np.float16, np.float32, np.float64, np.linalg.norm()
import os # os.remove()
import sys
import time
from scipy import sparse
import scipy as sc

# The input binary file contains all needed numbers for cgne()
# the layout looks as follows
# vol nmx res
# eta[0] ... eta[vol*24]
# psi[0] ... psi[vol*24]
# D[0,0]        D[0,1]      ... D[0,vol*24]
# D[1,0]        D[1,1]      ... D[1,vol*24]
# ...
# D[vol*24,0]   D[24*vol,1] ... D[vol*24,vol*24]
#-------------------------------------------------------------------------------------------
# The output binary file contains Dop_dble_Re and Dop_dble_Im (+ both transposed) in csr format
# the layout looks as follows
# vol nmx res
# eta[0] ... eta[vol*24]
# psi[0] ... psi[vol*24]
# Re_len_rdata Re_rdata[0] ... Re_rdata[Re_len_data-1]
# Re_len_col_index Re_col_index[0] ... Re_col_index[Re_len_col_index-1]
# Re_len_row_ptr Re_row_ptr[0] ... Re_row_ptr[Re_len_row_ptr-1]
# Re_len_tr_rdata Re_tr_rdata[0] ... Re_tr_rdata[Re_len_tr_data-1]
# Re_len_tr_col_index Re_tr_col_index[0] ... Re_tr_col_index[Re_len_tr_col_index-1]
# Re_len_tr_row_ptr Re_tr_row_ptr[0] ... Re_tr_row_ptr[Re_len_tr_row_ptr-1]
# Im_len_rdata Im_rdata[0] ... Im_rdata[Im_len_data-1]
# Im_len_col_index Im_col_index[0] ... Im_col_index[Im_len_col_index-1]
# Im_len_row_ptr Im_row_ptr[0] ... Im_row_ptr[Im_len_row_ptr-1]
# Im_len_tr_rdata Im_tr_rdata[0] ... Im_tr_rdata[Im_len_tr_data-1]
# Im_len_tr_col_index Im_tr_col_index[0] ... Im_tr_col_index[Im_len_tr_col_index-1]
# Im_len_tr_row_ptr Im_tr_row_ptr[0] ... Im_tr_row_ptr[Im_len_tr_row_ptr-1]

def convert(x, dtype):
    if type(x) == dtype:
        return x
    if dtype == sp.posit32 or dtype == sp.posit16 or dtype == sp.posit8:
        return dtype(np.float64(x))
    else:
        return dtype(x)

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
        y = np.zeros(self.N, dtype=self.dtype)
        if self.transposed:
            ptr = self.tr_row_ptr
            data = self.tr_rdata
            index = self.tr_col_index
        else:
            ptr = self.row_ptr
            data = self.rdata
            index = self.col_index

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

"""
print("##### TESTS (1/2) #####")
a = csr_matrix([[5, 8, 3, 6], [0, 1, 2, 1], [0, 1, 2, 3, 4]], np.float64)
b = np.array([[5, 0, 0, 0],
              [0, 8, 0, 0],
              [0, 0, 3, 0],
              [0, 6, 0, 0]], dtype=np.float64)

print("should be true: ", np.array_equal(a.asarray(), b))
a.transpose()
print("should be true: ", np.array_equal(a.asarray(), b.transpose()))
a.transpose()
print("should be true: ", np.array_equal(a.asarray(), b))
exit()
"""

def waitForRead(file, size, fifo=False):
    if fifo:
        while True:
            data = file.read(size)
            if len(data) == 0:
                return None
            return data
    else:
        return file.read(size)

parser = argparse.ArgumentParser(description='convert Dop_dble() from raw to csr format.')
parser.add_argument('-v', '--vol', help='spacetime volume.', type=int)
required = parser.add_argument_group('required named arguments')
required.add_argument('-i', '--infile', help='input file.', required=True)
required.add_argument('-o', '--outfile', help='output file.', required=True)
required.add_argument('-t', '--temp', help='temporary file path with enough space.', required=True)
args = parser.parse_args()

int_size, float_size, double_size = 4, 4, 8 # bytes

entry_size = double_size
itype = np.float64
otype = np.float64
ifmt = '<{0}d'
ofmt = '<{0}d'

with open(args.infile, "rb") as f:
    lastime = time.time()
    print("retrieving parameters ...")
    vol = struct.unpack('<1I', f.read(int_size))[0] # 1 unsigned int I, < little endian
    #vol = struct.unpack('<1I', waitForRead(f, int_size, args.fifo))[0] # 1 unsigned int I, < little endian
    N = 12*vol
    nmx = struct.unpack('<1I', f.read(int_size))[0]
    res = struct.unpack('<1d', f.read(double_size))[0] # 1 double
    eta = struct.unpack('<{0}d'.format(2*N), f.read(double_size*2*N)) # 24*vol doubles
    psi = struct.unpack('<{0}d'.format(2*N), f.read(double_size*2*N)) # 24*vol doubles

    if args.vol != None:
        vol = args.vol
        N = 12*vol

    eta = np.array([otype(x) for x in eta[:2*N]], dtype=otype)
    psi = np.array([otype(x) for x in psi[:2*N]], dtype=otype)

    print('vol={0}, nmx={1}, res={2}, N={3}'.format(vol, nmx, res, N))

    print("time", time.time(), time.time() - lastime); lastime = time.time();
    print("retrieving Dirac matrix in csc format (from input file or pipe) ...")

    csc_cptr_Re = np.zeros(N+1, dtype=int)
    csc_cptr_Im = np.zeros(N+1, dtype=int)

    t1p = '{0}/tmp_data_Re.dat'.format(args.temp)
    t2p = '{0}/tmp_data_Im.dat'.format(args.temp)
    t3p = '{0}/tmp_rindex_Re.dat'.format(args.temp)
    t4p = '{0}/tmp_rindex_Im.dat'.format(args.temp)
    if os.path.exists(t1p) or os.path.exists(t2p) or os.path.exists(t3p) or os.path.exists(t4p):
        print('At least one of {0}, {1}, {2}, {3} already exists, you should remove them, exitting ...'.format(t1p, t2p, t3p, t4p))
        exit()
    t1 = open(t1p, "wb")
    t2 = open(t2p, "wb")
    t3 = open(t3p, "wb")
    t4 = open(t4p, "wb")
    t1s = 0
    t2s = 0
    t3s = 0
    t4s = 0

    #csc_data_dble_Re = np.array([], dtype=otype)
    #csc_data_dble_Im = np.array([], dtype=otype)
    #csc_rindex_Re = np.array([], dtype=int)
    #csc_rindex_Im = np.array([], dtype=int)
    k = int(N/100) if int(N/100) != 0 else 2
    # iterate column wise
    for j in range(0, N):
        if j % k == 0:
            print("time", time.time(), time.time() - lastime); lastime = time.time();
            print("{0}%, {1}/{2}".format(int(round(j*100/N)), j, N), end="\n")

        # 24N numbers, one column of Dop (alternating real and imaginary parts)
        dj = struct.unpack(ifmt.format(2*N), f.read(entry_size*2*N))

        column_Re = np.array(dj[::2]) # only even indices, real part
        column_Im = np.array(dj[1::2]) # only odd indices, imag part
        idx_Re = np.nonzero(column_Re) # all indices of nonzero elements in column_Re
        idx_Im = np.nonzero(column_Im) # all indices of nonzero elements in column_Im
        #csc_data_dble_Re = np.append(csc_data_dble_Re, [otype(x) for x in column_Re[idx_Re]])
        #csc_data_dble_Im = np.append(csc_data_dble_Im, [otype(x) for x in column_Im[idx_Im]])

        # store the stuff in the temp files
        t1d = np.array([otype(x) for x in column_Re[idx_Re]])
        t2d = np.array([otype(x) for x in column_Im[idx_Im]])
        t3d = idx_Re[0]
        t4d = idx_Im[0]
        t1.write(struct.pack('<{0}d'.format(t1d.size), *t1d))
        t2.write(struct.pack('<{0}d'.format(t2d.size), *t2d))
        t3.write(struct.pack('<{0}I'.format(t3d.size), *t3d))
        t4.write(struct.pack('<{0}I'.format(t4d.size), *t4d))
        t1s += t1d.size
        t2s += t2d.size
        t3s += t3d.size
        t4s += t4d.size

        #csc_rindex_Re = np.append(csc_rindex_Re, idx_Re[0])
        #csc_rindex_Im = np.append(csc_rindex_Im, idx_Im[0])
        csc_cptr_Re[j+1] = csc_cptr_Re[j] + idx_Re[0].size
        csc_cptr_Im[j+1] = csc_cptr_Im[j] + idx_Im[0].size

    t1.close()
    t2.close()
    t3.close()
    t4.close()

    print("time", time.time(), time.time() - lastime); lastime = time.time();
    print("retrieving Dirac matrix in csc format (from temp files)...")

    t1 = open(t1p, "rb")
    t2 = open(t2p, "rb")
    t3 = open(t3p, "rb")
    t4 = open(t4p, "rb")

    csc_data_dble_Re = np.array(struct.unpack('<{0}d'.format(t1s), t1.read(double_size*t1s)))
    csc_data_dble_Im = np.array(struct.unpack('<{0}d'.format(t2s), t2.read(double_size*t2s)))
    csc_rindex_Re    = np.array(struct.unpack('<{0}I'.format(t3s), t3.read(int_size*t3s)))
    csc_rindex_Im    = np.array(struct.unpack('<{0}I'.format(t4s), t4.read(int_size*t4s)))

    t1.close()
    t2.close()
    t3.close()
    t4.close()

    # in row major format
    print("time", time.time(), time.time() - lastime); lastime = time.time();
    print(csc_data_dble_Re.size, csc_data_dble_Im.size)

    print("converting Dop_dble (Re) from csc to csr format (+transposing) (1/2)...")
    #csr_Dop_dble_Re = csc_matrix([csc_data_dble_Re, csc_rindex_Re, csc_cptr_Re], otype)
    #csr_Dop_dble_Re = csr_matrix(csr_Dop_dble_Re.getcsr(), otype)
    #csr_Dop_dble_Re.tr_rdata, csr_Dop_dble_Re.tr_col_index, csr_Dop_dble_Re.tr_row_ptr = csc_data_dble_Re, csc_rindex_Re, csc_cptr_Re
    #csr_Dop_dble_Re.tr_calculated = True
    sc_Dop_dble_Re_transposed = sc.sparse.csc_matrix((csc_data_dble_Re, csc_rindex_Re, csc_cptr_Re), dtype=otype)
    sc_Dop_dble_Re = sc_Dop_dble_Re_transposed.tocsr(copy=True)
    csr_Dop_dble_Re = csr_matrix([sc_Dop_dble_Re.data, sc_Dop_dble_Re.indices, sc_Dop_dble_Re.indptr], otype)
    csr_Dop_dble_Re.tr_rdata = sc_Dop_dble_Re_transposed.data
    csr_Dop_dble_Re.tr_col_index = sc_Dop_dble_Re_transposed.indices
    csr_Dop_dble_Re.tr_row_ptr = sc_Dop_dble_Re_transposed.indptr
    csr_Dop_dble_Re.tr_calculated = True

    print("time", time.time(), time.time() - lastime); lastime = time.time();
    print("converting Dop_dble (Im) from csc to csr format (+transposing) (1/2)...")
    #csr_Dop_dble_Im = csc_matrix([csc_data_dble_Im, csc_rindex_Im, csc_cptr_Im], otype)
    #csr_Dop_dble_Im = csr_matrix(csr_Dop_dble_Im.getcsr(), otype)
    #csr_Dop_dble_Im.tr_rdata, csr_Dop_dble_Im.tr_col_index, csr_Dop_dble_Im.tr_row_ptr = csc_data_dble_Im, csc_rindex_Im, csc_cptr_Im
    #csr_Dop_dble_Im.tr_calculated = True
    sc_Dop_dble_Im_transposed = sc.sparse.csc_matrix((csc_data_dble_Im, csc_rindex_Im, csc_cptr_Im), dtype=otype)
    sc_Dop_dble_Im = sc_Dop_dble_Im_transposed.tocsr(copy=True)
    csr_Dop_dble_Im = csr_matrix([sc_Dop_dble_Im.data, sc_Dop_dble_Im.indices, sc_Dop_dble_Im.indptr], otype)
    csr_Dop_dble_Im.tr_rdata = sc_Dop_dble_Im_transposed.data
    csr_Dop_dble_Im.tr_col_index = sc_Dop_dble_Im_transposed.indices
    csr_Dop_dble_Im.tr_row_ptr = sc_Dop_dble_Im_transposed.indptr
    csr_Dop_dble_Im.tr_calculated = True
    print("time", time.time(), time.time() - lastime); lastime = time.time();

    print("transposing Dop_dble (Re) (2/2) ...")
    csr_Dop_dble_Re.transpose()
    csr_Dop_dble_Re.transpose()
    print("time", time.time(), time.time() - lastime); lastime = time.time();
    print("transposing Dop_dble (Im) (2/2) ...")
    csr_Dop_dble_Im.transpose()
    csr_Dop_dble_Im.transpose()
    print("time", time.time(), time.time() - lastime); lastime = time.time();

    with open(args.outfile, 'wb') as outfile:
        print("storing parameters ...")
        outfile.write(struct.pack('<1I', vol))
        outfile.write(struct.pack('<1I', nmx))
        outfile.write(struct.pack('<1d', res))
        outfile.write(struct.pack(ofmt.format(2*N), *eta))
        outfile.write(struct.pack(ofmt.format(2*N), *psi))
        print("time", time.time(), time.time() - lastime); lastime = time.time();

        print("storing Dirac matrix in csr format ...")
        # Re_len_rdata Re_rdata[0] ... Re_rdata[Re_len_data-1]
        # Re_len_col_index Re_col_index[0] ... Re_col_index[Re_len_col_index-1]
        # Re_len_row_ptr Re_row_ptr[0] ... Re_row_ptr[Re_len_row_ptr-1]
        outfile.write(struct.pack('<1I', csr_Dop_dble_Re.rdata.size))
        outfile.write(struct.pack(ofmt.format(csr_Dop_dble_Re.rdata.size), *csr_Dop_dble_Re.rdata))
        outfile.write(struct.pack('<1I', csr_Dop_dble_Re.col_index.size))
        outfile.write(struct.pack('<{0}I'.format(csr_Dop_dble_Re.col_index.size), *csr_Dop_dble_Re.col_index))
        outfile.write(struct.pack('<1I', csr_Dop_dble_Re.row_ptr.size))
        outfile.write(struct.pack('<{0}I'.format(csr_Dop_dble_Re.row_ptr.size), *csr_Dop_dble_Re.row_ptr))

        # Re_len_tr_rdata Re_tr_rdata[0] ... Re_tr_rdata[Re_len_tr_data-1]
        # Re_len_tr_col_index Re_tr_col_index[0] ... Re_tr_col_index[Re_len_tr_col_index-1]
        # Re_len_tr_row_ptr Re_tr_row_ptr[0] ... Re_tr_row_ptr[Re_len_tr_row_ptr-1]
        outfile.write(struct.pack('<1I', csr_Dop_dble_Re.tr_rdata.size))
        outfile.write(struct.pack(ofmt.format(csr_Dop_dble_Re.tr_rdata.size), *csr_Dop_dble_Re.tr_rdata))
        outfile.write(struct.pack('<1I', csr_Dop_dble_Re.tr_col_index.size))
        outfile.write(struct.pack('<{0}I'.format(csr_Dop_dble_Re.tr_col_index.size), *csr_Dop_dble_Re.tr_col_index))
        outfile.write(struct.pack('<1I', csr_Dop_dble_Re.tr_row_ptr.size))
        outfile.write(struct.pack('<{0}I'.format(csr_Dop_dble_Re.tr_row_ptr.size), *csr_Dop_dble_Re.tr_row_ptr))

        # Im_len_rdata Im_rdata[0] ... Im_rdata[Im_len_data-1]
        # Im_len_col_index Im_col_index[0] ... Im_col_index[Im_len_col_index-1]
        # Im_len_row_ptr Im_row_ptr[0] ... Im_row_ptr[Im_len_row_ptr-1]
        outfile.write(struct.pack('<1I', csr_Dop_dble_Im.rdata.size))
        outfile.write(struct.pack(ofmt.format(csr_Dop_dble_Im.rdata.size), *csr_Dop_dble_Im.rdata))
        outfile.write(struct.pack('<1I', csr_Dop_dble_Im.col_index.size))
        outfile.write(struct.pack('<{0}I'.format(csr_Dop_dble_Im.col_index.size), *csr_Dop_dble_Im.col_index))
        outfile.write(struct.pack('<1I', csr_Dop_dble_Im.row_ptr.size))
        outfile.write(struct.pack('<{0}I'.format(csr_Dop_dble_Im.row_ptr.size), *csr_Dop_dble_Im.row_ptr))

        # Im_len_tr_rdata Im_tr_rdata[0] ... Im_tr_rdata[Im_len_tr_data-1]
        # Im_len_tr_col_index Im_tr_col_index[0] ... Im_tr_col_index[Im_len_tr_col_index-1]
        # Im_len_tr_row_ptr Im_tr_row_ptr[0] ... Im_tr_row_ptr[Im_len_tr_row_ptr-1]
        outfile.write(struct.pack('<1I', csr_Dop_dble_Im.tr_rdata.size))
        outfile.write(struct.pack(ofmt.format(csr_Dop_dble_Im.tr_rdata.size), *csr_Dop_dble_Im.tr_rdata))
        outfile.write(struct.pack('<1I', csr_Dop_dble_Im.tr_col_index.size))
        outfile.write(struct.pack('<{0}I'.format(csr_Dop_dble_Im.tr_col_index.size), *csr_Dop_dble_Im.tr_col_index))
        outfile.write(struct.pack('<1I', csr_Dop_dble_Im.tr_row_ptr.size))
        outfile.write(struct.pack('<{0}I'.format(csr_Dop_dble_Im.tr_row_ptr.size), *csr_Dop_dble_Im.tr_row_ptr))
        print("time", time.time(), time.time() - lastime); lastime = time.time();



exit()
print("##### TESTS (2/2) #####")

with open(args.infile, "rb") as f:
    lastime = time.time()
    print("retrieving parameters from infile ...")
    vol = struct.unpack('<1I', f.read(int_size))[0] # 1 unsigned int I, < little endian
    N = 12*vol
    nmx = struct.unpack('<1I', f.read(int_size))[0]
    res = struct.unpack('<1d', f.read(double_size))[0] # 1 double
    eta = struct.unpack('<{0}d'.format(2*N), f.read(double_size*2*N)) # 24*vol doubles
    psi = struct.unpack('<{0}d'.format(2*N), f.read(double_size*2*N)) # 24*vol doubles

    if args.vol != None:
        vol = args.vol
        N = 12*vol

    eta = np.array([otype(x) for x in eta[:2*N]], dtype=otype)
    psi = np.array([otype(x) for x in psi[:2*N]], dtype=otype)

    print('vol={0}, nmx={1}, res={2}, N={3}'.format(vol, nmx, res, N))

    print("time", time.time(), time.time() - lastime); lastime = time.time();
    print("retrieving Dirac matrix in raw format ...")

    Dop_Re = np.zeros((N,N), dtype=otype)
    Dop_Im = np.zeros((N,N), dtype=otype)
    k = int(N/100) if int(N/100) != 0 else 2
    # iterate column wise
    for j in range(0, N):
        if j % k == 0:
            print("{0}%, {1}/{2}".format(int(round(j*100/N)), j, N), end="\r")

        # 24N numbers, one column of Dop (alternating real and imaginary parts)
        dj = struct.unpack(ifmt.format(2*N), f.read(entry_size*2*N))

        # in column major format
        Dop_Re[j] = np.array([otype(x) for x in dj[::2]], dtype=otype) # only even indices
        Dop_Im[j] = np.array([otype(x) for x in dj[1::2]], dtype=otype) # only odd indices
    
    # in row major format
    Dop_Re = Dop_Re.transpose()
    Dop_Im = Dop_Im.transpose()

with open(args.outfile, "rb") as f:
    lastime = time.time()
    print("retrieving parameters from outfile ...")
    vol = struct.unpack('<1I', f.read(int_size))[0] # 1 unsigned int I, < little endian
    N = 12*vol
    nmx = struct.unpack('<1I', f.read(int_size))[0]
    res = struct.unpack('<1d', f.read(double_size))[0] # 1 double
    eta = struct.unpack('<{0}d'.format(2*N), f.read(double_size*2*N)) # 24*vol doubles
    psi = struct.unpack('<{0}d'.format(2*N), f.read(double_size*2*N)) # 24*vol doubles

    eta = np.array([otype(x) for x in eta], dtype=otype)
    psi = np.array([otype(x) for x in psi], dtype=otype)

    print('vol={0}, nmx={1}, res={2}, N={3}'.format(vol, nmx, res, N))
    print("time", time.time(), time.time() - lastime); lastime = time.time();

    print("retrieving Dirac matrix in csr format ...")

    # length numbers, data, index and ptr of Dop (Re)
    length = struct.unpack('<1I', f.read(int_size))[0]
    data = struct.unpack(ofmt.format(length), f.read(entry_size*length))
    length = struct.unpack('<1I', f.read(int_size))[0]
    index = struct.unpack('<{0}I'.format(length), f.read(int_size*length))
    length = struct.unpack('<1I', f.read(int_size))[0]
    ptr = struct.unpack('<{0}I'.format(length), f.read(int_size*length))

    # length numbers, tr_data, tr_index and tr_ptr of Dop (Re)
    length = struct.unpack('<1I', f.read(int_size))[0]
    tr_data = np.array([otype(x) for x in struct.unpack(ofmt.format(length), f.read(entry_size*length))], dtype=otype)
    length = struct.unpack('<1I', f.read(int_size))[0]
    tr_index = np.array(struct.unpack('<{0}I'.format(length), f.read(int_size*length)))
    length = struct.unpack('<1I', f.read(int_size))[0]
    tr_ptr = np.array(struct.unpack('<{0}I'.format(length), f.read(int_size*length)))

    # creating the object
    csr_Re = csr_matrix([data, index, ptr], otype)
    csr_Re.tr_rdata, csr_Re.tr_col_index, csr_Re.tr_row_ptr = tr_data, tr_index, tr_ptr
    csr_Re.tr_calculated = True

    # length numbers, data, index and ptr of Dop (Re)
    length = struct.unpack('<1I', f.read(int_size))[0]
    data = struct.unpack(ofmt.format(length), f.read(entry_size*length))
    length = struct.unpack('<1I', f.read(int_size))[0]
    index = struct.unpack('<{0}I'.format(length), f.read(int_size*length))
    length = struct.unpack('<1I', f.read(int_size))[0]
    ptr = struct.unpack('<{0}I'.format(length), f.read(int_size*length))

    # length numbers, tr_data, tr_index and tr_ptr of Dop (Re)
    length = struct.unpack('<1I', f.read(int_size))[0]
    tr_data = np.array([otype(x) for x in struct.unpack(ofmt.format(length), f.read(entry_size*length))], dtype=otype)
    length = struct.unpack('<1I', f.read(int_size))[0]
    tr_index = np.array(struct.unpack('<{0}I'.format(length), f.read(int_size*length)))
    length = struct.unpack('<1I', f.read(int_size))[0]
    tr_ptr = np.array(struct.unpack('<{0}I'.format(length), f.read(int_size*length)))

    # creating the object
    csr_Im = csr_matrix([data, index, ptr], otype)
    csr_Im.tr_rdata, csr_Im.tr_col_index, csr_Im.tr_row_ptr = tr_data, tr_index, tr_ptr
    csr_Im.tr_calculated = True

    print("transposing Dop_dble (Re) (2/2) ...")
    csr_Re.transpose()
    csr_Re.transpose()
    print("time", time.time(), time.time() - lastime); lastime = time.time();
    print("transposing Dop_dble (Im) (2/2) ...")
    csr_Im.transpose()
    csr_Im.transpose()
    print("time", time.time(), time.time() - lastime); lastime = time.time();

    print(Dop_Re.shape)
    print(csr_Re.asarray().shape)
    csr_Re.transpose()
    print("should be true: ", np.array_equal(Dop_Re.transpose(), csr_Re.asarray()))
    csr_Re.transpose()
    print("should be true: ", np.array_equal(Dop_Re, csr_Re.asarray()))
    print("should be true: ", np.array_equal(Dop_Re, csr_Re.transpose().transpose().asarray()))
    print("should be true: ", np.array_equal(csr_Re.asarray(), csr_Re.transpose().transpose().asarray()))
    print("should be true: ", np.array_equal(csr_Dop_dble_Re.asarray(), csr_Re.asarray()))
    print("should be true: ", np.array_equal(csr_Dop_dble_Re.transpose().asarray(), csr_Re.transpose().asarray()))

    print(Dop_Im.shape)
    print(csr_Im.asarray().shape)
    csr_Im.transpose()
    print("should be true: ", np.array_equal(Dop_Im.transpose(), csr_Im.asarray()))
    csr_Im.transpose()
    print("should be true: ", np.array_equal(Dop_Im, csr_Im.asarray()))
    print("should be true: ", np.array_equal(Dop_Im, csr_Im.transpose().transpose().asarray()))
    print("should be true: ", np.array_equal(csr_Im.asarray(), csr_Im.transpose().transpose().asarray()))
    print("should be true: ", np.array_equal(csr_Dop_dble_Im.asarray(), csr_Im.asarray()))
    print("should be true: ", np.array_equal(csr_Dop_dble_Im.transpose().asarray(), csr_Im.transpose().asarray()))

