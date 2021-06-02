#!/usr/bin/env python3
#
# convert Dop_dble() from csr format to A = D^dagger D in csr format

import argparse
import struct # struct.unpack(), struct.pack()
import numpy as np # np.asarray, np.zeros(), np.float16, np.float32, np.float64, np.linalg.norm()
import softposit as sp # sp.posit8, sp.posit16, sp.posit32
import floats as fl
import os # os.remove()
import sys
import time
from scipy import sparse
from cgne_pp2 import convert, csc_matrix
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

    # self*x
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
                for j in range(ptr[i],ptr[i+1]):
                    y[i] += data[j]*x[index[j]]

        return y

    # return i-th column
    def getcolumn(self, i):
        if not self.transposed:
            self.transpose()
            self.transpose()
        column = np.zeros(self.N, dtype=self.dtype)
        column_start = self.tr_row_ptr[i]
        column_end   = self.tr_row_ptr[i+1]
        column[self.tr_col_index[column_start:column_end]] = self.tr_rdata[column_start:column_end]
        return column

    # return i-th row
    def getrow(self, i):
        row = np.zeros(self.N, dtype=self.dtype)
        row_start = self.row_ptr[i]
        row_end   = self.row_ptr[i+1]
        row[self.col_index[row_start:row_end]] = self.rdata[row_start:row_end]
        return row

    # return self^T B
    def transpose_times(self, B):
        lastime = time.time()
        print("time", time.time(), time.time() - lastime); lastime = time.time();
        data = np.array([], dtype=self.dtype)
        cindex = np.array([], dtype=int)
        rptr = np.array([int(0)], dtype=int) # ptr always starts with 0
        nz = 0

        k = int(self.N/100) if int(self.N/100) != 0 else 2
        k = 1
        for i in range(0,self.N):
            print("time", time.time(), time.time() - lastime); lastime = time.time();
            if i % k == 0:
                print("{0}%, {1}/{2}".format(int(round(i*100/self.N)), i, self.N))
                #print("{0}%, {1}/{2}".format(int(round(i*100/self.N)), i, self.N), end="\r")

            col = self.getcolumn(i)
            B.transpose()
            row = B.dot(col) # the i-th row of the resulting matrix
            B.transpose()

            idx = np.nonzero(row) # all indices of nonzero elements in row
            data = np.append(data, [self.dtype(x) for x in row[idx]])
            cindex = np.append(cindex, idx[0])
            nz = rptr[-1] + idx[0].size
            rptr = np.append(rptr, nz)

        return csr_matrix([data, cindex, rptr], self.dtype)

    # return self + c*B
    def plus(self, c, B):
        c = self.dtype(c)
        data = np.array([], dtype=self.dtype)
        cindex = np.array([], dtype=int)
        rptr = np.array([int(0)], dtype=int) # ptr always starts with 0
        nz = 0

        k = int(self.N/100) if int(self.N/100) != 0 else 2
        for i in range(0,self.N):
            if i % k == 0:
                print("{0}%, {1}/{2}".format(int(round(i*100/self.N)), i, self.N), end="\r")

            rowA = self.getrow(i)
            rowB = B.getrow(i)
            row = rowA + c*rowB

            idx = np.nonzero(row) # all indices of nonzero elements in row
            data = np.append(data, [self.dtype(x) for x in row[idx]])
            cindex = np.append(cindex, idx[0])
            nz = rptr[-1] + idx[0].size
            rptr = np.append(rptr, nz)

        return csr_matrix([data, cindex, rptr], self.dtype)

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


parser = argparse.ArgumentParser(description='convert Dop_dble() from csr format to A = D^dagger D in csr format.')
required = parser.add_argument_group('required named arguments')
required.add_argument('-i', '--infile', help='input file containing the matrix as Dop_dble() from cgne_pp1.', required=True)
required.add_argument('-o', '--outfile', help='output file will contain A = D^dagger D.', required=True)
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
    N = 12*vol
    nmx = struct.unpack('<1I', f.read(int_size))[0]
    res = struct.unpack('<1d', f.read(double_size))[0] # 1 double
    eta = struct.unpack('<{0}d'.format(2*N), f.read(double_size*2*N)) # 24*vol doubles
    psi = struct.unpack('<{0}d'.format(2*N), f.read(double_size*2*N)) # 24*vol doubles

    print('vol={0}, nmx={1}, res={2}, N={3}'.format(vol, nmx, res, N))

    # b = b_Re + i*b_Im
    b_dble_Re = np.array([itype(x) for x in eta[:2*N:2]], dtype=itype)
    b_dble_Im = np.array([itype(x) for x in eta[1:2*N:2]], dtype=itype)

    # x0 = 0
    x0_dble_Re = np.zeros(N, dtype=itype)
    x0_dble_Im = np.zeros(N, dtype=itype)

    print("time", time.time(), time.time() - lastime); lastime = time.time();
    print("retrieving Dirac matrix in csr format (Re,dble) ...")

    # length numbers, data, index and ptr of Dop (Re)
    length = struct.unpack('<1I', f.read(int_size))[0]
    data = np.array(struct.unpack(ifmt.format(length), f.read(entry_size*length)))
    length = struct.unpack('<1I', f.read(int_size))[0]
    index = struct.unpack('<{0}I'.format(length), f.read(int_size*length))
    length = struct.unpack('<1I', f.read(int_size))[0]
    ptr = struct.unpack('<{0}I'.format(length), f.read(int_size*length))

    # length numbers, tr_data, tr_index and tr_ptr of Dop (Re)
    length = struct.unpack('<1I', f.read(int_size))[0]
    tr_data = np.array([itype(x) for x in struct.unpack(ifmt.format(length), f.read(entry_size*length))], dtype=itype)
    length = struct.unpack('<1I', f.read(int_size))[0]
    tr_index = np.array(struct.unpack('<{0}I'.format(length), f.read(int_size*length)))
    length = struct.unpack('<1I', f.read(int_size))[0]
    tr_ptr = np.array(struct.unpack('<{0}I'.format(length), f.read(int_size*length)))

    # creating the itype object
    Dop_Re = csr_matrix([data, index, ptr], itype)
    Dop_Re.tr_rdata, Dop_Re.tr_col_index, Dop_Re.tr_row_ptr = tr_data, tr_index, tr_ptr
    Dop_Re.tr_calculated = True
    sc_Dop_Re = sc.sparse.csr_matrix((data, index, ptr), dtype=itype)
    sc_Dop_Re_transposed = sc.sparse.csr_matrix((tr_data, tr_index, tr_ptr), dtype=itype)

    print("time", time.time(), time.time() - lastime); lastime = time.time();
    print("retrieving Dirac matrix in csr format (Im,dble) ...")

    # length numbers, data, index and ptr of Dop (Re)
    length = struct.unpack('<1I', f.read(int_size))[0]
    data = struct.unpack(ifmt.format(length), f.read(entry_size*length))
    length = struct.unpack('<1I', f.read(int_size))[0]
    index = struct.unpack('<{0}I'.format(length), f.read(int_size*length))
    length = struct.unpack('<1I', f.read(int_size))[0]
    ptr = struct.unpack('<{0}I'.format(length), f.read(int_size*length))

    # length numbers, tr_data, tr_index and tr_ptr of Dop (Re)
    length = struct.unpack('<1I', f.read(int_size))[0]
    tr_data = np.array([itype(x) for x in struct.unpack(ifmt.format(length), f.read(entry_size*length))], dtype=itype)
    length = struct.unpack('<1I', f.read(int_size))[0]
    tr_index = np.array(struct.unpack('<{0}I'.format(length), f.read(int_size*length)))
    length = struct.unpack('<1I', f.read(int_size))[0]
    tr_ptr = np.array(struct.unpack('<{0}I'.format(length), f.read(int_size*length)))

    # creating the object
    Dop_Im = csr_matrix([data, index, ptr], itype)
    Dop_Im.tr_rdata, Dop_Im.tr_col_index, Dop_Im.tr_row_ptr = tr_data, tr_index, tr_ptr
    Dop_Im.tr_calculated = True
    sc_Dop_Im = sc.sparse.csr_matrix((data, index, ptr), dtype=itype)
    sc_Dop_Im_transposed = sc.sparse.csr_matrix((tr_data, tr_index, tr_ptr), dtype=itype)

    print("time", time.time(), time.time() - lastime); lastime = time.time();

    print("calculating D^dagger D matrix in csr format (Re,dble) ...")
    #A_Re = Dop_Re.transpose_times(Dop_Re).plus(1,  Dop_Im.transpose_times(Dop_Im))
    ReTransposeRe = sc.sparse.csr_matrix(sc_Dop_Re_transposed.dot(sc_Dop_Re), dtype=otype)
    ImTransposeIm = sc.sparse.csr_matrix(sc_Dop_Im_transposed.dot(sc_Dop_Im), dtype=otype)
    sc_A_Re = sc.sparse.csr_matrix(ReTransposeRe + ImTransposeIm, dtype=otype)
    print("time", time.time(), time.time() - lastime); lastime = time.time();

    print("calculating D^dagger D matrix in csr format (Im,dble) ...")
    ReTransposeIm = sc.sparse.csr_matrix(sc_Dop_Re_transposed.dot(sc_Dop_Im), dtype=otype)
    ImTransposeRe = sc.sparse.csr_matrix(sc_Dop_Im_transposed.dot(sc_Dop_Re), dtype=otype)
    sc_A_Im = sc.sparse.csr_matrix(ReTransposeIm - ImTransposeRe, dtype=otype)
    #A_Im = Dop_Re.transpose_times(Dop_Im).plus(-1, Dop_Im.transpose_times(Dop_Re))
    print("time", time.time(), time.time() - lastime); lastime = time.time();

    A_Re = csr_matrix([sc_A_Re.data, sc_A_Re.indices, sc_A_Re.indptr], itype)
    A_Im = csr_matrix([sc_A_Im.data, sc_A_Im.indices, sc_A_Im.indptr], itype)

    A_Re.tr_rdata = A_Re.rdata
    A_Re.tr_col_index = A_Re.col_index
    A_Re.tr_row_ptr = A_Re.row_ptr
    A_Re.tr_calculated = True
    A_Im.tr_rdata = A_Im.rdata
    A_Im.tr_col_index = A_Im.col_index
    A_Im.tr_row_ptr = A_Im.row_ptr
    A_Im.tr_calculated = True

    with open(args.outfile, 'wb') as outfile:
        print("storing parameters ...")
        outfile.write(struct.pack('<1I', vol))
        outfile.write(struct.pack('<1I', nmx))
        outfile.write(struct.pack('<1d', res))
        outfile.write(struct.pack(ofmt.format(2*N), *eta))
        outfile.write(struct.pack(ofmt.format(2*N), *psi))
        print("time", time.time(), time.time() - lastime); lastime = time.time();

        print("storing A=D^dagger D matrix in csr format ...")
        # Re_len_rdata Re_rdata[0] ... Re_rdata[Re_len_data-1]
        # Re_len_col_index Re_col_index[0] ... Re_col_index[Re_len_col_index-1]
        # Re_len_row_ptr Re_row_ptr[0] ... Re_row_ptr[Re_len_row_ptr-1]
        outfile.write(struct.pack('<1I', A_Re.rdata.size))
        outfile.write(struct.pack(ofmt.format(A_Re.rdata.size), *A_Re.rdata))
        outfile.write(struct.pack('<1I', A_Re.col_index.size))
        outfile.write(struct.pack('<{0}I'.format(A_Re.col_index.size), *A_Re.col_index))
        outfile.write(struct.pack('<1I', A_Re.row_ptr.size))
        outfile.write(struct.pack('<{0}I'.format(A_Re.row_ptr.size), *A_Re.row_ptr))

        # Re_len_tr_rdata Re_tr_rdata[0] ... Re_tr_rdata[Re_len_tr_data-1]
        # Re_len_tr_col_index Re_tr_col_index[0] ... Re_tr_col_index[Re_len_tr_col_index-1]
        # Re_len_tr_row_ptr Re_tr_row_ptr[0] ... Re_tr_row_ptr[Re_len_tr_row_ptr-1]
        outfile.write(struct.pack('<1I', A_Re.tr_rdata.size))
        outfile.write(struct.pack(ofmt.format(A_Re.tr_rdata.size), *A_Re.tr_rdata))
        outfile.write(struct.pack('<1I', A_Re.tr_col_index.size))
        outfile.write(struct.pack('<{0}I'.format(A_Re.tr_col_index.size), *A_Re.tr_col_index))
        outfile.write(struct.pack('<1I', A_Re.tr_row_ptr.size))
        outfile.write(struct.pack('<{0}I'.format(A_Re.tr_row_ptr.size), *A_Re.tr_row_ptr))

        # Im_len_rdata Im_rdata[0] ... Im_rdata[Im_len_data-1]
        # Im_len_col_index Im_col_index[0] ... Im_col_index[Im_len_col_index-1]
        # Im_len_row_ptr Im_row_ptr[0] ... Im_row_ptr[Im_len_row_ptr-1]
        outfile.write(struct.pack('<1I', A_Im.rdata.size))
        outfile.write(struct.pack(ofmt.format(A_Im.rdata.size), *A_Im.rdata))
        outfile.write(struct.pack('<1I', A_Im.col_index.size))
        outfile.write(struct.pack('<{0}I'.format(A_Im.col_index.size), *A_Im.col_index))
        outfile.write(struct.pack('<1I', A_Im.row_ptr.size))
        outfile.write(struct.pack('<{0}I'.format(A_Im.row_ptr.size), *A_Im.row_ptr))

        # Im_len_tr_rdata Im_tr_rdata[0] ... Im_tr_rdata[Im_len_tr_data-1]
        # Im_len_tr_col_index Im_tr_col_index[0] ... Im_tr_col_index[Im_len_tr_col_index-1]
        # Im_len_tr_row_ptr Im_tr_row_ptr[0] ... Im_tr_row_ptr[Im_len_tr_row_ptr-1]
        outfile.write(struct.pack('<1I', A_Im.tr_rdata.size))
        outfile.write(struct.pack(ofmt.format(A_Im.tr_rdata.size), *A_Im.tr_rdata))
        outfile.write(struct.pack('<1I', A_Im.tr_col_index.size))
        outfile.write(struct.pack('<{0}I'.format(A_Im.tr_col_index.size), *A_Im.tr_col_index))
        outfile.write(struct.pack('<1I', A_Im.tr_row_ptr.size))
        outfile.write(struct.pack('<{0}I'.format(A_Im.tr_row_ptr.size), *A_Im.tr_row_ptr))
        print("time", time.time(), time.time() - lastime); lastime = time.time();


    """
    # TEST 1
    three_times = Dop_Re.plus(2, Dop_Re)
    print("should be 3 times true: ", np.array_equal(three_times.rdata, 3*Dop_Re.rdata), np.array_equal(three_times.col_index, Dop_Re.col_index), np.array_equal(three_times.row_ptr, Dop_Re.row_ptr))

    # TEST 2
    result = Dop_Re.plus(2, Dop_Im)
    print("should be true: ", np.array_equal(result.asarray(), Dop_Re.asarray() + 2*Dop_Im.asarray()))

    # TEST 3
    test = Dop_Re.transpose_times(Dop_Re)
    test.tr_calculated = False
    print("should be true: ", np.array_equal(test.asarray(), test.transpose().asarray()))
    test.transpose()
    """
