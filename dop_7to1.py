#!/usr/bin/env python3
#
# Convert 7 files to one

import argparse
import struct # struct.unpack(), struct.pack()
import numpy as np # np.asarray, np.zeros(), np.float16, np.float32, np.float64, np.linalg.norm()
import sys
import os
import time
import scipy as sc
import scipy.sparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Conver to 7 to 1 file.')
    required = parser.add_argument_group('required named arguments')
    required.add_argument('-e', '--env', help='env file.', required=True)
    required.add_argument('-dr', '--data_real', help='real data file.', required=True)
    required.add_argument('-di', '--data_imag', help='imaginary data file.', required=True)
    required.add_argument('-ir', '--index_real', help='real index file.', required=True)
    required.add_argument('-ii', '--index_imag', help='imaginary index file.', required=True)
    required.add_argument('-pr', '--ptr_real', help='real ptr file.', required=True)
    required.add_argument('-pi', '--ptr_imag', help='imaginary ptr file.', required=True)
    required.add_argument('-o', '--outfile', help='output file.', required=True)
    args = parser.parse_args()

    int_size, float_size, double_size = 4, 4, 8 # bytes
    byteorder='little'

    with open(args.env, "rb") as f:
        lastime = time.time()
        print("retrieving parameters ...")
        vol = struct.unpack('<1I', f.read(int_size))[0] # 1 unsigned int I, < little endian
        N = 12*vol
        nmx = struct.unpack('<1I', f.read(int_size))[0]
        res = struct.unpack('<1d', f.read(double_size))[0] # 1 double
        print('vol={0}x{0}x{0}x{0}, res={1}, N={2}'.format(int(np.sqrt(np.sqrt(vol))), res, N))
        eta = struct.unpack('<{0}d'.format(2*N), f.read(double_size*2*N)) # 24*vol doubles
        psi = struct.unpack('<{0}d'.format(2*N), f.read(double_size*2*N)) # 24*vol doubles

    with open(args.data_real, "rb") as f:
        print("retrieving data (Re) ...")
        length = int(os.path.getsize(args.data_real)/double_size) # #bytes/8
        data_Re = np.array(struct.unpack('<{0}d'.format(length), f.read(double_size*length)))
    
    with open(args.data_imag, "rb") as f:
        print("retrieving data (Im) ...")
        length = int(os.path.getsize(args.data_imag)/double_size) # #bytes/8
        data_Im = np.array(struct.unpack('<{0}d'.format(length), f.read(double_size*length)))

    with open(args.index_real, "rb") as f:
        print("retrieving index (Re) ...")
        length = int(os.path.getsize(args.index_real)/int_size) # #bytes/4
        index_Re = np.array(struct.unpack('<{0}I'.format(length), f.read(int_size*length)))

    with open(args.index_imag, "rb") as f:
        print("retrieving index (Im) ...")
        length = int(os.path.getsize(args.index_imag)/int_size) # #bytes/4
        index_Im = np.array(struct.unpack('<{0}I'.format(length), f.read(int_size*length)))

    with open(args.ptr_real, "rb") as f:
        print("retrieving ptr (Re) ...")
        length = int(os.path.getsize(args.ptr_real)/int_size) # #bytes/4
        ptr_Re = np.array(struct.unpack('<{0}I'.format(length), f.read(int_size*length)))

    with open(args.ptr_imag, "rb") as f:
        print("retrieving ptr (Im) ...")
        length = int(os.path.getsize(args.ptr_imag)/int_size) # #bytes/4
        ptr_Im = np.array(struct.unpack('<{0}I'.format(length), f.read(int_size*length)))

    print("time", time.time(), time.time() - lastime); lastime = time.time();
    re = sc.sparse.csr_matrix((data_Re, index_Re, ptr_Re), shape=(N, N))
    im = sc.sparse.csr_matrix((data_Im, index_Im, ptr_Im), shape=(N, N))
    print("time", time.time(), time.time() - lastime); lastime = time.time();
    ret = sc.sparse.csr_matrix(re.transpose(copy = True))
    imt = sc.sparse.csr_matrix(im.transpose(copy = True))

    print(re.shape)
    print(im.shape)
    print(ret.shape)
    print(imt.shape)
    exit()
    print("time", time.time(), time.time() - lastime); lastime = time.time();

    with open(args.outfile, 'wb') as outfile:
        print("storing parameters ...")
        ofmt = '<{0}d'
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
        outfile.write(struct.pack('<1I', re.data.size))
        outfile.write(struct.pack(ofmt.format(re.data.size), *re.data))
        outfile.write(struct.pack('<1I', re.indices.size))
        outfile.write(struct.pack('<{0}I'.format(re.indices.size), *re.indices))
        outfile.write(struct.pack('<1I', re.indptr.size))
        outfile.write(struct.pack('<{0}I'.format(re.indptr.size), *re.indptr))

        # Re_len_tr_rdata Re_tr_rdata[0] ... Re_tr_rdata[Re_len_tr_data-1]
        # Re_len_tr_col_index Re_tr_col_index[0] ... Re_tr_col_index[Re_len_tr_col_index-1]
        # Re_len_tr_row_ptr Re_tr_row_ptr[0] ... Re_tr_row_ptr[Re_len_tr_row_ptr-1]
        outfile.write(struct.pack('<1I', ret.data.size))
        outfile.write(struct.pack(ofmt.format(ret.data.size), *ret.data))
        outfile.write(struct.pack('<1I', ret.indices.size))
        outfile.write(struct.pack('<{0}I'.format(ret.indices.size), *ret.indices))
        outfile.write(struct.pack('<1I', ret.indptr.size))
        outfile.write(struct.pack('<{0}I'.format(ret.indptr.size), *ret.indptr))

        # Im_len_rdata Im_rdata[0] ... Im_rdata[Im_len_data-1]
        # Im_len_col_index Im_col_index[0] ... Im_col_index[Im_len_col_index-1]
        # Im_len_row_ptr Im_row_ptr[0] ... Im_row_ptr[Im_len_row_ptr-1]
        outfile.write(struct.pack('<1I', im.data.size))
        outfile.write(struct.pack(ofmt.format(im.data.size), *im.data))
        outfile.write(struct.pack('<1I', im.indices.size))
        outfile.write(struct.pack('<{0}I'.format(im.indices.size), *im.indices))
        outfile.write(struct.pack('<1I', im.indptr.size))
        outfile.write(struct.pack('<{0}I'.format(im.indptr.size), *im.indptr))

        # Im_len_tr_rdata Im_tr_rdata[0] ... Im_tr_rdata[Im_len_tr_data-1]
        # Im_len_tr_col_index Im_tr_col_index[0] ... Im_tr_col_index[Im_len_tr_col_index-1]
        # Im_len_tr_row_ptr Im_tr_row_ptr[0] ... Im_tr_row_ptr[Im_len_tr_row_ptr-1]
        outfile.write(struct.pack('<1I', imt.data.size))
        outfile.write(struct.pack(ofmt.format(imt.data.size), *imt.data))
        outfile.write(struct.pack('<1I', imt.indices.size))
        outfile.write(struct.pack('<{0}I'.format(imt.indices.size), *imt.indices))
        outfile.write(struct.pack('<1I', imt.indptr.size))
        outfile.write(struct.pack('<{0}I'.format(imt.indptr.size), *imt.indptr))
        print("time", time.time(), time.time() - lastime); lastime = time.time();

