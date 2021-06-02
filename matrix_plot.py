#!/usr/bin/env python3
# script to make a 2d plot of the dirac matrix

import argparse
import matplotlib.pyplot as plt
import matplotlib as matplotlib
import struct # struct.unpack(), struct.pack()
import numpy as np
import json # json.dump(), json.load()
import sys
import time
sys.path.append('../python/libs/')
import latex as lt

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

        if False: #self.dtype == sp.posit32 or self.dtype == sp.posit16 or self.dtype == sp.posit8:
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

parser = argparse.ArgumentParser(description='2D plot of the matrix given by the file (usually Dop_dble()).')
required = parser.add_argument_group('required named arguments')
required.add_argument('-f', '--file', help='input file (dirac matrix in CSR format outputted by cgne_pp1.py).', required=True)
required.add_argument('-p', '--pixel', help='size of one pixel in the plot. Must be a divisor of the dimension N (ex 1 pixel = 12x12 elements -> -p 12)', required=True, type=int)
parser.add_argument('-o', '--out', help='output file in pdf format with the plot, omit the flag and the plot is shown directly.', default=False)
args = parser.parse_args()

int_size, float_size, double_size = 4, 4, 8 # bytes
itype = np.float64
fmt = '<{0}d'
entry_size = double_size

with open(args.file, "rb") as f:
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
    b_Re = np.array([itype(x) for x in eta[:2*N:2]], dtype=itype)
    b_Im = np.array([itype(x) for x in eta[1:2*N:2]], dtype=itype)

    # x0 = 0
    x0_Re = np.zeros(N, dtype=itype)
    x0_Im = np.zeros(N, dtype=itype)

    print("time", time.time(), time.time() - lastime); lastime = time.time();
    print("retrieving Dirac matrix in csr format (Re,dble) ...")

    # length numbers, data, index and ptr of Dop (Re)
    length = struct.unpack('<1I', f.read(int_size))[0]
    data = np.array([itype(x) for x in struct.unpack(fmt.format(length), f.read(entry_size*length))], dtype=itype)
    length = struct.unpack('<1I', f.read(int_size))[0]
    index = struct.unpack('<{0}I'.format(length), f.read(int_size*length))
    length = struct.unpack('<1I', f.read(int_size))[0]
    ptr = struct.unpack('<{0}I'.format(length), f.read(int_size*length))

    # length numbers, tr_data, tr_index and tr_ptr of Dop (Re)
    length = struct.unpack('<1I', f.read(int_size))[0]
    tr_data = np.array([itype(x) for x in struct.unpack(fmt.format(length), f.read(entry_size*length))], dtype=itype)
    length = struct.unpack('<1I', f.read(int_size))[0]
    tr_index = np.array(struct.unpack('<{0}I'.format(length), f.read(int_size*length)))
    length = struct.unpack('<1I', f.read(int_size))[0]
    tr_ptr = np.array(struct.unpack('<{0}I'.format(length), f.read(int_size*length)))

    # creating the object
    csr_Dop_dble_Re = csr_matrix([data, index, ptr], itype)
    csr_Dop_dble_Re.tr_rdata, csr_Dop_dble_Re.tr_col_index, csr_Dop_dble_Re.tr_row_ptr = tr_data, tr_index, tr_ptr
    csr_Dop_dble_Re.tr_calculated = True

    print("time", time.time(), time.time() - lastime); lastime = time.time();
    print("retrieving Dirac matrix in csr format (Im,dble) ...")

    # length numbers, data, index and ptr of Dop (Re)
    length = struct.unpack('<1I', f.read(int_size))[0]
    data = np.array([itype(x) for x in struct.unpack(fmt.format(length), f.read(entry_size*length))], dtype=itype)
    length = struct.unpack('<1I', f.read(int_size))[0]
    index = struct.unpack('<{0}I'.format(length), f.read(int_size*length))
    length = struct.unpack('<1I', f.read(int_size))[0]
    ptr = struct.unpack('<{0}I'.format(length), f.read(int_size*length))

    # length numbers, tr_data, tr_index and tr_ptr of Dop (Re)
    length = struct.unpack('<1I', f.read(int_size))[0]
    tr_data = np.array([itype(x) for x in struct.unpack(fmt.format(length), f.read(entry_size*length))], dtype=itype)
    length = struct.unpack('<1I', f.read(int_size))[0]
    tr_index = np.array(struct.unpack('<{0}I'.format(length), f.read(int_size*length)))
    length = struct.unpack('<1I', f.read(int_size))[0]
    tr_ptr = np.array(struct.unpack('<{0}I'.format(length), f.read(int_size*length)))

    # creating the object
    csr_Dop_dble_Im = csr_matrix([data, index, ptr], itype)
    csr_Dop_dble_Im.tr_rdata, csr_Dop_dble_Im.tr_col_index, csr_Dop_dble_Im.tr_row_ptr = tr_data, tr_index, tr_ptr
    csr_Dop_dble_Im.tr_calculated = True

    print("time", time.time(), time.time() - lastime); lastime = time.time();

    b=0
    z1 = np.zeros((int(N/args.pixel), int(N/args.pixel)))
    k = int((N/args.pixel)) if int((N/args.pixel)) != 0 else 2
    print("calculating the plot (Re,dble) ...")
    for i in range(0, int(N/args.pixel)):
        rrange = range(i*args.pixel, (i+1)*args.pixel)
        for j in range(0, int(N/args.pixel)):

            if b % k == 0:
                print("{0}%, {1}/{2}".format(int(round(b*100/int((N/args.pixel)**2))), b, int((N/args.pixel)**2)), end="\r")

            crange = range(j*args.pixel, (j+1)*args.pixel)
            s=0
            for row in rrange:
                start = csr_Dop_dble_Re.row_ptr[row]
                end = csr_Dop_dble_Re.row_ptr[row+1]
                non_zero_idxs_in_block = list(set(csr_Dop_dble_Re.col_index[start:end]) & set(list(crange)))
                z1[i,j] += np.sum(csr_Dop_dble_Re.rdata[non_zero_idxs_in_block])
            z1[i,j] /= args.pixel**2
            #print("{0}x{0} block {1} ({2},{3}), sum={4}".format(args.pixel, b, i, j, z1[i,j]))
            b+=1

    print("time", time.time(), time.time() - lastime); lastime = time.time();

    b=0
    z2 = np.zeros((int(N/args.pixel), int(N/args.pixel)))
    k = int((N/args.pixel)) if int((N/args.pixel)) != 0 else 2
    print("calculating the plot (Im,dble) ...")
    for i in range(0, int(N/args.pixel)):
        rrange = range(i*args.pixel, (i+1)*args.pixel)
        for j in range(0, int(N/args.pixel)):

            if b % k == 0:
                print("{0}%, {1}/{2}".format(int(round(b*100/int((N/args.pixel)**2))), b, int((N/args.pixel)**2)), end="\r")

            crange = range(j*args.pixel, (j+1)*args.pixel)
            s=0
            for row in rrange:
                start = csr_Dop_dble_Im.row_ptr[row]
                end = csr_Dop_dble_Im.row_ptr[row+1]
                non_zero_idxs_in_block = list(set(csr_Dop_dble_Im.col_index[start:end]) & set(list(crange)))
                z2[i,j] += np.sum(csr_Dop_dble_Im.rdata[non_zero_idxs_in_block])
            z2[i,j] /= args.pixel**2
            #print("{0}x{0} block {1} ({2},{3}), sum={4}".format(args.pixel, b, i, j, z2[i,j]))
            b+=1

    print("time", time.time(), time.time() - lastime); lastime = time.time();

    x = np.arange(0, int(N/12), int(args.pixel/12))
    y = np.arange(0, int(N/12), int(args.pixel/12))

    # only black and white
    z1[z1!=0] = 1
    z2[z2!=0] = 1

    z_max = np.abs(z1, z2).max()
    #z_min = -z_max # with color
    z_min = 0 # only black and white

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    #re = ax1.pcolormesh(x, y, z1, cmap='RdBu', vmin=z_min, vmax=z_max) # with color
    re = ax1.pcolormesh(x, y, z1, cmap='Greys', vmin=z_min, vmax=z_max) # only black and white
    ax1.invert_yaxis()
    ax1.xaxis.tick_top()
    ax1.set_title(r'$Re(D)$, density: {0:.1f}%'.format(100*csr_Dop_dble_Re.rdata.size/N**2), y=-0.075)
    
    #im = ax2.pcolormesh(x, y, z2, cmap='RdBu', vmin=z_min, vmax=z_max) # with color
    im = ax2.pcolormesh(x, y, z2, cmap='Greys', vmin=z_min, vmax=z_max) # only black and white
    ax2.invert_yaxis()
    ax2.xaxis.tick_top()    
    ax2.set_title(r'$Im(D)$, density: {0:.1f}%'.format(100*csr_Dop_dble_Re.rdata.size/N**2), y=-0.075)

    add_colorbar = False
    if add_colorbar:
        fig.subplots_adjust(right=0.9)
        cbar_ax = fig.add_axes([0.925, 0.15, 0.015, 0.7])
        fig.colorbar(re, ax=ax1, cax=cbar_ax)

    #fig.tight_layout()
    if (args.out == False):
        plt.show()
    else:
        plt.savefig(args.out, dpi=fig.dpi)
