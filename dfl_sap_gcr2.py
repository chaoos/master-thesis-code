#!/usr/bin/env python3
#
# Implementation of SAP GCR algortihm

import argparse
import struct # struct.unpack(), struct.pack()
import numpy as np # np.asarray, np.zeros(), np.float16, np.float32, np.float64, np.linalg.norm()
import softposit as sp # sp.posit8, sp.posit16, sp.posit32,
import os # os.remove(), os.system()
import sys
import time
from random import randrange
import scipy as sc
import json # json.dump(), json.load()
import floats as fl
from cgne_pp2 import get_epsilon, get_precision_limit
from agnostic import complex_matrix_vector_prod, complex_matrix_dagger_vector_prod, complex_scalar_prod, complex_norm_sq, complex_axpy, axpy, compile_gpu_kernels
import itertools # repeat() for endless for loops
import cupyx as cpx # cpx.scipy.sparse.csr_matrix()
import cupy as cp # cp.array()
import multiprocessing
from scipy.io import mmread

# returns the FLT_MIN variable in case of float32, else the corresponding value
def get_min(dtype):
    if dtype == np.float64:
        return 1e-308 # DBL_MIN
    elif dtype == np.float32:
        return 1e-37 # FLT_MIN
    elif dtype == np.float16:
        return 1e-5

def get_eta(file, vol):
    N = 12*vol
    double_size = 8
    with open(file, "rb") as f:
        return struct.unpack('<{0}d'.format(2*N), f.read(double_size*2*N)) # 24*vol doubles

def fgcr(dim, blk_dim, vol, nmx, res, nkv, ncy, nmr, Dop, Mop, Dop_ddtype, eta_ddtype_Re, eta_ddtype_Im, psi_ddtype_Re, psi_ddtype_Im, ddtype, sdtype, cdtype, xp, xp2, args, prnt):

    start_time = time.time()

    spinor_size = eta_ddtype_Re.size

    # start gcr_init()
    # phi = 0 (nkv of these)
    phi_sdtype_Re = xp.zeros((nkv, spinor_size), dtype=sdtype)
    phi_sdtype_Im = xp.zeros((nkv, spinor_size), dtype=sdtype)

    # rho = eta (only one of these)
    rho_sdtype_Re = xp.copy(xp.array(eta_ddtype_Re, dtype=sdtype))
    rho_sdtype_Im = xp.copy(xp.array(eta_ddtype_Im, dtype=sdtype))

    # chi (nkv of these)
    chi_sdtype_Re = xp.zeros((nkv, spinor_size), dtype=sdtype)
    chi_sdtype_Im = xp.zeros((nkv, spinor_size), dtype=sdtype)

    wrk_ddtype_Re = xp.zeros_like(rho_sdtype_Re, dtype=ddtype) # ddtype
    wrk_ddtype_Im = xp.zeros_like(rho_sdtype_Re, dtype=ddtype) # ddtype

    rn = complex_norm_sq(rho_sdtype_Re, rho_sdtype_Im, cdtype) # ddtype
    rn = np.sqrt(rn) # ddtype
    # end gcr_init()

    tol = ddtype(res)*rn
    status = 0

    a_Re = xp.zeros(nkv*(nkv+1), dtype=cdtype) # was sdtype, but should be cdtype
    a_Im = xp.zeros(nkv*(nkv+1), dtype=cdtype) # was sdtype, but should be cdtype
    b = xp.zeros(nkv, dtype=cdtype)            # was sdtype, but should be cdtype
    c_Re = xp.zeros(nkv, dtype=cdtype)         # was sdtype, but should be cdtype
    c_Im = xp.zeros(nkv, dtype=cdtype)         # was sdtype, but should be cdtype

    ls = int(np.ceil((eta_ddtype_Re.size/12)**(1/4))) # lattice size -> 4, 8, 16, ...

    json_data = {}
    json_data['gpu'] = args.device == 'gpu'
    json_data['device'] = args.device
    json_data['infile'] = args.infile
    json_data['matfile'] = args.matfile
    json_data['k'] = args.k
    json_data['kc'] = args.kc
    json_data['reordering'] = not args.no_reordering
    json_data['preconditioning'] = not args.no_preconditioning
    json_data['vol'] = vol
    json_data['dim'] = dim
    json_data['lattice'] = '{0}x{0}x{0}x{0}'.format(ls)
    json_data['blk_dim'] = blk_dim
    json_data['bs'] = '{0}x{1}x{2}x{3}'.format(args.b0, args.b1, args.b2, args.b3)
    json_data['res'] = np.float64(res)
    json_data['tol'] = np.float64(tol)
    json_data['nmx'] = nmx
    json_data['nkv'] = nkv
    json_data['nmr'] = 0.5 if args.extension == 1 else nmr # 0.5 = adaptive
    json_data['ncy'] = 0.5 if args.extension == 1 else ncy # 0.5 = adaptive
    json_data['sdtype'] = type(sdtype(0.0)).__name__
    json_data['ddtype'] = type(ddtype(0.0)).__name__
    json_data['cdtype'] = type(cdtype(0.0)).__name__
    json_data['extension'] = args.extension
    json_data['ext_alpha'] = args.ext_alpha
    json_data['ext_beta'] = args.ext_beta
    json_data['deflation'] = args.little != None
    json_data['dfl_little'] = args.little
    json_data['dfl_nmx'] = args.dfl_nmx
    json_data['dfl_res'] = args.dfl_res
    json_data['rn'] = []
    json_data['i'] = []
    json_data['reset'] = []
    
    i = 0

    while rn > tol:
        rn_old = rn

        for k in itertools.count(0): # k=0, 1, ..., infinite
            # start gcr_step()
            if prnt and args.verbose > 0: print('gcr_step() i={0}, status={1}, rn({2}), tol({3}), rn={4} > tol={5}'.format(i, status, type(rn), type(tol), rn, tol))

            json_data['i'].append(i)

            phi_sdtype_Re[k], phi_sdtype_Im[k], chi_sdtype_Re[k], chi_sdtype_Im[k] = Mop(i, rho_sdtype_Re, rho_sdtype_Im, sdtype, cdtype, tol)

            if args.verbose > 1:
                Dphi_Re, Dphi_Im = Dop_ddtype(phi_sdtype_Re[k], phi_sdtype_Im[k])
                axpy_Re, axpy_Im = axpy(-1.0, Dphi_Re, Dphi_Im, rho_sdtype_Re, rho_sdtype_Im, ddtype)
                rn_Mop = np.sqrt(complex_norm_sq(phi_sdtype_Re[k], phi_sdtype_Im[k], ddtype))
                print('Mop() |phik|={0}'.format(rn_Mop))
                print('Mop() |rho|={0} > |Dphik-rho|={1}'.format(np.sqrt(complex_norm_sq(rho_sdtype_Re, rho_sdtype_Im)), np.sqrt(complex_norm_sq(axpy_Re, axpy_Im))))

            # GS orthogonalisation
            for l in range(0, k):
                wrk1, wrk2 = complex_scalar_prod(chi_sdtype_Re[l], chi_sdtype_Im[l], chi_sdtype_Re[k], chi_sdtype_Im[k], cdtype)
                a_Re[nkv*l+k], a_Im[nkv*l+k] = wrk1, wrk2
                if l == nkv: c_Re[k], c_Im[k] = a_Re[nkv**2 + k], a_Im[nkv**2 + k] # set the corresponding c values
                z_Re = -a_Re[nkv*l+k]
                z_Im = -a_Im[nkv*l+k]
                chi_sdtype_Re[k], chi_sdtype_Im[k] = complex_axpy(z_Re, z_Im, chi_sdtype_Re[l], chi_sdtype_Im[l], chi_sdtype_Re[k], chi_sdtype_Im[k], sdtype)

            b[k] = np.sqrt(complex_norm_sq(chi_sdtype_Re[k], chi_sdtype_Im[k], cdtype))
            if args.verbose > 1: print('b[{0}] = {1} ({2})'.format(k, b[k], type(b[k])))
            chi_sdtype_Re[k], chi_sdtype_Im[k] = sdtype(1.0)/b[k]*chi_sdtype_Re[k], sdtype(1.0)/b[k]*chi_sdtype_Im[k]

            wrk1, wrk2 = complex_scalar_prod(chi_sdtype_Re[k], chi_sdtype_Im[k], rho_sdtype_Re, rho_sdtype_Im, cdtype)
            c_Re[k], c_Im[k] = wrk1, wrk2
            a_Re[nkv**2 + k], a_Im[nkv**2 + k] = c_Re[k], c_Im[k] # set the corresponding a values
            z_Re = -c_Re[k]
            z_Im = -c_Im[k]
            rho_sdtype_Re, rho_sdtype_Im = complex_axpy(z_Re, z_Im, chi_sdtype_Re[k], chi_sdtype_Im[k], rho_sdtype_Re, rho_sdtype_Im, sdtype)

            rn = complex_norm_sq(rho_sdtype_Re, rho_sdtype_Im, cdtype)
            rn = np.sqrt(rn)
            json_data['rn'].append(np.float64(rn))
            # end gcr_step()

            status += 1
            i += 1

            if args.verbose > 1: print(type(phi_sdtype_Re[k][0]), type(phi_sdtype_Re[k][0]), type(chi_sdtype_Re[k][0]), type(chi_sdtype_Re[k][0]))

            if rn <= tol or rn < get_precision_limit(ddtype)*rn_old or k+1 == nkv or status == nmx:
                if prnt and args.verbose > 0 and rn <= tol:
                    print('gcr_step() rn <= tol rn({0}), tol({1}), rn={2}, tol={3}'.format(type(rn), type(tol), rn, tol))
                if prnt and args.verbose > 0 and rn < (get_precision_limit(ddtype)*rn_old):
                    print('gcr_step() rn < PRECISION_LIMIT*rn_old rn({0}), rn_old({1}), rn={2} <= rn_old={3}'.format(type(rn), type(rn_old), rn, rn_old))
                if prnt and args.verbose > 0 and k+1 == nkv:
                    print('gcr_step() k+1 == nkv')
                if prnt and args.verbose > 0 and status == nmx:
                    print('gcr_step() status == nmx rn({0}), rn_old({1}), rn={2}, rn_old={3}'.format(type(rn), type(rn_old), rn, rn_old))
                break

        # start update_psi()
        for l in range(k, -1, -1): # l=k, k-1, k-2, ..., 1, 0
            z_Re = c_Re[l] # sdtype
            z_Im = c_Im[l]

            for j in range(l+1,k+1):
                z_Re -= (a_Re[l*nkv+j]*c_Re[j] - a_Im[l*nkv+j]*c_Im[j])
                z_Im -= (a_Re[l*nkv+j]*c_Im[j] + a_Im[l*nkv+j]*c_Re[j])

            r = sdtype(1.0)/b[l]
            c_Re[l] = z_Re*r
            c_Im[l] = z_Im*r
            a_Re[nkv**2 + l], a_Im[nkv**2 + l] = c_Re[l], c_Im[l] # set the corresponding a values

        # rho = 0
        rho_sdtype_Re = xp.zeros_like(rho_sdtype_Re, dtype=sdtype)
        rho_sdtype_Im = xp.zeros_like(rho_sdtype_Im, dtype=sdtype)

        for l in range(k, -1, -1): # l=k, k-1, k-2, ..., 1, 0
            rho_sdtype_Re, rho_sdtype_Im = complex_axpy(c_Re[l], c_Im[l], phi_sdtype_Re[l], phi_sdtype_Im[l], rho_sdtype_Re, rho_sdtype_Im, sdtype)

        rho_ddtype_Re, rho_ddtype_Im = xp.array(rho_sdtype_Re, copy=True, dtype=ddtype), xp.array(rho_sdtype_Im, copy=True, dtype=ddtype)
        psi_ddtype_Re, psi_ddtype_Im = axpy(1.0, psi_ddtype_Re, psi_ddtype_Im, rho_ddtype_Re, rho_ddtype_Im, ddtype)
        wrk_ddtype_Re, wrk_ddtype_Im = Dop_ddtype(psi_ddtype_Re, psi_ddtype_Im)
        rho_ddtype_Re, rho_ddtype_Im = axpy(-1.0, wrk_ddtype_Re, wrk_ddtype_Im, eta_ddtype_Re, eta_ddtype_Im, ddtype)
        rho_sdtype_Re, rho_sdtype_Im = xp.array(rho_ddtype_Re, dtype=sdtype), xp.array(rho_ddtype_Im, dtype=sdtype)

        rn = complex_norm_sq(rho_sdtype_Re, rho_sdtype_Im, ddtype)
        rn = np.sqrt(rn)
        if prnt and args.verbose > 0: print('update_psi() i={0}, status={1}, rn({2}), tol({3}), rn={4} > tol={5}'.format(i, status, type(rn), type(tol), rn, tol))
        json_data['rn'].append(np.float64(rn))
        json_data['i'].append(i)
        json_data['reset'].append(i)
        i += 1
        # end update_psi()

        if status == nmx and rn > tol:
            status = -1 # not converged in the given number of iterations
            end_time = time.time()
            json_data['duration'] = end_time - start_time
            json_data['status'] = status
            return psi_ddtype_Re, psi_ddtype_Im, rn, json_data

        if xp.isnan(rn) or xp.isinf(rn):
            status = -2 # diverged or division by zero
            end_time = time.time()
            json_data['duration'] = end_time - start_time
            json_data['status'] = status
            return psi_ddtype_Re, psi_ddtype_Im, rn, json_data

    end_time = time.time()
    json_data['duration'] = end_time - start_time
    json_data['status'] = status
    return psi_ddtype_Re, psi_ddtype_Im, rn, json_data

# psi = phi, rho = chi
# note that psi, rho, Dop_blk must be in the same datatype
def sap(i, vol, nmr, psi_Re, psi_Im, rho_Re, rho_Im, bs, Dop_blk, cdtype, xp, xp2, tol):
    blk_vol = bs[0]*bs[1]*bs[2]*bs[3] # ex: 4x4x4x4
    nb = int(vol/blk_vol)
    nbh = int(nb/2)
    wrk_Re = xp.zeros_like(psi_Re)
    wrk_Im = xp.zeros_like(psi_Im)
    aimr = xp.empty(nb)
    i = 0

    for col, opposite_col in zip([0,1], [1,0]): # 0=black, 1=white
        blk_start = col*nbh
        blk_end = (col+1)*nbh

        # loop over all blocks of the same color
        for n in range(blk_start, blk_end):
            n = block_reorder[n]
            br, bs, be = range_block(blk_dim, blk_vol, n)  # range of nth block
            blk_rho_Re = xp2.array(rho_Re[bs:be], copy=True, dtype=sdtype)
            blk_rho_Im = xp2.array(rho_Im[bs:be], copy=True, dtype=sdtype)

            # solve every block individually
            blk_psi_Re, blk_psi_Im, blk_rho_Re, blk_rho_Im, imr = blk_mres(n, nmr, blk_rho_Re, blk_rho_Im, cdtype, sdtype, xp2, tol/nb)
            aimr[i] = imr
            i += 1
            if args.verbose > 1: print('blk_mres() rn={0}, n={1}'.format(np.sqrt(complex_norm_sq(blk_rho_Re, blk_rho_Im)), n))

            if args.device == 'hybrid':
                blk_psi_Re = blk_psi_Re.get()
                blk_psi_Im = blk_psi_Im.get()
                blk_rho_Re = blk_rho_Re.get()
                blk_rho_Im = blk_rho_Im.get()

            # store the block result in wrk for later
            wrk_Re[bs:be] = xp.array(blk_psi_Re, copy=True)
            wrk_Im[bs:be] = xp.array(blk_psi_Im, copy=True)

            # update the global solution vector (on the current block)
            psi_Re[bs:be] += blk_psi_Re
            psi_Im[bs:be] += blk_psi_Im

            # update the global residual vector (on the current block)
            rho_Re[bs:be] = xp.array(blk_rho_Re, copy=True) # NOT += here (as it is written in the paper, page 10 (b), they also have = in their code)
            rho_Im[bs:be] = xp.array(blk_rho_Im, copy=True) # NOT += here (as it is written in the paper, page 10 (b), they also have = in their code)

            """
            # more expensive (because done for every block), but works all the times, even if the black (white) blocks in the 
            # matrix are not independent.
            # update the global residual vector (on the external boundary of the current block using the internal boundary operator)
            bnd_eb_Dpsi_Re, bnd_eb_Dpsi_Im = Dop_int_bnd_sdtype(n, blk_psi_Re, blk_psi_Im)

            # substract it on the external boundary (bnd_eb_Dpsi is zero elsewhere)
            rho_Re -= bnd_eb_Dpsi_Re
            rho_Im -= bnd_eb_Dpsi_Im
            """

        # less expensive (because done only for color), but only works if the blocks where reordered accordingly
        # substract it on the external boundary (bnd_eb_Dpsi is zero elsewhere)
        bnd_eb_Dpsi_Re, bnd_eb_Dpsi_Im = Dop_int_color_bnd_sdtype(col, wrk_Re, wrk_Im)
        rho_Re -= bnd_eb_Dpsi_Re
        rho_Im -= bnd_eb_Dpsi_Im

    return psi_Re, psi_Im, rho_Re, rho_Im, xp.average(aimr)

# gives the range() in the global arrays of the block n
def range_block(blk_dim, blk_vol, n):
    start = blk_dim*n
    end   = blk_dim*(n+1)
    return range(start, end, 1), start, end

# gives the range() in the global arrays of the black or white boundary
def range_color(dim, blk_vol, nb, col):
    start = int(dim/2)*col
    end   = int(dim/2)*(col+1)
    return range(start, end, 1), start, end

# mnr steps of mres
# @param n block number
# @param mnr number of mr-steps to take
# @param eta source vector
# @param cdtype datatype for collective variables
# @param xp datatype of the arrays (np or cp)
# returns approx solution psi and residual rho
def blk_mres(n, nmr, eta_Re, eta_Im, cdtype, sdtype, xp, tol):
    shape = eta_Re.shape
    psi_Re = xp.zeros(shape, dtype=sdtype) # psi=0, same shape and datatype
    psi_Im = xp.zeros(shape, dtype=sdtype)
    rho_Re, rho_Im = xp.copy(eta_Re), xp.copy(eta_Im)

    if args.extension == 1: rn = np.sqrt(complex_norm_sq(rho_Re, rho_Im, cdtype))

    for imr in range(0, nmr):
        wrk_Re, wrk_Im = Dop_blk_sdtype(n, rho_Re, rho_Im)
        r = complex_norm_sq(wrk_Re, wrk_Im, cdtype)

        if args.verbose > 1: print("blk_mres() imr={0}, r={1}, {2}".format(imr, r, type(psi_Re)))

        if r < 2.0*get_min(sdtype): # overflow danger
            if args.verbose > 1: print("blk_mres() imr={0}, break".format(imr))
            break

        z_Re, z_Im = complex_scalar_prod(wrk_Re, wrk_Im, rho_Re, rho_Im, cdtype)

        r = 1.0/r
        z_Re *= r
        z_Im *= r
        psi_Re, psi_Im = complex_axpy(z_Re, z_Im, rho_Re, rho_Im, psi_Re, psi_Im, sdtype)

        z_Re = -z_Re
        z_Im = -z_Im
        rho_Re, rho_Im = complex_axpy(z_Re, z_Im, wrk_Re, wrk_Im, rho_Re, rho_Im, sdtype)

        if args.extension == 1:
            rn_old = rn
            rn = np.sqrt(complex_norm_sq(rho_Re, rho_Im, cdtype))
            a = rn/rn_old
            if args.verbose > 1: print('mr_step() imr={0}, rn={1}, alpha={2}'.format(imr, rn, a))
            if (a >= args.ext_alpha and imr+1 >= 4) or a >= 1 or rn <= tol:
                if args.verbose > 1:
                    if (a >= args.ext_alpha and imr+1 >= 4): print('mr_step(): broke because of alpha >= {0} and imr+1 >= 3'.format(args.ext_alpha))
                    elif a >= 1: print('mr_step(): broke because of alpha >= 1')
                    elif rn <= tol: print('mr_step(): broke because of rn <= tol')
                #print('mr_step() took only nmr={0}/{1}'.format(imr+1, nmr))
                break

        #if args.verbose > 1:
        #    rn = np.sqrt(complex_norm_sq(rho_Re, rho_Im))
        #    print('mr_step() imr={0}, rn({1}), rn={2}'.format(imr, type(rn), rn))

    #if imr != nmr-1: print('blk_mres() imr={0}'.format(imr))

    return psi_Re, psi_Im, rho_Re, rho_Im, imr+1

def load_matrix_from_file(args, dtype = np.float64, sdtype = np.float64):

    int_size, float_size, double_size = 4, 4, 8 # bytes
    byteorder='little'
    entry_size = double_size
    fmt = '<{0}d'
    bs = np.array([args.b0, args.b1, args.b2, args.b3], dtype=int)

    if args.device == 'gpu':
        identity = cpx.scipy.sparse.identity
        csr_matrix = cpx.scipy.sparse.csr_matrix
        csr_matrix2 = cpx.scipy.sparse.csr_matrix
        bmat = cpx.scipy.sparse.bmat
        xp = cp
        xp2 = cp
    elif args.device == 'hybrid': # blocked ops are on the gpu, full op and bnd op are on cpu
        identity = sc.sparse.identity
        csr_matrix = sc.sparse.csr_matrix
        csr_matrix2 = cpx.scipy.sparse.csr_matrix
        bmat = sc.sparse.bmat
        xp = np
        xp2 = cp
    else:
        identity = sc.sparse.identity
        csr_matrix = sc.sparse.csr_matrix
        csr_matrix2 = sc.sparse.csr_matrix
        bmat = sc.sparse.bmat
        xp = np
        xp2 = np

    if args.seven == False:

        with open(args.infile, "rb") as f:
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

            print('vol={0}x{0}x{0}x{0}, nmx={1}, nkv={2}, ncy={3}, nmr={4}, res={5}, N={6}, bs={7}x{8}x{9}x{10}'.format(int(np.sqrt(np.sqrt(vol))), nmx, args.nkv, args.ncy, args.nmr, res, N, args.b0, args.b1, args.b2, args.b3))

            if args.eta != None:
                eta = get_eta(args.eta, vol)
            
            if args.random_b:
                eta = np.random.randn(24*vol)

            # b = b_Re + i*b_Im
            b_ddtype_Re = xp.array([dtype(x) for x in eta[:2*N:2]], dtype=dtype)
            b_ddtype_Im = xp.array([dtype(x) for x in eta[1:2*N:2]], dtype=dtype)

            # x0 = 0
            x0_ddtype_Re = xp.zeros(N, dtype=dtype)
            x0_ddtype_Im = xp.zeros(N, dtype=dtype)

            pos = f.tell()

        if args.matfile != None:

            print("time", time.time(), time.time() - lastime); lastime = time.time();
            print("retrieving Dirac matrix in mm format to csr format (dble) ...")
            y = mmread(args.matfile).tocsr()

            # prepare A = id - k*D, the matfile contains D
            id = identity(N, dtype=dtype)
            data, index, ptr = xp.array(np.real(y.data)), xp.array(y.indices), xp.array(y.indptr)
            re = csr_matrix((data, index, ptr), dtype=dtype)
            Dop_ddtype_Re = id - args.k*re
            Dop_ddtype_Re = csr_matrix(Dop_ddtype_Re, dtype=dtype)
            data, index, ptr = xp.array(np.imag(y.data)), xp.array(y.indices), xp.array(y.indptr)
            im = csr_matrix((data, index, ptr), dtype=dtype)
            Dop_ddtype_Im = -args.k*im
            Dop_ddtype_Im = csr_matrix(Dop_ddtype_Im, dtype=dtype)
            del y, data, index, ptr, re, im, id

        else:
            with open(args.infile, "rb") as f:
                f.seek(pos)

                print("time", time.time(), time.time() - lastime); lastime = time.time();
                print("retrieving Dirac matrix in csr format (Re,dble) ...")

                # length numbers, data, index and ptr of Dop (Re)
                length = struct.unpack('<1I', f.read(int_size))[0]
                data = xp.array(struct.unpack(fmt.format(length), f.read(entry_size*length)))
                length = struct.unpack('<1I', f.read(int_size))[0]
                index = xp.array(struct.unpack('<{0}I'.format(length), f.read(int_size*length)))
                length = struct.unpack('<1I', f.read(int_size))[0]
                ptr = xp.array(struct.unpack('<{0}I'.format(length), f.read(int_size*length)))

                # length numbers, tr_data, tr_index and tr_ptr of Dop (Re)
                length = struct.unpack('<1I', f.read(int_size))[0]
                tr_data = np.array([dtype(x) for x in struct.unpack(fmt.format(length), f.read(entry_size*length))], dtype=dtype)
                length = struct.unpack('<1I', f.read(int_size))[0]
                tr_index = np.array(struct.unpack('<{0}I'.format(length), f.read(int_size*length)))
                length = struct.unpack('<1I', f.read(int_size))[0]
                tr_ptr = np.array(struct.unpack('<{0}I'.format(length), f.read(int_size*length)))

                p = xp.ones(N+1, dtype=dtype)
                p[0:ptr.size] = ptr
                p[ptr.size:] *= ptr[-1]
                Dop_ddtype_Re = csr_matrix((data, index, p), shape=(N, N), dtype=dtype)

                print("time", time.time(), time.time() - lastime); lastime = time.time();
                print("retrieving Dirac matrix in csr format (Im,dble) ...")

                # length numbers, data, index and ptr of Dop (Re)
                length = struct.unpack('<1I', f.read(int_size))[0]
                data = xp.array(struct.unpack(fmt.format(length), f.read(entry_size*length)))
                length = struct.unpack('<1I', f.read(int_size))[0]
                index = xp.array(struct.unpack('<{0}I'.format(length), f.read(int_size*length)))
                length = struct.unpack('<1I', f.read(int_size))[0]
                ptr = xp.array(struct.unpack('<{0}I'.format(length), f.read(int_size*length)))

                # length numbers, tr_data, tr_index and tr_ptr of Dop (Re)
                length = struct.unpack('<1I', f.read(int_size))[0]
                tr_data = np.array([dtype(x) for x in struct.unpack(fmt.format(length), f.read(entry_size*length))], dtype=dtype)
                length = struct.unpack('<1I', f.read(int_size))[0]
                tr_index = np.array(struct.unpack('<{0}I'.format(length), f.read(int_size*length)))
                length = struct.unpack('<1I', f.read(int_size))[0]
                tr_ptr = np.array(struct.unpack('<{0}I'.format(length), f.read(int_size*length)))

                p = xp.ones(N+1, dtype=dtype)
                p[0:ptr.size] = ptr
                p[ptr.size:] *= ptr[-1]
                Dop_ddtype_Im = csr_matrix((data, index, p), shape=(N, N), dtype=dtype)

    else: # args.seven is active!
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

            if args.nmx != None:
                nmx = args.nmx
            if args.vol != None:
                vol = args.vol
                N = 12*vol

            if args.res != None:
                res = args.res

            print('vol={0}x{0}x{0}x{0}, nmx={1}, nkv={2}, ncy={3}, nmr={4}, res={5}, N={6}, bs={7}x{8}x{9}x{10}'.format(int(np.sqrt(np.sqrt(vol))), nmx, args.nkv, args.ncy, args.nmr, res, N, args.b0, args.b1, args.b2, args.b3))

            if args.eta != None:
                eta = get_eta(args.eta, vol)
            
            if args.random_b:
                eta = np.random.rand(2*N)

            # b = b_Re + i*b_Im
            b_ddtype_Re = xp.array([dtype(x) for x in eta[:2*N:2]], dtype=dtype)
            b_ddtype_Im = xp.array([dtype(x) for x in eta[1:2*N:2]], dtype=dtype)

            # x0 = 0
            x0_ddtype_Re = xp.zeros(N, dtype=dtype)
            x0_ddtype_Im = xp.zeros(N, dtype=dtype)

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

        Dop_ddtype_Re = sc.sparse.csr_matrix((data_Re, index_Re, ptr_Re), shape=(N, N))
        Dop_ddtype_Im = sc.sparse.csr_matrix((data_Im, index_Im, ptr_Im), shape=(N, N))
        print("time", time.time(), time.time() - lastime); lastime = time.time();

    dim = Dop_ddtype_Re.shape[0]
    blk_vol = bs[0]*bs[1]*bs[2]*bs[3] # ex: 4x4x4x4
    nb = int(vol/blk_vol) # number of blocks
    nbh = int(nb/2) # number of black blocks
    blk_dim = int(dim/nb) # dimension of a block

    print("time", time.time(), time.time() - lastime); lastime = time.time();
    print("Having {0} blocks in total".format(nb))
    print("Preparing the {0} block operators Dop_blk (sdtype)".format(nb**2))

    Dop_blks_sdtype_Re = np.empty((nb, nb), dtype=csr_matrix)
    Dop_blks_sdtype_Im = np.empty((nb, nb), dtype=csr_matrix)

    Dop_diag_sdtype_Re = np.empty(nb, dtype=csr_matrix2)
    Dop_diag_sdtype_Im = np.empty(nb, dtype=csr_matrix2)

    # first all black external boundaries
    for ib in range(0, nb):
        i_start = blk_dim*ib
        i_end   = blk_dim*(ib + 1)
        for jb in range(0, nb):
            j_start = blk_dim*(jb)
            j_end   = blk_dim*(jb + 1)
            if ib == jb:
                if args.device == 'hybrid':
                    arr1 = csr_matrix(Dop_ddtype_Re[i_start:i_end,j_start:j_end], dtype=sdtype)
                    arr2 = csr_matrix(Dop_ddtype_Im[i_start:i_end,j_start:j_end], dtype=sdtype)
                    Dop_diag_sdtype_Re[ib] = csr_matrix2((xp2.array(arr1.data), xp2.array(arr1.indices), xp2.array(arr1.indptr)), shape=arr1.shape)
                    Dop_diag_sdtype_Im[ib] = csr_matrix2((xp2.array(arr2.data), xp2.array(arr2.indices), xp2.array(arr2.indptr)), shape=arr2.shape)
                    del arr1, arr2
                else:
                    Dop_diag_sdtype_Re[ib] = csr_matrix(Dop_ddtype_Re[i_start:i_end,j_start:j_end], dtype=sdtype)
                    Dop_diag_sdtype_Im[ib] = csr_matrix(Dop_ddtype_Im[i_start:i_end,j_start:j_end], dtype=sdtype)
            else:
                Dop_blks_sdtype_Re[ib,jb] = csr_matrix(Dop_ddtype_Re[i_start:i_end,j_start:j_end], dtype=sdtype)
                Dop_blks_sdtype_Im[ib,jb] = csr_matrix(Dop_ddtype_Im[i_start:i_end,j_start:j_end], dtype=sdtype)
                #if Dop_blks_sdtype_Re[ib,jb].shape != (3072, 3072): print(Dop_blks_sdtype_Re[ib,jb].shape)

    print("time", time.time(), time.time() - lastime); lastime = time.time();

    return vol, nmx, res, b_ddtype_Re, b_ddtype_Im, x0_ddtype_Re, x0_ddtype_Im, Dop_ddtype_Re, Dop_ddtype_Im, Dop_blks_sdtype_Re, Dop_blks_sdtype_Im, Dop_diag_sdtype_Re, Dop_diag_sdtype_Im, lastime, csr_matrix, bmat


if __name__ == '__main__':
    multiprocessing.set_start_method('forkserver')
    dtypes = ["binary16", "binary32", "binary64", "posit8", "posit16", "posit32", "bfloat16", "tfloat32"]
    parser = argparse.ArgumentParser(description='Ax = b, with A = Dop() using GCR with SAP-preconditioning (the block-problems are solved by MR steps).')
    parser.add_argument('-s', '--sdtype', help='"small" datatype.', default="binary64", choices=dtypes)
    parser.add_argument('-c', '--cdtype', help='datatype for the colletive (reduction) variables.', default="binary64", choices=dtypes)
    parser.add_argument('-d', '--ddtype', help='"large" datatype.', default="binary64", choices=dtypes)
    parser.add_argument('--vol', help='spacetime volume (vol=L1*L2*L3*L4).', type=int)
    parser.add_argument('-r', '--res', help='relative residue.', type=float)
    parser.add_argument('--nmx', help='maximal number of iterations for the outer GCR.', type=int)
    parser.add_argument('--nkv', help='number of iterations for the outer GCR when to restart.', type=int, default=7)
    parser.add_argument('--ncy', help='number of Schwarz cycles.', type=int, default=2)
    parser.add_argument('--nmr', help='number of inner MR steps (blocks).', type=int, default=5)
    parser.add_argument('-np', '--no_preconditioning', help='whether to use SAP-preconditioning or not.', default=False, action="store_true")
    parser.add_argument('-n', '--runs', help='number of runs to perform.', type=int, default=1)
    parser.add_argument('-nr', '--no_reordering', help='whether to use reorder the blocks or not.', default=False, action="store_true")
    parser.add_argument('--device', help='the device to use, can be either cpu, gpu or hybrid.', default='cpu', choices=['cpu', 'gpu', 'hybrid'])
    parser.add_argument('-b0', help='block size in direction 0, The spacetime volume should be a multiple of this in every direction. default is (b0xb1xb2xb3=4x4x4x4).', type=int, default=4)
    parser.add_argument('-b1', help='block size in direction 1 (see -b0).', type=int, default=4)
    parser.add_argument('-b2', help='block size in direction 2 (see -b0).', type=int, default=4)
    parser.add_argument('-b3', help='block size in direction 3 (see -b0).', type=int, default=4)
    parser.add_argument('-o', '--out', help='output json file with result + diagnostics (defaults to STDOUT).', default="-")
    parser.add_argument('-m', '--matfile', help='optional file containing the matrix (can be matrix market format).', default=None)
    parser.add_argument('-k', help='the critical parameter 0 <= k < kc (only for matrices given by -m).', type=float, default=0.0)
    parser.add_argument('-kc', help='the critical value of k (only for matrices given by -m).', type=float, default=0.0)
    parser.add_argument('--eta', help='b vector in Ax=b.')
    parser.add_argument('-rb', '--random_b', help='random vallues for b=eta.', default=False, action="store_true")
    
    parser.add_argument('-7', '--seven', help='enable 7-input file.', default=False, action="store_true")
    parser.add_argument('-ie', '--env', help='env file.')
    parser.add_argument('-idr', '--data_real', help='real data file.')
    parser.add_argument('-idi', '--data_imag', help='imaginary data file.')
    parser.add_argument('-iir', '--index_real', help='real index file.')
    parser.add_argument('-iii', '--index_imag', help='imaginary index file.')
    parser.add_argument('-ipr', '--ptr_real', help='real ptr file.')
    parser.add_argument('-ipi', '--ptr_imag', help='imaginary ptr file.')

    parser.add_argument('-l', '--little', help='the file containing the little operator.')
    parser.add_argument('--dfl_nmx', help='maximal number of iterations when solving the little equation.', type=int, default=256)
    parser.add_argument('--dfl_nkv', help='maximal number of iterations when to restart solving the little equation.', type=int, default=24)
    parser.add_argument('--dfl_res', help='relative residue for the little equation.', type=float, default=1e-2)

    parser.add_argument('-v', '--verbose', help='amount of debug messages, the higher the more (0: nothing, 1:info, 2:much).', type=int, default=1)

    # extension arguments
    parser.add_argument('-e', '--extension', help='activation of extension (0: no extension, 1:adaptive).', type=int, default=0)
    parser.add_argument('--ext_alpha', help='If extension=1, the alpha parameter.', type=float, default=0.9)
    parser.add_argument('--ext_beta', help='If extension=1, the beta parameter.', type=float, default=1)

    required = parser.add_argument_group('required named arguments')
    required.add_argument('-i', '--infile', help='the input file containing the parameters and Dop in double precision in CSR format.', required=True)
    args = parser.parse_args()
    bs = np.array([args.b0, args.b1, args.b2, args.b3], dtype=int)
    args.nmr = 0 if args.no_preconditioning else args.nmr
    args.ncy = 0 if args.no_preconditioning else args.ncy

    # "large" datatype
    if args.ddtype == "binary16":
        ddtype = np.float16
    elif args.ddtype == "binary32":
        ddtype = np.float32
    elif args.ddtype == "binary64":
        ddtype = np.float64
    elif args.ddtype == "posit8":
        ddtype = sp.posit8
    elif args.ddtype == "posit16":
        ddtype = sp.posit16
    elif args.ddtype == "posit32":
        ddtype = sp.posit32
    elif args.ddtype == "bfloat16":
        ddtype = fl.bfloat16
    elif args.ddtype == "tfloat32":
        ddtype = fl.tfloat32

    # simulation datatype
    if args.sdtype == "binary16":
        sdtype = np.float16
    elif args.sdtype == "binary32":
        sdtype = np.float32
    elif args.sdtype == "binary64":
        sdtype = np.float64
    elif args.sdtype == "posit8":
        sdtype = sp.posit8
    elif args.sdtype == "posit16":
        sdtype = sp.posit16
    elif args.sdtype == "posit32":
        sdtype = sp.posit32
    elif args.sdtype == "bfloat16":
        sdtype = fl.bfloat16
    elif args.sdtype == "tfloat32":
        sdtype = fl.tfloat32

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

    vol, nmx, res, b_ddtype_Re, b_ddtype_Im, x0_ddtype_Re, x0_ddtype_Im, Dop_ddtype_Re, Dop_ddtype_Im, Dop_blks_sdtype_Re, Dop_blks_sdtype_Im, Dop_diag_sdtype_Re, Dop_diag_sdtype_Im, lastime, csr_matrix, bmat = load_matrix_from_file(args, ddtype, sdtype )


    if args.device == 'gpu':
        xp = cp
        xp2 = cp
    elif args.device == 'hybrid':
        xp = np
        xp2 = cp
    else:
        xp = np
        xp2 = np

    dim = Dop_ddtype_Re.shape[0]
    blk_vol = bs[0]*bs[1]*bs[2]*bs[3] # ex: 4x4x4x4
    nb = int(vol/blk_vol) # number of blocks
    nbh = int(nb/2) # number of black blocks
    blk_dim = int(dim/nb) # dimension of a block
    block_reorder = np.arange(0, nb) # first proposal for the block order

    # takes real amd imag parts and returns real and imaginary parts in ddtype
    # @param xp.array re real part of vector (size=problemsize)
    # @param xp.array im imag part of vector (size=problemsize)
    # @return xp.array, xp.array real and imag part of the result (size=problemsize)
    def Dop_ddtype(re, im):
        return complex_matrix_vector_prod(Dop_ddtype_Re, Dop_ddtype_Im, re, im, ddtype)

    Dop_sdtype_Re = csr_matrix(Dop_ddtype_Re, dtype=sdtype)
    Dop_sdtype_Im = csr_matrix(Dop_ddtype_Im, dtype=sdtype)

    # takes real amd imag parts and returns real and imaginary parts in sdtype
    # @param xp.array re real part of vector (size=problemsize)
    # @param xp.array im imag part of vector (size=problemsize)
    # @return xp.array, xp.array real and imag part of the result (size=problemsize)
    def Dop_sdtype(re, im):
        return complex_matrix_vector_prod(Dop_sdtype_Re, Dop_sdtype_Im, re, im, sdtype)

    # @param xp.array re real part of vector (size=blocksize)
    # @param xp.array im imag part of vector (size=blocksize)
    # @param int n number of the block
    # @return xp.array, xp.array real and imag part of the result (size=problemsize)
    def Dop_int_bnd_sdtype(n, re, im):
        res_Re = xp.zeros(dim)
        res_Im = xp.zeros(dim)

        for i in range(0, nb):
            if i == n: continue # skip the diagonal (block) part
            blk, bs, be = range_block(blk_dim, blk_vol, i)
            res_Re[bs:be], res_Im[bs:be] = complex_matrix_vector_prod(Dop_blks_sdtype_Re[i,n], Dop_blks_sdtype_Im[i,n], re, im, sdtype)
        return res_Re, res_Im

    # @param xp.array re real part of vector (size=blocksize)
    # @param xp.array im imag part of vector (size=blocksize)
    # @param int n number of the block
    # @return xp.array, xp.array real and imag part of the result (size=blocksize)
    def Dop_blk_sdtype(n, re, im):
        return complex_matrix_vector_prod(Dop_diag_sdtype_Re[n], Dop_diag_sdtype_Im[n], re, im, ddtype)

    if args.no_preconditioning == False:

        if args.no_reordering == False:
            print("time", time.time(), time.time() - lastime); lastime = time.time();
            print("Determining reordering of the blocks")

            # determine the block order, such that all blocks that have no interaction among themselves
            # are associated to the 2 colors black and white. By definition, even (black) indices come first,
            # then odd (white) indices.
            wrk_Re = xp.ones(blk_dim)
            wrk_Im = xp.ones(blk_dim)

            black_blocks = np.array([0], dtype=np.int64) # first block is by definition black
            white_blocks = np.array([], dtype=np.int64)
            all_blocks = np.array([0], dtype=np.int64)
            it, maxit = 0, 10

            while all_blocks.size < nb and it <= maxit:
                it += 1
                for n in range(0, nb):
                    Dwrk_Re, Dwrk_Im = Dop_int_bnd_sdtype(n, wrk_Re, wrk_Im)
                    for m in range(0, nb):
                        br, s, e = range_block(blk_dim, blk_vol, m)
                        if xp.count_nonzero(Dwrk_Re[s:e]) != 0: # the block should be in the opposite color
                            if n in black_blocks:
                                white_blocks = np.append(white_blocks, [m])
                                all_blocks = np.unique(np.append(all_blocks, [m]))
                            elif n in white_blocks:
                                black_blocks = np.append(black_blocks, [m])
                                all_blocks = np.unique(np.append(all_blocks, [m]))

            black_blocks = np.unique(black_blocks)
            white_blocks = np.unique(white_blocks)
            block_reorder = np.append(black_blocks, white_blocks) # first the black, then white
            print("block_reorder", block_reorder)

            if it == maxit+1 or block_reorder.size != nb:
                print("Error: not able to reorder the blocks in a way such that all even and odd are independent.")
                exit(1)


        print("time", time.time(), time.time() - lastime); lastime = time.time();
        print("Construct the black and white color internal boundary operators")

        if args.no_reordering:

            dh = int(dim/2)
            z = csr_matrix((int(dim/2), int(dim/2)), dtype=sdtype)

            # lower left block
            Dop_black_int_bnd_sdtype_Re = bmat([[z, z], [Dop_ddtype_Re[dh:dim,0:dh], z]], format='csr', dtype=sdtype)
            Dop_black_int_bnd_sdtype_Im = bmat([[z, z], [Dop_ddtype_Im[dh:dim,0:dh], z]], format='csr', dtype=sdtype)
            # upper right block
            Dop_white_int_bnd_sdtype_Re = bmat([[z, Dop_ddtype_Re[0:dh,dh:dim]], [z, z]], format='csr', dtype=sdtype)
            Dop_white_int_bnd_sdtype_Im = bmat([[z, Dop_ddtype_Im[0:dh,dh:dim]], [z, z]], format='csr', dtype=sdtype)

        else:

            Dop_black_int_bnd_sdtype_Re = sc.sparse.lil_matrix((dim, dim), dtype=sdtype)
            Dop_black_int_bnd_sdtype_Im = sc.sparse.lil_matrix((dim, dim), dtype=sdtype)
            Dop_white_int_bnd_sdtype_Re = sc.sparse.lil_matrix((dim, dim), dtype=sdtype)
            Dop_white_int_bnd_sdtype_Im = sc.sparse.lil_matrix((dim, dim), dtype=sdtype)

            # construct the black operator
            for bb in black_blocks:
                bbr, bbs, bbe = range_block(blk_dim, blk_vol, bb)
                for wb in white_blocks:
                    wbr, wbs, wbe = range_block(blk_dim, blk_vol, wb)
                    if args.device == 'gpu':
                        Dop_black_int_bnd_sdtype_Re[wbs:wbe,bbs:bbe] = Dop_blks_sdtype_Re[wb,bb].get()
                        Dop_black_int_bnd_sdtype_Im[wbs:wbe,bbs:bbe] = Dop_blks_sdtype_Im[wb,bb].get()
                    else:
                        Dop_black_int_bnd_sdtype_Re[wbs:wbe,bbs:bbe] = Dop_blks_sdtype_Re[wb,bb]
                        Dop_black_int_bnd_sdtype_Im[wbs:wbe,bbs:bbe] = Dop_blks_sdtype_Im[wb,bb]

            Dop_black_int_bnd_sdtype_Re = Dop_black_int_bnd_sdtype_Re.tocsr()
            Dop_black_int_bnd_sdtype_Im = Dop_black_int_bnd_sdtype_Im.tocsr()

            # construct the white operator
            for wb in white_blocks:
                wbr, wbs, wbe = range_block(blk_dim, blk_vol, wb)
                for bb in black_blocks:
                    bbr, bbs, bbe = range_block(blk_dim, blk_vol, bb)
                    if args.device == 'gpu':
                        Dop_white_int_bnd_sdtype_Re[bbs:bbe,wbs:wbe] = Dop_blks_sdtype_Re[bb,wb].get()
                        Dop_white_int_bnd_sdtype_Im[bbs:bbe,wbs:wbe] = Dop_blks_sdtype_Im[bb,wb].get()
                    else:
                        Dop_white_int_bnd_sdtype_Re[bbs:bbe,wbs:wbe] = Dop_blks_sdtype_Re[bb,wb]
                        Dop_white_int_bnd_sdtype_Im[bbs:bbe,wbs:wbe] = Dop_blks_sdtype_Im[bb,wb]

            Dop_white_int_bnd_sdtype_Re = Dop_white_int_bnd_sdtype_Re.tocsr()
            Dop_white_int_bnd_sdtype_Im = Dop_white_int_bnd_sdtype_Im.tocsr()

            # transfer the arrays to the GPU (if csr_matrix is cupy)
            Dop_black_int_bnd_sdtype_Re = csr_matrix(Dop_black_int_bnd_sdtype_Re)
            Dop_black_int_bnd_sdtype_Im = csr_matrix(Dop_black_int_bnd_sdtype_Im)
            Dop_white_int_bnd_sdtype_Re = csr_matrix(Dop_white_int_bnd_sdtype_Re)
            Dop_white_int_bnd_sdtype_Im = csr_matrix(Dop_white_int_bnd_sdtype_Im)

        # @param int color the color (0 for black, 1 for white)
        # @param xp.array re real part of vector (size=problemsize)
        # @param xp.array im imag part of vector (size=problemsize)
        # @return xp.array, xp.array real and imag part of the result (size=problemsize)
        def Dop_int_color_bnd_sdtype(color, re, im):
            if color == 0:
                return complex_matrix_vector_prod(Dop_black_int_bnd_sdtype_Re, Dop_black_int_bnd_sdtype_Im, re, im, sdtype)
            elif color == 1:
                return complex_matrix_vector_prod(Dop_white_int_bnd_sdtype_Re, Dop_white_int_bnd_sdtype_Im, re, im, sdtype)


    # Preconditioning operator that does no preconditioning
    def Mop_no_preconditioning(k, rho_Re, rho_Im, sdtype, dtype, tol, Dop):
        #phi_Re, phi_Im = xp.copy(rho_Re), xp.copy(rho_Im)
        chi_Re, chi_Im = Dop(rho_Re, rho_Im)
        return rho_Re, rho_Im, chi_Re, chi_Im

    # Preconditioning operator that contains sap-preconditioning
    def Mop_sap(k, rho_Re, rho_Im, sdtype, cdtype, tol, args, Dop_blk_sdtype):
        phi_Re = xp.zeros_like(rho_Re)
        phi_Im = xp.zeros_like(rho_Im)
        chi_Re = xp.copy(rho_Re)
        chi_Im = xp.copy(rho_Im)
        
        if args.extension == 1: rn = np.sqrt(complex_norm_sq(chi_Re, chi_Im, cdtype))
        
        tnmr = 0
        for n in range(0, args.ncy):
            phi_Re, phi_Im, chi_Re, chi_Im, anmr = sap(n, vol, args.nmr, phi_Re, phi_Im, chi_Re, chi_Im, bs, Dop_blk_sdtype, cdtype, xp, xp2, tol)
            if args.extension == 1:
                tnmr += anmr*nb
                rn_old = rn
                rn = np.sqrt(complex_norm_sq(chi_Re, chi_Im, cdtype))
                b = rn/rn_old
                if args.verbose > 1: print('sap() icy={0}, ncy={1}, rn={2} > tol={3}, beta={4}, nmr={5}'.format(n, args.ncy, rn, tol, b, args.nmr))
                if (b >= max(args.ext_beta, 1)) or b >= args.ext_beta or rn < tol: break

        if args.extension == 1 and args.verbose > 0: print('sap() took ncy={0}/{1}, avg nmr={2}/{3}'.format(n+1, args.ncy, tnmr/(nb*(n+1)), args.nmr))
        chi_Re, chi_Im = axpy(-1.0, chi_Re, chi_Im, rho_Re, rho_Im, sdtype)
        return phi_Re, phi_Im, chi_Re, chi_Im

    # Preconditioning operator that solves the little equation and containing sap-preconditioning
    def Mop_dfl_sap(k, rho_Re, rho_Im, sdtype, cdtype, tol, args, orth_Re, orth_Im, ltl_Dop, ltl_Mop, Dop):
        v_Re, v_Im = xp.empty(dfl_dim), xp.empty(dfl_dim)
        for i in range(0, dfl_dim):
            v_Re[i], v_Im[i] = complex_scalar_prod(orth_Re[i], orth_Im[i], rho_Re, rho_Im)
        v_Re, v_Im, vn, j = fgcr(dfl_dim, dfl_dim/dfl_nb, dfl_nb, args.dfl_nmx, args.dfl_res, args.dfl_nkv, 0, 0, ltl_Dop, ltl_Mop, ltl_Dop, v_Re, v_Im, v_Re, v_Im, ddtype, sdtype, cdtype, xp, xp2, args, False)
        wrk_Re, wrk_Im = xp.zeros_like(rho_Re), xp.zeros_like(rho_Im)
        for i in range(0, dfl_dim): # TODO: this is the thing that takes long
            wrk_Re, wrk_Im = complex_axpy(v_Re[i], v_Im[i], orth_Re[i], orth_Im[i], wrk_Re, wrk_Im)
        
        chi_Re, chi_Im = Dop(wrk_Re, wrk_Im)
        chi_Re, chi_Im = axpy(-1.0, chi_Re, chi_Im, rho_Re, rho_Im, sdtype)
        phi_Re = xp.zeros_like(rho_Re)
        phi_Im = xp.zeros_like(rho_Im)

        if args.extension == 1: rn = np.sqrt(complex_norm_sq(chi_Re, chi_Im, cdtype))
        
        tnmr = 0
        for n in range(0, args.ncy):
            phi_Re, phi_Im, chi_Re, chi_Im, anmr = sap(n, vol, args.nmr, phi_Re, phi_Im, chi_Re, chi_Im, bs, Dop_blk_sdtype, cdtype, xp, xp2, tol)
            if args.extension == 1:
                tnmr += anmr*nb
                rn_old = rn
                rn = np.sqrt(complex_norm_sq(chi_Re, chi_Im, cdtype))
                b = rn/rn_old
                if args.verbose > 1: print('sap() icy={0}, ncy={1}, rn={2} > tol={3}, beta={4}, nmr={5}'.format(n, args.ncy, rn, tol, b, args.nmr))
                if (b >= max(args.ext_beta, 1)) or b >= args.ext_beta or rn < tol: break

        if args.extension == 1 and args.verbose > 0: print('sap() took ncy={0}/{1}, avg nmr={2}/{3}'.format(n+1, args.ncy, tnmr/(nb*(n+1)), args.nmr))
        chi_Re, chi_Im = axpy(-1.0, chi_Re, chi_Im, rho_Re, rho_Im, sdtype)
        phi_Re, phi_Im = axpy(1.0, wrk_Re, wrk_Im, phi_Re, phi_Im)
        return phi_Re, phi_Im, chi_Re, chi_Im

    # set up the actual preconditioner that is used
    if args.no_preconditioning:
        Mop = lambda k, rho_Re, rho_Im, sdtype, dtype, tol: Mop_no_preconditioning(k, rho_Re, rho_Im, sdtype, dtype, tol, Dop_sdtype)
    elif args.little == None:
        Mop = lambda k, rho_Re, rho_Im, sdtype, dtype, tol: Mop_sap(k, rho_Re, rho_Im, sdtype, dtype, tol, args, Dop_blk_sdtype)
    else:
        with open(args.little, 'rb') as f:
            print("extracting little Dop")
            fmt = '<{0}d'
            int_size, float_size, double_size = 4, 4, 8 # bytes
            dfl_Ns = struct.unpack('<1I', f.read(int_size))[0]
            dfl_nb = struct.unpack('<1I', f.read(int_size))[0]
            dfl_dim = dfl_nb*dfl_Ns
            A_Re = np.zeros((dfl_dim, dfl_dim))
            A_Im = np.zeros((dfl_dim, dfl_dim))
            orth_Re = np.empty((dfl_dim, dim))
            orth_Im = np.empty((dfl_dim, dim))
            for l in range(dfl_dim):
                A_Re[l] = struct.unpack('<{0}d'.format(dfl_dim), f.read(double_size*dfl_dim))
            for l in range(dfl_dim):
                A_Im[l] = struct.unpack('<{0}d'.format(dfl_dim), f.read(double_size*dfl_dim))
            for l in range(dfl_dim):
                orth_Re[l] = struct.unpack('<{0}d'.format(dim), f.read(double_size*dim))
            for l in range(dfl_dim):
                orth_Im[l] = struct.unpack('<{0}d'.format(dim), f.read(double_size*dim))

            # the little Dirac operator
            ltl_Dop = lambda re, im: complex_matrix_vector_prod(A_Re, A_Im, re, im, sdtype)

            # the preconditioning operator for the little equation; does no preconditioning
            ltl_Mop = lambda k, rho_Re, rho_Im, sdtype, dtype, tol: Mop_no_preconditioning(k, rho_Re, rho_Im, sdtype, dtype, tol, ltl_Dop)
        
            # preconditioning includes deflation
            Mop = lambda k, rho_Re, rho_Im, sdtype, dtype, tol: Mop_dfl_sap(k, rho_Re, rho_Im, sdtype, dtype, tol, args, orth_Re, orth_Im, ltl_Dop, ltl_Mop, Dop_sdtype)

    # truncate the spinor arrays if necessary, such that they match the Dop()
    b_ddtype_Re = xp.array(b_ddtype_Re[0:dim])
    b_ddtype_Im = xp.array(b_ddtype_Im[0:dim])
    x0_ddtype_Re = xp.array(x0_ddtype_Re[0:dim])
    x0_ddtype_Im = xp.array(x0_ddtype_Im[0:dim])

    #br, bi = xp.array(b_ddtype_Re, dtype=sdtype), xp.array(b_ddtype_Im, dtype=sdtype)
    #sci, scr = Dop_sdtype(bi, br)
    #dci, dcr = Dop_ddtype(b_ddtype_Re, b_ddtype_Im)
    #print(complex_norm_sq(sci, scr, cdtype))
    #print(complex_norm_sq(dci, dcr, cdtype))
    #exit()


    # by launching the fgcr() solver with only one step, all GPU-kernel get comiled
    print("preparing GPU kernels ...")
    a1, a2, a3, a4 = fgcr(dim, blk_dim, vol, 1, res, args.nkv, args.ncy, args.nmr, Dop_ddtype, Mop, Dop_ddtype, b_ddtype_Re, b_ddtype_Im, x0_ddtype_Re, x0_ddtype_Im, ddtype, sdtype, cdtype, xp, xp2, args, False)
    del a1, a2, a3, a4

    # fill the duration array with the run-times
    duration = []
    for i in range(args.runs):
        print("time", time.time(), time.time() - lastime); lastime = time.time();
        print("calculating Ax=b using SAP_GCR")
        x_Re, x_Im, rn, json_data = fgcr(dim, blk_dim, vol, nmx, res, args.nkv, args.ncy, args.nmr, Dop_ddtype, Mop, Dop_ddtype, b_ddtype_Re, b_ddtype_Im, x0_ddtype_Re, x0_ddtype_Im, ddtype, sdtype, cdtype, xp, xp2, args, True)
        print("time", time.time(), time.time() - lastime); lastime = time.time();
        print(rn, type(x_Re[0]), type(x_Im[0]))
        print("status", json_data['status'])
        duration.append(json_data['duration'])

        if json_data['status'] == -1:
            break

    json_data['duration'] = duration

    # the solution is verified
    print("checking |Ax - b| ...")
    Dx_Re, Dx_Im = Dop_ddtype(x_Re, x_Im)
    r_Re, r_Im = axpy(-1.0, Dx_Re, Dx_Im, b_ddtype_Re, b_ddtype_Im, ddtype)
    r = np.sqrt(complex_norm_sq(r_Re, r_Im))
    if (np.isnan(r) or np.isinf(r)):
        exit_code = 1
    else:
        exit_code = 0
    print("rn = {0}".format(r))

    if args.out == "-":
        json.dump(json_data, sys.stdout, indent=2)
        print("\n")
    else:
        with open(args.out, 'w') as outfile:
            json.dump(json_data, outfile, indent=2)

    exit(exit_code)
