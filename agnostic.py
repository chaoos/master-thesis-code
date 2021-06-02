#!/usr/bin/env python3
#
# Device agnostic functions
# should work with cupy arrays as well as numpy arrays

from __future__ import division
from numba import cuda, float32, float64
import numpy as np
import math
import time
import scipy.linalg
import cupy as cp
import cupyx as cpx
import scipy as sc

# A: real part of matrix (must be a scipy.sparse of cupy.sparse matrix)
# B: imag part of matrix (must be a scipy.sparse of cupy.sparse matrix)
# x: real part of vector
# y: imag part of vector
def complex_matrix_vector_prod(A, B, x, y, dtype=np.float64):
    return A.dot(x) - B.dot(y), B.dot(x) + A.dot(y)

# A: real part of matrix (must be a scipy.sparse of cupy.sparse matrix)
# B: imag part of matrix (must be a scipy.sparse of cupy.sparse matrix)
# x: real part of vector
# y: imag part of vector
def complex_matrix_dagger_vector_prod(A, B, x, y, dtype=np.float64):
    return A.transpose().dot(x) + B.transpose().dot(y), A.transpose().dot(y) - B.transpose().dot(x)

def real_scalar_prod(x, y, dtype=np.float64):
    raise Exception('not yet implemented using cdtype')
    return x.dot(y)

def real_norm_sq(x, dtype=np.float64):
    raise Exception('not yet implemented using cdtype')
    return x.dot(x)

# cuda real part of complex norm reduction with a float64 output datatype
complex_scalar_prod_Re_cuda_cdtype_binary64 = cp.ReductionKernel(
    'T x_Re, T x_Im, T y_Re, T y_Im',  # input params
    'float64 z',  # output params
    'x_Re*y_Re + x_Im*y_Im',  # map
    'a + b',  # reduce
    'z = a',  # post-reduction map
    '0',  # identity value
    'complex_scalar_prod_Re_cuda_cdtype_binary64',  # kernel name
    'double' # reduce_type
)

# cuda imag part of complex norm reduction with a float64 output datatype
complex_scalar_prod_Im_cuda_cdtype_binary64 = cp.ReductionKernel(
    'T x_Re, T x_Im, T y_Re, T y_Im',  # input params
    'float64 z',  # output params
    'x_Re*y_Im - x_Im*y_Re',  # map
    'a + b',  # reduce
    'z = a',  # post-reduction map
    '0',  # identity value
    'complex_scalar_prod_Im_cuda_cdtype_binary64',  # kernel name
    'double' # reduce_type
)

# cuda real part of complex norm reduction with a float32 output datatype
complex_scalar_prod_Re_cuda_cdtype_binary32 = cp.ReductionKernel(
    'T x_Re, T x_Im, T y_Re, T y_Im',  # input params
    'float32 z',  # output params
    'x_Re*y_Re + x_Im*y_Im',  # map
    'a + b',  # reduce
    'z = a',  # post-reduction map
    '0',  # identity value
    'complex_scalar_prod_Re_cuda_cdtype_binary32',  # kernel name
    'float' # reduce_type
)

# cuda imag part of complex norm reduction with a float32 output datatype
complex_scalar_prod_Im_cuda_cdtype_binary32 = cp.ReductionKernel(
    'T x_Re, T x_Im, T y_Re, T y_Im',  # input params
    'float32 z',  # output params
    'x_Re*y_Im - x_Im*y_Re',  # map
    'a + b',  # reduce
    'z = a',  # post-reduction map
    '0',  # identity value
    'complex_scalar_prod_Im_cuda_cdtype_binary32',  # kernel name
    'float' # reduce_type
)

def complex_scalar_prod(x_Re, x_Im, y_Re, y_Im, dtype=np.float64):
    if isinstance(x_Re, np.ndarray):
        if type(x_Re[0]) == dtype:
            return x_Re.dot(y_Re) + x_Im.dot(y_Im), x_Re.dot(y_Im) - x_Im.dot(y_Re)
        else:
            return np.add.reduce(x_Re*y_Re, dtype=dtype) + np.add.reduce(x_Im*y_Im, dtype=dtype), np.add.reduce(x_Re*y_Im, dtype=dtype) - np.add.reduce(x_Im*y_Re, dtype=dtype)
    else:
        if dtype == np.float64:
            return complex_scalar_prod_Re_cuda_cdtype_binary64(x_Re, x_Im, y_Re, y_Im), complex_scalar_prod_Im_cuda_cdtype_binary64(x_Re, x_Im, y_Re, y_Im)
        elif dtype == np.float32:
            return complex_scalar_prod_Re_cuda_cdtype_binary32(x_Re, x_Im, y_Re, y_Im), complex_scalar_prod_Im_cuda_cdtype_binary32(x_Re, x_Im, y_Re, y_Im)
        else:
            raise Exception('Unknown cdtype {0}'.format(dtype))

# cuda complex norm reduction with a float64 output datatype
complex_norm_sq_cuda_cdtype_binary64 = cp.ReductionKernel(
    'T x_Re, T x_Im',  # input params
    'float64 z',  # output params
    'x_Re*x_Re + x_Im*x_Im',  # map
    'a + b',  # reduce
    'z = a',  # post-reduction map
    '0',  # identity value
    'complex_norm_sq_cuda_cdtype_binary64',  # kernel name
    'double' # reduce_type
)

# cuda complex norm reduction with a float32 output datatype
complex_norm_sq_cuda_cdtype_binary32 = cp.ReductionKernel(
    'T x_Re, T x_Im',  # input params
    'float32 z',  # output params
    'x_Re*x_Re + x_Im*x_Im',  # map
    'a + b',  # reduce
    'z = a',  # post-reduction map
    '0',  # identity value
    'complex_norm_sq_cuda_cdtype_binary32',  # kernel name
    'float' # reduce_type
)

def complex_norm_sq(re, im, dtype=np.float64):
    if isinstance(re, np.ndarray):
        if type(re[0]) == dtype:
            return re.dot(re) + im.dot(im)
        else:
            return np.add.reduce(re**2, dtype=dtype) + np.add.reduce(im**2, dtype=dtype)
    else:
        if dtype == np.float64:
            return complex_norm_sq_cuda_cdtype_binary64(re, im)
        elif dtype == np.float32:
            return complex_norm_sq_cuda_cdtype_binary32(re, im)
        else:
            raise Exception('Unknown cdtype {0}'.format(dtype))

complex_axpy_Re_cuda = cp.ElementwiseKernel(
   'A a_Re, A a_Im, T x_Re, T x_Im, T y_Re', # in_params 
   'T z', # out_params 
   'z = a_Re*x_Re - a_Im*x_Im + y_Re', # operation 
   'complex_axpy_Re_cuda' # kernel name
)

complex_axpy_Im_cuda = cp.ElementwiseKernel(
   'A a_Re, A a_Im, T x_Re, T x_Im, T y_Im', # in_params 
   'T z', # out_params 
   'z = a_Re*x_Im + a_Im*x_Re + y_Im', # operation 
   'complex_axpy_Im_cuda' # kernel name
)

# dtype has no meaning here
def complex_axpy(a_Re, a_Im, x_Re, x_Im, y_Re, y_Im, dtype=np.float64):
    if isinstance(x_Re, np.ndarray): # use blas in case of numpy
        copy_Re = np.array(y_Re, copy = True)
        copy_Im = np.array(y_Im, copy = True)
        if type(x_Re[0]) == np.float64:
            blas_axpy = scipy.linalg.blas.daxpy
        else:
            blas_axpy = scipy.linalg.blas.saxpy
        blas_axpy(x_Re, copy_Re, a = a_Re)
        blas_axpy(x_Im, copy_Re, a =-a_Im)
        blas_axpy(x_Re, copy_Im, a = a_Im)
        blas_axpy(x_Im, copy_Im, a = a_Re)
        return copy_Re, copy_Im
    else:
        return complex_axpy_Re_cuda(a_Re, a_Im, x_Re, x_Im, y_Re), complex_axpy_Im_cuda(a_Re, a_Im, x_Re, x_Im, y_Im)
        #return a_Re*x_Re - a_Im*x_Im + y_Re, a_Re*x_Im + a_Im*x_Re + y_Im

axpy_cuda = cp.ElementwiseKernel(
   'A a, T x, T y', # in_params 
   'T z', # out_params 
   'z = a*x + y', # operation 
   'axpy_cuda' # kernel name
)

# dtype has no meaning here
def axpy(a, x_Re, x_Im, y_Re, y_Im, dtype=np.float64):
    #return a*x_Re + y_Re, a*x_Im + y_Im
    if isinstance(x_Re, np.ndarray): # use blas in case of numpy
        copy_Re = np.array(y_Re, copy = True) # arrays need to be copied, else they are overwritten y <- ax + y
        copy_Im = np.array(y_Im, copy = True)
        if type(x_Re[0]) == np.float64:
            blas_axpy = scipy.linalg.blas.daxpy
        else:
            blas_axpy = scipy.linalg.blas.saxpy
        blas_axpy(x_Re, copy_Re, a = a)
        blas_axpy(x_Im, copy_Im, a = a)
        return copy_Re, copy_Im
    else:
        A = cp.array(a, dtype=dtype)
        return axpy_cuda(A, x_Re, y_Re), axpy_cuda(A, x_Im, y_Im)

# run all the GPU kernels once, such that they are compiled
def compile_gpu_kernels(size):
    a_Re = cp.array(np.float64(1.0))
    a_Im = cp.array(np.float64(0.5))
    d_v1_Re = cp.array(np.random.rand(size), dtype=np.float64)
    d_v1_Im = cp.array(np.random.rand(size), dtype=np.float64)
    d_v2_Re = cp.array(np.random.rand(size), dtype=np.float64)
    d_v2_Im = cp.array(np.random.rand(size), dtype=np.float64)
    c, b = complex_axpy(a_Re, a_Im, d_v1_Re, d_v1_Im, d_v2_Re, d_v2_Im)
    c, b = axpy(a_Re, d_v1_Re, d_v1_Im, d_v2_Re, d_v2_Im)
    c = complex_norm_sq(d_v1_Re, d_v1_Im, np.float64)
    c = complex_norm_sq(d_v1_Re, d_v1_Im, np.float32)
    c, b = complex_scalar_prod(d_v1_Re, d_v1_Im, d_v2_Re, d_v2_Im, np.float64)
    c, b = complex_scalar_prod(d_v1_Re, d_v1_Im, d_v2_Re, d_v2_Im, np.float32)

    a_Re = cp.array(np.float32(1.0))
    a_Im = cp.array(np.float32(0.5))
    d_v1_Re = cp.array(np.random.rand(size), dtype=np.float32)
    d_v1_Im = cp.array(np.random.rand(size), dtype=np.float32)
    d_v2_Re = cp.array(np.random.rand(size), dtype=np.float32)
    d_v2_Im = cp.array(np.random.rand(size), dtype=np.float32)
    c, b = complex_axpy(a_Re, a_Im, d_v1_Re, d_v1_Im, d_v2_Re, d_v2_Im)
    c, b = axpy(a_Re, d_v1_Re, d_v1_Im, d_v2_Re, d_v2_Im)
    c = complex_norm_sq(d_v1_Re, d_v1_Im, np.float64)
    c = complex_norm_sq(d_v1_Re, d_v1_Im, np.float32)
    c, b = complex_scalar_prod(d_v1_Re, d_v1_Im, d_v2_Re, d_v2_Im, np.float64)
    c, b = complex_scalar_prod(d_v1_Re, d_v1_Im, d_v2_Re, d_v2_Im, np.float32)

# needed agnostic functions:
# np.zeros_like, np.zeros, np.array, convert, 

if __name__ == '__main__':
    from sap_gcr import load_matrix_from_file
    from cgne_pp2 import get_precision_limit, get_epsilon

    print("testing ...")
    print("setup ...")

    class Object(object):
        pass

    nt = 1
    size = 12*(8**4) # N of the matrix from args.infile
    dtype = np.float64
    cdtype = np.float64
    rtol = get_epsilon(dtype)*size*nt
    atol = get_epsilon(dtype)*size*nt
    args = Object()
    args.infile = '../matrix/cgne_init_4096_dble_csr.dat'
    args.gpu = False
    args.b0 = 4
    args.b1 = 4
    args.b2 = 4
    args.b3 = 4
    args.nmx = 256
    args.vol = None
    args.res = None
    args.nkv = 7
    args.ncy = 2
    args.nmr = 4

    def reset(size, dtype):
        return np.zeros(size, dtype=dtype), np.zeros(size, dtype=dtype)

    def device_reset(size, dtype):
        return cp.array(np.zeros(size, dtype=dtype), dtype=dtype), cp.array(np.zeros(size, dtype=dtype), dtype=dtype)

    print("preparing matrices (cpu) ...")

    vol, nmx, res, b_Re, b_Im, x0_Re, x0_Im, M_Re, M_Im, Dop_blk_Re, Dop_blk_Im, Dop_bnd_Re, Dop_bnd_Im, lastime = load_matrix_from_file(args)
    size = M_Re.shape[0]

    # numpy matrices
    M_Re = scipy.sparse.csr_matrix(M_Re, dtype=dtype)
    M_Im = scipy.sparse.csr_matrix(M_Im, dtype=dtype)

    print("preparing vectors (cpu) ...")

    # numpy vectors
    v1_Re = np.array(np.random.rand(nt, size), dtype=dtype)
    v2_Re = np.array(np.random.rand(nt, size), dtype=dtype)
    v1_Im = np.array(np.random.rand(nt, size), dtype=dtype)
    v2_Im = np.array(np.random.rand(nt, size), dtype=dtype)
    r_Re, r_Im = reset(size, dtype)

    print("preparing scalars (cpu) ...")

    # numpy scalars
    a_Re = np.array(np.random.rand(nt), dtype=dtype)
    a_Im = np.array(np.random.rand(nt), dtype=dtype)

    print("preparing vectors (gpu) ...")

    # cupy vectors
    d_v1_Re = cp.array(v1_Re, dtype=dtype)
    d_v2_Re = cp.array(v2_Re, dtype=dtype)
    d_v1_Im = cp.array(v1_Im, dtype=dtype)
    d_v2_Im = cp.array(v2_Im, dtype=dtype)
    d_r_Re, d_r_Im = device_reset(size, dtype)

    print("preparing scalars (cpu) ...")

    # cupy scalars
    d_a_Re = cp.array(a_Re, dtype=dtype)
    d_a_Im = cp.array(a_Im, dtype=dtype)

    print("preparing matrices (gpu) ...")

    # cupy matrices
    d_M_Re  = cpx.scipy.sparse.csr_matrix(M_Re, dtype=dtype)
    d_M_Im  = cpx.scipy.sparse.csr_matrix(M_Im, dtype=dtype)

    print("preparing GPU kernels ...")

    compile_gpu_kernels(size)
    a, b = complex_matrix_vector_prod(d_M_Re, d_M_Im, d_v1_Re[0], d_v1_Im[0])
    a, b = complex_matrix_dagger_vector_prod(d_M_Re, d_M_Im, d_v1_Re[0], d_v1_Im[0])
    del a,b

    print("testing complex_axpy() ({0} times)...".format(nt))

    lastime = time.time()
    for i in range(0, nt):
        r_Re, r_Im = complex_axpy(a_Re[i], a_Im[i], v1_Re[i], v1_Im[i], r_Re, r_Im)
    cpu_took = time.time() - lastime

    lastime = time.time()
    for i in range(0, nt):
        d_r_Re, d_r_Im = complex_axpy(d_a_Re[i], d_a_Im[i], d_v1_Re[i], d_v1_Im[i], d_r_Re, d_r_Im)
    gpu_took = time.time() - lastime

    cp.testing.assert_allclose(r_Re, d_r_Re.get(), rtol=rtol, atol=atol, err_msg='complex_axpy(): real part is not equal', verbose=True) 
    cp.testing.assert_allclose(r_Im, d_r_Im.get(), rtol=rtol, atol=atol, err_msg='complex_axpy(): imag part is not equal', verbose=True)
    print('cpu: {0}\ngpu: {1}'.format(cpu_took, gpu_took))

    r_Re, r_Im  = reset(size, dtype)
    d_r_Re, d_r_Im = device_reset(size, dtype)

    print("testing axpy() ({0} times) ...".format(nt))

    lastime = time.time()
    for i in range(0, nt):
        int_Re, int_Im = axpy(a_Re[i], v1_Re[i], v1_Im[i], v2_Re[i], v2_Im[i])
        r_Re += int_Re
        r_Im += int_Im
    cpu_took = time.time() - lastime

    lastime = time.time()
    for i in range(0, nt):
        d_int_Re, d_int_Im = axpy(d_a_Re[i], d_v1_Re[i], d_v1_Im[i], d_v2_Re[i], d_v2_Im[i])
        d_r_Re += d_int_Re
        d_r_Im += d_int_Im
    gpu_took = time.time() - lastime

    cp.testing.assert_allclose(r_Re, d_r_Re, rtol=rtol, atol=atol, err_msg='axpy(): real part is not equal', verbose=True) 
    cp.testing.assert_allclose(r_Im, d_r_Im, rtol=rtol, atol=atol, err_msg='axpy(): imag part is not equal', verbose=True)
    print('cpu: {0}\ngpu: {1}'.format(cpu_took, gpu_took))

    print("testing complex_norm_sq() ({0} times) ...".format(nt))

    r_Re, r_Im = reset(nt, cdtype)
    d_r_Re, d_r_Im = device_reset(nt, cdtype)

    lastime = time.time()
    for i in range(0, nt):
        r_Re[i] = complex_norm_sq(v1_Re[i], v1_Im[i], cdtype)
    cpu_took = time.time() - lastime

    lastime = time.time()
    for i in range(0, nt):
        d_r_Re[i] = complex_norm_sq(d_v1_Re[i], d_v1_Im[i], cdtype)
    gpu_took = time.time() - lastime

    cp.testing.assert_allclose(r_Re, d_r_Re, rtol=rtol, atol=atol, err_msg='complex_norm_sq(): norms are not equal', verbose=True)
    print('cpu: {0}\ngpu: {1}'.format(cpu_took, gpu_took))

    print("testing complex_scalar_prod() ({0} times) ...".format(nt))

    r_Re, r_Im  = reset(nt, cdtype)
    d_r_Re, d_r_Im  = device_reset(nt, cdtype)

    lastime = time.time()
    for i in range(0, nt):
        r_Re[i], r_Im[i] = complex_scalar_prod(v1_Re[i], v1_Im[i], v2_Re[i], v2_Im[i], cdtype)
    cpu_took = time.time() - lastime

    lastime = time.time()
    for i in range(0, nt):
        d_r_Re[i], d_r_Im[i] = complex_scalar_prod(d_v1_Re[i], d_v1_Im[i], d_v2_Re[i], d_v2_Im[i], cdtype)
    gpu_took = time.time() - lastime

    cp.testing.assert_allclose(r_Re, d_r_Re, rtol=rtol, atol=atol, err_msg='complex_scalar_prod(): real part is not equal', verbose=True) 
    cp.testing.assert_allclose(r_Im, d_r_Im, rtol=rtol, atol=atol, err_msg='complex_scalar_prod(): imag part is not equal', verbose=True)
    print('cpu: {0}\ngpu: {1}'.format(cpu_took, gpu_took))

    print("testing complex_matrix_vector_prod() ({0} times) ...".format(nt))

    r_Re, r_Im  = reset(size, dtype)
    d_r_Re, d_r_Im  = device_reset(size, dtype)

    lastime = time.time()
    for i in range(0, nt):
        int_Re, int_Im = complex_matrix_vector_prod(M_Re, M_Im, v1_Re[i], v1_Im[i])
        r_Re += int_Re
        r_Im += int_Im
    cpu_took = time.time() - lastime

    lastime = time.time()
    for i in range(0, nt):
        d_int_Re, d_int_Im = complex_matrix_vector_prod(d_M_Re, d_M_Im, d_v1_Re[i], d_v1_Im[i])
        d_r_Re += d_int_Re
        d_r_Im += d_int_Im
    gpu_took = time.time() - lastime

    cp.testing.assert_allclose(r_Re, d_r_Re, rtol=rtol, atol=atol, err_msg='complex_matrix_vector_prod(): real part is not equal', verbose=True) 
    cp.testing.assert_allclose(r_Im, d_r_Im, rtol=rtol, atol=atol, err_msg='complex_matrix_vector_prod(): imag part is not equal', verbose=True)
    print('cpu: {0}\ngpu: {1}'.format(cpu_took, gpu_took))

    print("testing complex_matrix_dagger_vector_prod() ({0} times) ...".format(nt))

    r_Re, r_Im  = reset(size, dtype)
    d_r_Re, d_r_Im  = device_reset(size, dtype)

    lastime = time.time()
    for i in range(0, nt):
        int_Re, int_Im = complex_matrix_dagger_vector_prod(M_Re, M_Im, v1_Re[i], v1_Im[i])
        r_Re += int_Re
        r_Im += int_Im
    cpu_took = time.time() - lastime

    lastime = time.time()
    for i in range(0, nt):
        d_int_Re, d_int_Im = complex_matrix_dagger_vector_prod(d_M_Re, d_M_Im, d_v1_Re[i], d_v1_Im[i])
        d_r_Re += d_int_Re
        d_r_Im += d_int_Im
    gpu_took = time.time() - lastime

    cp.testing.assert_allclose(r_Re, d_r_Re, rtol=rtol, atol=atol, err_msg='complex_matrix_dagger_vector_prod(): real part is not equal', verbose=True) 
    cp.testing.assert_allclose(r_Im, d_r_Im, rtol=rtol, atol=atol, err_msg='complex_matrix_dagger_vector_prod(): imag part is not equal', verbose=True)
    print('cpu: {0}\ngpu: {1}'.format(cpu_took, gpu_took))

