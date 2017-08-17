import os, struct
import numpy as np
import matplotlib.pyplot as plt
import array

import sys
import select

from scipy import sparse
import pdb


def load_mnist(dataset="training", digits=np.arange(10), path="."):
    """
    Loads MNIST files into numpy arrays

    Adapted from: http://abel.ee.ucla.edu/cvxopt/_downloads/mnist.py
    """
    if dataset == "training":
        fname_img = os.path.join(path, 'train-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels-idx1-ubyte')
    elif dataset == "testing":
        fname_img = os.path.join(path, 't10k-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels-idx1-ubyte')
    else:
        raise ValueError("dataset must be 'testing' or 'training'")

    flbl = open(fname_lbl, 'rb')
    magic_nr, size = struct.unpack(">II", flbl.read(8))
    lbl = array.array("b", flbl.read())
    flbl.close()

    fimg = open(fname_img, 'rb')
    magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
    img = array.array("B", fimg.read())
    fimg.close()

    ind = [ k for k in range(size) if lbl[k] in digits ]
    N = len(ind)

#    images = np.zeros((N, rows, cols), dtype=np.uint8)
#    labels = np.zeros((N, 1), dtype=np.int8)
#    for i in range(len(ind)):
#        images[i] = np.array(img[ ind[i]*rows*cols : (ind[i]+1)*rows*cols ]).reshape((rows, cols))
#        labels[i] = lbl[ind[i]]
    
    images = np.zeros((rows * cols, N))
    labels = np.zeros(N, dtype=np.int8)
    if np.all(digits==np.arange(10)):
        images = np.reshape(img, (rows*cols,-1), order='F').astype('uint8')
        labels = np.array(lbl).astype('int8')
    else:
        for i in range(len(ind)):
            images[:,i] = img[ ind[i]*rows*cols : (ind[i]+1)*rows*cols ]
            labels[i] = lbl[ind[i]]
    
    return images, labels

def visualize_dictionary(D, num, M, N, count_img, fig=None, labels=None, gt_labels=None, shape=None):
    """
    Visualize num atoms of the dictionary D, where atoms are columns of D.
    The atoms are assumed to be vectorized images of size M x N.
    
    If fig is given, draw into that figure
    
    If labels or gt_labels are given, print the number in these vectors along with the image.
    Labels will have a red background, gt_labels a green one.
    
    If shape is given, the image will consist of shape[0] x shape[1] patches of size M x N,
    otherwise the size is computed automatically.
    """
    
    if shape:
        sz = shape
    else:
        sz = (np.ceil(np.sqrt(num)).astype('int'),)*2
        
    if sz[0]*sz[1] < num:
        raise RuntimeError('Not enough space to display '+str(num)+' patches with a shape of '+str(sz[0])+' x '+str(sz[1]))
    vis = np.zeros((sz[0]*M,sz[1]*N))        
    for row in range(sz[0]):
        for col in range(sz[1]):
            idx = row*sz[1]+col
            if idx>=num: break
            d = np.reshape(D[:,idx],(M,N)); d = d - np.min(d); d = d / np.max(d)
            vis[row*M:(row+1)*M, col*N:(col+1)*N] = d

    if fig:
        plt.figure(fig); plt.clf(); plt.imshow(vis, interpolation='none', cmap='gray')
    else:
        plt.figure(); plt.imshow(vis, interpolation='none', cmap='gray')
        
    
    if labels is not None:
        for row in range(sz[0]):
            for col in range(sz[1]):
                plt.text(col*N, row*M+4, str(labels[row*sz[1]+col]), color='white', bbox=dict(facecolor='red', alpha=0.5))
    if gt_labels is not None:
        for row in range(sz[0]):
            for col in range(sz[1]):
                plt.text(col*N+8, row*M+4, str(gt_labels[row*sz[1]+col]), color='white', bbox=dict(facecolor='green', alpha=0.5))
    plt.savefig(str(count_img) + '.png')
    plt.pause(0.0001)

def visualize_reconstruction(D, C, M, N, num, fig=None, labels=None, gt_labels=None, shape=None):
    
    R = np.dot(D,C[:,0:num])
    visualize_dictionary(R, R.shape[1], M, N, 200, fig, labels, gt_labels, shape)

def prox(x_dach, x, tau, lam):
    """
    http://www.seas.ucla.edu/~vandenbe/236C/lectures/proxgrad.pdf, p6
    http://www.onmyphd.com/?p=proximal.operator
    :param x_dach:
    :param tau:
    :return:
    """
    first_row = x_dach[0].copy()
    x_dach[np.where(x_dach >= tau)] -= lam*tau
    x_dach[np.where(np.logical_and(x_dach >= -tau, x_dach <= tau))] = 0.0
    x_dach[np.where(x_dach <= -tau)] += lam*tau
    x_dach[0] = first_row

    return x_dach

def sparse_coding(imgs, M, N, D, C, Lambda, maxiter, check=10):
    """
    Compute a sparse coding: :math:`min_{D,C} \|DC-B\|_2^2+\lambda\|C\|_1`.
    B is the data given in imgs.
    
    Parameters:
    -----------
    imgs : array (MN x n_samples)
        Data to compute sparse coding for. Every column in the array is a vectorized image of
        size M x N.
    
    M,N : image size
    
    D : array (MN x n_features)
        Dictionary with n_features atoms
    
    C : array (n_features x n_samples)
        Representation coefficients
    
    Lambda : scalar
        Regularization parameter
    
    maxiter : scalar
        number of iterations
    
    check : scalar (default=10)
        iteration interval for debug output
        
    Returns:
    --------
    (D,C) : tuple
        Learned dictionary and representation coefficients
    
    """

    def f(D, C, B, lam):
        tmp = D.dot(C)-B
        return np.linalg.norm(tmp)**2 + lam*np.sum(np.vectorize(lambda x: np.abs(x))(C[:,1:]))

    print("D, C = {}".format(D.shape))
    n_features = D.shape[1]
    energy = []

    B = imgs.copy()
    # maxiter = 100
    D_prev = D.copy()
    C_prev = C.copy()
    for k in xrange(1, maxiter):
        tau = 1.0/np.linalg.norm(D.T.dot(D))

        beta = (k-1) / (k+2)
        C_k = C+beta*(C-C_prev)
        D_k = D+beta*(D-D_prev)

        nabla_C = D.T.dot(D.dot(C_k)-B)
        C_k_1 = C_k - tau*nabla_C
        C_k_1 = prox(C_k_1, C, tau, Lambda)
        sig = 1.0/np.linalg.norm(C_k_1.dot(C_k_1.T))

        nabla_D = (D_k.dot(C_k_1)-B).dot(C_k_1.T)
        D_k_1 = D_k-sig*nabla_D

        D_k_1 /= np.max(np.vstack((np.linalg.norm(D_k_1, axis=0).reshape((1,D_k_1.shape[1])), np.ones((1,D_k_1.shape[1])))), axis=0)

        D_prev = D.copy()
        C_prev = C.copy()
        D = D_k_1.copy()
        C = C_k_1.copy()

        #tmp_err = f(D, C, B, Lambda)
        #print "error: {}".format(tmp_err)
        #energy.append(tmp_err)

        # User break the loop by pressing ENTER!
        if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
            break

        absolute_val = np.linalg.norm(C, axis=1)
        tmp = 0.5*np.sum((D.dot(C)-B)**2.0)+Lambda*(np.sum(absolute_val)-absolute_val[0])
        print "k={},\t  error={}".format(k, tmp)
        visualize_reconstruction(D,C,M,N, n_features, 1)

        # pdb.set_trace()
    # print '--------------------------------------------'
    # print ', '.join(str(p) for p in energy)

    return D, C

def sparse_coding_test(imgs, M, N, D, C, Lambda, maxiter, check=10):
    """
    Compute a sparse coding: :math:`min_{D,C} \|DC-B\|_2^2+\lambda\|C\|_1`.
    B is the data given in imgs.

    Parameters:
    -----------
    imgs : array (MN x n_samples)
        Data to compute sparse coding for. Every column in the array is a vectorized image of
        size M x N.

    M,N : image size

    D : array (MN x n_features)
        Dictionary with n_features atoms

    C : array (n_features x n_samples)
        Representation coefficients

    Lambda : scalar
        Regularization parameter

    maxiter : scalar
        number of iterations

    check : scalar (default=10)
        iteration interval for debug output

    Returns:
    --------
    (D,C) : tuple
        Learned dictionary and representation coefficients

    """

    def f(D, C, B, lam):
        tmp = D.dot(C)-B
        return np.linalg.norm(tmp)**2 + lam*np.sum(np.vectorize(lambda x: np.abs(x))(C[:,1:]))

    print("D, C = {}".format(D.shape))
    n_features = D.shape[1]
    energy = []

    B = imgs.copy()
    # maxiter = 100
    D_prev = D.copy()
    C_prev = C.copy()
    for k in xrange(1, maxiter):
        tau = 1.0/np.linalg.norm(D.T.dot(D))

        beta = (k-1) / (k+2)
        C_k = C+beta*(C-C_prev)
        #D_k = D+beta*(D-D_prev)

        nabla_C = D.T.dot(D.dot(C_k)-B)
        C_k_1 = C_k - tau*nabla_C
        C_k_1 = prox(C_k_1, C, tau, Lambda)
        sig = 1.0/np.linalg.norm(C_k_1.dot(C_k_1.T))

        #nabla_D = (D_k.dot(C_k_1)-B).dot(C_k_1.T)
        #D_k_1 = D_k-sig*nabla_D

        #D_k_1 /= np.max(np.vstack((np.linalg.norm(D_k_1, axis=0).reshape((1,D_k_1.shape[1])), np.ones((1,D_k_1.shape[1])))), axis=0)

        #D_prev = D.copy()
        C_prev = C.copy()
        #D = D_k_1.copy()
        C = C_k_1.copy()

        #tmp_err = f(D, C, B, Lambda)
        #print "error: {}".format(tmp_err)
        #energy.append(tmp_err)

        # User break the loop by pressing ENTER!
        if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
            break

        absolute_val = np.linalg.norm(C, axis=1)
        tmp = 0.5*np.sum((D.dot(C)-B)**2.0)+Lambda*(np.sum(absolute_val)-absolute_val[0])
        print "k={},\t  error={}".format(k, tmp)

        # pdb.set_trace()
    # print '--------------------------------------------'
    # print ', '.join(str(p) for p in energy)

    return C