#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import scipy.misc
import scipy.sparse

import os
import scipy.ndimage
from skimage import filters
import skimage
from scipy.sparse import csr_matrix

from coherence import make_derivatives_2D, make_derivatives_hp_sym_2D, rgb2gray


# Kx, Ky = make_derivatives_hp_sym_2D((3,3))
# print ' kx ', kx, ' ', 'ky', ky

plt.close('all')
I = rgb2gray(scipy.misc.imread('gandalf_small.jpg')) / 255.0
# I = rgb2gray(scipy.misc.imread('gandalf_big.jpg')) / 255.0
# I = I[100:200, 300:400]
# I = I[60: 130, 80: 120]
# I = I[108: 110, 250: 252]

I = np.minimum(1, np.maximum(0, I + np.random.randn(*I.shape) * 0.001))
plt.figure();
plt.imshow(I, cmap='gray', vmin=0, vmax=1, interpolation='none')
# plt.figure(); plt.imshow(I[100:200, 300:400], cmap='gray', vmin=0, vmax=1, interpolation='none')
# plt.show()

# Kx,Ky = make_derivatives_2D(I.shape)
Kx,Ky = make_derivatives_hp_sym_2D(I.shape)
nabla = scipy.sparse.vstack([Kx,Ky]).tocsc()


identity = scipy.sparse.eye(np.prod(I.shape), format='csc')

alpha = 0.00005
gamma = 0.00001

end_time = 100
tau = 5
sigma_grad = 0.7
sigma_tensor = 1.5
u = I.copy()

def g(mu_1, mu_2):
    tmp = np.linalg.norm(mu_1 - mu_2)
    return np.exp(-(tmp**2) / (2.0 * (gamma**2)))

def lamb_2(vals):
    return alpha + (1.0 - alpha) * (1.0 - g(vals[0], vals[1]))

def reorderEigenVals(vals, vecs):
    order_val = vals.copy()
    order_vec = vecs.copy()

    if(vals[0] <= vals[1]):
        order_val[0] = vals[1]
        order_val[1] = vals[0]
        order_vec[:, 0] = vecs[:, 1]
        order_vec[:, 1] = vecs[:, 0]

    return order_val, order_vec

def calc_stensor(u):
    u_quer = scipy.ndimage.filters.gaussian_filter(u, sigma_grad)

    print type(u_quer), u_quer.shape

    u_quer_flatten = u_quer.flatten()
    u_quer_x = (Kx * u_quer_flatten).reshape(I.shape[:2])
    u_quer_y = (Ky * u_quer_flatten).reshape(I.shape[:2])

    # structure tensor
    u_quer_x_y = u_quer_x*u_quer_y
    s_tensor_x_y = filters.gaussian(u_quer_x_y, sigma_tensor).flatten()
    s_tensor_x = filters.gaussian(u_quer_x**2, sigma_tensor).flatten()
    s_tensor_y = filters.gaussian(u_quer_y**2, sigma_tensor).flatten()

    stensor =  np.array([[s_tensor_x, s_tensor_x_y], [s_tensor_x_y, s_tensor_y]])

    return stensor

def calc_d(stensor):
    Dtensor = np.zeros((2, 2, stensor.shape[2]))

    for i in xrange(stensor.shape[2]):
        vals, vecs = np.linalg.eig(stensor[: ,:, i])

        # first EVal must be bigger. if not so, swap EVec
        vals, vecs = reorderEigenVals(vals, vecs)

        dtensor = vecs
        dtensor = np.dot(dtensor, np.diag((alpha, lamb_2(vals))))
        Dtensor[:, :, i] = np.dot(dtensor, vecs.T)

    d_00 = scipy.sparse.diags(Dtensor[0,0])
    d_01 = scipy.sparse.diags(Dtensor[0,1])
    d_10 = scipy.sparse.diags(Dtensor[1,0])
    d_11 = scipy.sparse.diags(Dtensor[1,1])

    d_tmp1 = scipy.sparse.hstack([d_00, d_01])
    d_tmp2 = scipy.sparse.hstack([d_10, d_11])

    return scipy.sparse.vstack([d_tmp1, d_tmp2], format='csc')

print '-----------------------------------------------------------------'
u_minus_1 = I.copy()
for t in np.arange(0, end_time, tau):
    D = calc_d(calc_stensor(u_minus_1))
    A = identity + tau * nabla.T.dot(D).dot(nabla)
    u = scipy.sparse.linalg.spsolve(A, u_minus_1.flatten()).copy()
    u_minus_1 = np.array(u.copy())

    plt.figure(10); plt.clf()
    plt.imshow(u.reshape((I.shape[:2])), cmap='gray', vmin=0, vmax=1)
    plt.title('t=' + str(t))
    plt.pause(0.01)
    print 'time', t

plt.figure();
plt.imshow(u.reshape((I.shape[:2])), cmap='gray', vmin=0, vmax=1)
plt.title('result')
# plt.savefig('report/' + 'make_derivatives_2D_hyp_big' + '.png')
# plt.savefig('report/' + 'make_derivatives_2D_big' + '.png')
plt.savefig('report/' + 'make_derivatives_2D_' + str(alpha) + '_' + str(gamma) + '.png')
plt.show()
