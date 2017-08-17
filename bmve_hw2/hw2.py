#! /usr/bin/python2.7
import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.misc
import scipy.ndimage
from skimage.draw import polygon
from skimage import measure

import cProfile
import re

plt.close('all')
I = scipy.misc.imread('city.png') / 255.0
# I = scipy.misc.imread('report/neg/stone1.jpg') / 255.0
# I = scipy.misc.imread('report/neg/stone2.jpg') / 255.0
# I = scipy.misc.imread('report/neg/neg.png') / 255.0

sigma_noise = 0.1 # noise variance
I_noisy = I + np.random.randn(*I.shape)*sigma_noise
I_noisy = np.minimum(1, np.maximum(0, I_noisy))   # clamp to [0,1]
# scipy.misc.imsave('report/pos/pos_noisy.png', I_noisy)
# scipy.misc.imsave('report/neg/pos_noisy.png', I_noisy)

# a rectangle specified by upper left and lower right corner
# first row x-coordinates, second row y-coordinates
# run algorithm only inside rectangle to reduce runtime. When finished implementing,
# run algorithm on whole image
# rectangle = np.array([[400, 460], [240, 280]])
# rectangle = np.array([[390, 480], [240, 300]])
# rectangle = np.array([[300, 500], [200, 400]])
rectangle = np.array([[0, I.shape[1]], [0, I.shape[0]]]) # whole image

### x strip 10 px left
# rectangle = np.array([[0, 10], [0, I.shape[0]]]) # whole image
### x strip 10 px right
# rectangle = np.array([[I.shape[1]-10, I.shape[1]], [0, I.shape[0]]]) # whole image

### y strip 10 px up
# rectangle = np.array([[0, I.shape[1]], [0, 10]]) # whole image
### y strip 10 px down
# rectangle = np.array([[0, I.shape[1]], [I.shape[0] - 10, I.shape[0]]]) # whole image

# algorithm parameters
patch_radius = 3;         # 7x7 patches, std val = 3
search_radius = 15;       # 31x31 search range, std val = 15
sigma_weight = 0.05;      # std val = 0.05

def rgb2gray(I):
    return np.sum(I*np.array([0.299, 0.587, 0.114]), axis=2)

def get_patch(I, rect):
    return I[rect[1,0]:rect[1,1], rect[0,0]:rect[0,1]]

def show_gauss(sigma, kernel_size):
    '''
    copied from tut_2
    :param sigma:
    :param kernel_size:
    :return: h2
    '''
    ks2 = np.float(kernel_size / 2)

    x,y = np.meshgrid(np.arange(-ks2, ks2+1), np.arange(-ks2, ks2+1))
    h2 = np.exp( (-(x**2 + y**2) / (2*sigma**2)) )     # compute gaussian
    h2 = h2 / np.sum(h2)                               # normalize sum(coefficients)=1

    return h2

def show_psnr(F_noisy, G_filtered):
    '''
    :param F_noisy:
    :param G_filtered:
    :return: psnr
    '''
    difference = np.square(F_noisy - G_filtered)
    rows, cols = difference.shape[:2]
    mse = np.sum(np.sum(difference, axis=0), axis=0) / (np.float(rows * cols))
    # print 'mse', mse
    if not np.alltrue(mse > 0.0):
        return 0

    psnr = 10 * np.log10(np.mean(np.max(np.max(F_noisy, axis=0), axis=0) / mse))

    return psnr

I_filtered = np.zeros_like(I)
I_noisy = np.pad(I_noisy, ((patch_radius,)*2, (patch_radius,)*2, (0,0)), 'edge')
If = np.zeros_like(I_noisy)

# plt.figure(); plt.imshow(If, interpolation='none'); plt.title('asfdsfa')
# plt.figure(); plt.imshow(I_filtered, interpolation='none'); plt.title('aaasfdsfa')
# plt.figure(); plt.imshow(I_noisy, interpolation='none'); plt.title('asfdaaaasfa')
plt.show()

I_noisy_gray = rgb2gray(I_noisy)

######## Gauss Filter ########
print ' start gauss '
sigm_steps = [sig_s * 0.05 for sig_s in xrange(1, 25)]
patch_steps = [pat_s for pat_s in xrange(1, 15)]

list_psnr = list()
for sig in sigm_steps:
    for pat in patch_steps:
        h2 = show_gauss(sig, pat)
        for col in xrange(3):
            If[:,:,col] = scipy.ndimage.convolve(I_noisy[:,:,col], h2) # filter image
        psnr = show_psnr(I, If[patch_radius:-patch_radius, patch_radius:-patch_radius])
        list_psnr.append((sig, pat, psnr))
        # plt.figure(); plt.imshow(If, interpolation='none'); plt.title('gauss')
        # scipy.misc.imsave('report/gauss_imag/' + 's_' + str(sig) + '_rad_' + str(pat) + '.png', If)

res = max(list_psnr, key=lambda item: item[2])
print '(sig, patch, psnr) ', res

# ######## PSNR of NLM  ########
nlm_filtered = scipy.misc.imread('report/nlm_imag/nlm_res.png') / 255.0
orig = scipy.misc.imread('report/city.png') / 255.0
psnr = show_psnr(orig, nlm_filtered)
print " psnr ", psnr

######## NLM ########
rect = np.array([[280, 320], [150, 180]])
patch = get_patch(I, rect)

plt.figure(); plt.imshow(patch, interpolation='none'); plt.title('patch')

def weight(p_pixel_x, p_pixel_y, q_pixel_x, q_pixel_y):
    t = patch_radius
    s = np.float(np.square(2*t + 1))
    p_patch = get_patch(I_noisy_gray, np.array([[p_pixel_x - t, p_pixel_x + t + 1], [p_pixel_y - t, p_pixel_y + t + 1]]))
    q_patch = get_patch(I_noisy_gray, np.array([[q_pixel_x - t, q_pixel_x + t + 1], [q_pixel_y - t, q_pixel_y + t + 1]]))

    sum = np.sum((p_patch - q_patch)**2)

    return np.exp(-((sum / s) / np.square(sigma_weight))) # sigma_weight = h

max_weight = np.array([-1.0, -1.0, -1.0])
nlm_pixel = np.zeros(3) # the denoised pixel value

for y in range(rectangle[1,0], rectangle[1,1]):
    print y+1, 'of', rectangle[1,1]
    for x in range(rectangle[0,0], rectangle[0,1]):
        cx = x+patch_radius # center pixel in padded image
        cy = y+patch_radius

        list_w_pq = list()
        sum_weight = np.zeros(3)
        sum_w_pq = 0.
        sum_w_pq_pixel = np.zeros(3)

        for qy in range(max(cy - search_radius, patch_radius), min(rectangle[1, 1] - patch_radius + 1, cy + search_radius + 1)):
            for qx in range(max(cx - search_radius, patch_radius), min(rectangle[0, 1] - patch_radius + 1, cx + search_radius + 1)):
                if qy == cy or qx == cx:
                    continue

                w_pq = weight(cx, cy, qx, qy)
                sum_w_pq += w_pq
                sum_w_pq_pixel = sum_w_pq_pixel + (w_pq * I_noisy[qy,qx])
                list_w_pq.append(w_pq.tolist())

        max_weight = np.max(list_w_pq)
        I_filtered[y, x] = ((max_weight * I_noisy[cy, cx]) + sum_w_pq_pixel) / (max_weight + sum_w_pq)

        del list_w_pq[:]

I_noisy = I_noisy[patch_radius:-patch_radius, patch_radius:-patch_radius, :]
plt.figure(); plt.imshow(I, interpolation='none'); plt.title('original')
plt.figure(); plt.imshow(I_noisy, interpolation='none'); plt.title('noisy')
plt.figure(); plt.imshow(I_filtered, interpolation='none'); plt.title('result non local means')
# scipy.misc.imsave('report/nlm_imag/neg' + 's_' + str(sigma_weight) + '_rad_' + str(patch_radius) + '_search_' + str(search_radius) + '.png', I_filtered)
# scipy.misc.imsave('report/nlm_imag/neg_noisy' + 's_' + str(sigma_weight) + '_rad_' + str(patch_radius) + '_search_' + str(search_radius) + '.png', I_noisy)

# scipy.misc.imsave('report/neg/neg' + 's_' + str(sigma_weight) + '_rad_' + str(patch_radius) + '_search_' + str(search_radius) + '.png', I_filtered)
# scipy.misc.imsave('report/neg/neg_noisy' + 's_' + str(sigma_weight) + '_rad_' + str(patch_radius) + '_search_' + str(search_radius) + '.png', I_noisy)

# scipy.misc.imsave('report/neg/stone1' + 's_' + str(sigma_weight) + '_rad_' + str(patch_radius) + '_search_' + str(search_radius) + '.png', I_filtered)
# scipy.misc.imsave('report/neg/stone1_noisy' + 's_' + str(sigma_weight) + '_rad_' + str(patch_radius) + '_search_' + str(search_radius) + '.png', I_noisy)

# scipy.misc.imsave('report/neg/stone2' + 's_' + str(sigma_weight) + '_rad_' + str(patch_radius) + '_search_' + str(search_radius) + '.png', I_filtered)
# scipy.misc.imsave('report/neg/stone2_noisy' + 's_' + str(sigma_weight) + '_rad_' + str(patch_radius) + '_search_' + str(search_radius) + '.png', I_noisy)


plt.show()
