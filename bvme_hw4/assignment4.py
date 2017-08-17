
import numpy as np
import matplotlib.pyplot as plt
import scipy.misc

from gmm import fit_gmm, Image_Selector, gausspdf
from log_gmm import log_gmm
from rof import rof_pd

# I wanna save images, then true
bplot = False
# bplot = True
# plot_name = 'skier'
plot_name = 'todelete'

plt.ion()
plt.close('all')
############################################################################
########## pictures from the repository
# I = scipy.misc.imread('seestern.jpg') / 255.0
# I = scipy.misc.imread('child.jpg') / 255.0
# I = scipy.misc.imread('dolphin.jpg') / 255.0
# I = scipy.misc.imread('nemo.jpg') / 255.0
# I = scipy.misc.imread('romy.jpg') / 255.0
# I = scipy.misc.imread('tiger.jpg') / 255.0

############################################################################
########## own pictures
I = scipy.misc.imread('sphinx_small.jpg') / 255.0
# I = scipy.misc.imread('christmas_egg.jpg') / 255.0
# I = scipy.misc.imread('skier_small.jpg') / 255.0
# I = scipy.misc.imread('color.jpg') / 255.0

# plt.imshow(I[60: 130, 80: 120], cmap='gray')
# plt.show()
# I = I[60:180, 80:180]

############################################################################
########## RGB Histograms
# if bplot: # set True if you wanna plot rgb
if bplot: # set True if you wanna plot rgb
    Ir, Ig, Ib = I[:, :, 0], I[:, :, 1], I[:, :, 2]
    I8bit = (I*255.).astype(np.uint8)
    # plt.imshow(I8bit)
    # print("I8bit.shape = {}".format(I8bit.shape))
    # print("type(I8bit) = {}".format(type(I8bit)))
    red = plt.figure(300);
    plt.title("hist of r channel"); plt.hist(Ir.ravel(), bins=256, normed=True, color=["r"])
    green = plt.figure(301)
    plt.title("hist of g channel"); plt.hist(Ig.ravel(), bins=256, normed=True, color=["g"])
    blue = plt.figure(302)
    plt.title("hist of b channel"); plt.hist(Ib.ravel(), bins=256, normed=True, color=["b"])
    red.savefig('report/histograms/' + plot_name + '_r.png')
    green.savefig('report/histograms/' + plot_name + '_g.png')
    blue.savefig('report/histograms/' + plot_name + '_b.png')

###########################################################################################
im_selector = Image_Selector(I)
plt.show()
initial_rect = im_selector.selection_rect
print initial_rect

mask = np.zeros(I[:, :, 0].shape, dtype='bool')
mask[initial_rect[0, 1]:initial_rect[1, 1], initial_rect[0, 0]:initial_rect[1, 0]] = 1

### Parameters
###
K = 4 # gmm components
gmm_iters = 20 # gmm iterations
rof_iters = 300 # rof iterations
rof_check = 50 # print info every check iterations
N = 5    # number of times to iterate gmm fitting and segmentation

Ir,Ig,Ib = I[:,:,0], I[:,:,1], I[:,:,2]
x = np.vstack((Ir.ravel(), Ig.ravel(), Ib.ravel()))   # all pixels
for i in range(N):
    # get foreground and background pixels
    x_fg = np.vstack((Ir[mask],Ig[mask],Ib[mask]))
    x_bg = np.vstack((Ir[~mask],Ig[~mask],Ib[~mask]))

    # visualize segmentation
    Jr, Jg, Jb = Ir.copy(), Ig.copy(), Ib.copy()
    Jr[~mask] = 1; Jg[~mask] = 1; Jb[~mask] = 1
    plt.figure(20); plt.clf()
    plt.imshow(np.dstack((Jr,Jg,Jb)), interpolation='none'); plt.title('segmented image (iter '+str(i)+')')
    if bplot:
        plt.imsave(plot_name + '_seg.png', np.dstack((Jr, Jg, Jb)))
    plt.pause(0.001)

    if i==0:
        model_fg = fit_gmm(x_fg, gmm_iters, K)   # first iteration: init gmm
        model_bg = fit_gmm(x_bg, gmm_iters, K)
    else:
        model_fg = fit_gmm(x_fg, gmm_iters, model_fg)   # refine gmm
        model_bg = fit_gmm(x_bg, gmm_iters, model_bg)

    # print 'mu fg', model_fg['mu'].shape
    # print 'sigma fg', model_fg['sigma'].shape

    # gausspdf(x_fg, model_fg['mu'], model_fg['sigma'])

    w_fg = np.reshape(log_gmm(x, model_fg), Ir.shape)
    w_bg = np.reshape(log_gmm(x, model_bg), Ir.shape)
    w = w_bg-w_fg
    if bplot:
        fg = plt.figure(22);plt.clf();plt.imshow(w_fg,interpolation='none',cmap='gray');plt.colorbar();plt.title('FG')
        bg = plt.figure(24);plt.clf();plt.imshow(w_bg,interpolation='none',cmap='gray');plt.colorbar();plt.title('BG')
        weights = plt.figure(26);plt.clf();plt.imshow(w,interpolation='none',cmap='gray');plt.colorbar();plt.title('W')
        fg.savefig('report/weights/' + plot_name + '_fg.png')
        bg.savefig('report/weights/' + plot_name + '_bg.png')
        weights.savefig('report/weights/' + plot_name + '_weights.png')

    u = rof_pd(w, 0.06, rof_iters, rof_check)
    mask = u<0 # threhold to update mask

Jr,Jg,Jb = Ir.copy(), Ig.copy(), Ib.copy()
Jr[~mask] = 1; Jg[~mask] = 1; Jb[~mask] = 1
plt.figure(); plt.clf()
plt.imshow(np.dstack((Jr,Jg,Jb)), interpolation='none'); plt.title('final segmentation')
plt.show()