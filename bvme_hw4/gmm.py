import sys
import numpy as np
import scipy.sparse
from scipy.linalg import cholesky
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
from scipy.stats import norm


def fit_gmm(X, iters, init):
    """
    Fit a GMM to the datapoints

    Parameters:

    X : d x n array
        d-dimensional datapoints

    iters : scalar
        number of iterations for the EM algorithm

    init : scalar or dict
        If scalar, specify the number of components for the GMM. If dict, it is assumed
        the keys 'mu', 'sigma' and 'weights' hold the parameters of a GMM. These parameters
        are used as starting point for the EM algorithm.

    Returns:

    gmm : dict with keys 'mu', 'sigma' and 'weights
        The fitted GMM
    """
    if isinstance(init, int):
        model = initialize_model(X, init)
    elif isinstance(init, dict):
        if not init.has_key('mu') or not init.has_key('sigma') or not init.has_key('weights'):
            print ValueError('init must be a dict with keys mu, sigma and weights')
        model = init
    else:
        raise TypeError('init must be a scalar or a dict')

    d, n = np.atleast_2d(X).shape
    k = model['weights'].size # number of components
    alpha, mu, sig = model['weights'], model['mu'], model['sigma']
    # gausspdf(X, mu, sig)
    # plt.pause(0.1);raw_input()

    mu = mu.transpose(1, 0)
    sig = sig.transpose(2, 0, 1)

    for i in xrange(iters):
        # print 'manipulated'
        gausspdf(X, mu.transpose(1, 0), sig.transpose(1, 2, 0))
        # plt.pause(0.1); raw_input()
        gammas = np.zeros((n, k))
        sum_mu = np.zeros((k, d))
        for j, x in enumerate(X.T):
            diff = x - mu
            sig_diff = sig
            gam = alpha / (np.sqrt(2.0*np.pi)**d*np.linalg.det(sig_diff)**0.5) * np.exp(-0.5*np.einsum("ij,ijk,ik->i", diff, sig, diff))
            # print ' check gam ', gam.shape, alpha.shape
            gam_sum = np.sum(gam)
            gam = 1./n if gam_sum == 0. else gam/gam_sum
            gammas[j] = gam
            sum_mu += np.outer(gam, x)

        sum_gam = np.sum(gammas, axis=0)
        alpha = sum_gam / n

        mu = sum_mu / sum_gam.reshape((k, 1))

        sum_sig = np.zeros((k, d, d))
        for j, x in enumerate(X.T):
            xmm = x-mu
            sum_sig += gammas[j].reshape((k, 1, 1))*np.einsum("ij,ik->ijk", xmm, xmm)
        sig = sum_sig / sum_gam.reshape((k, 1, 1))
        print "i: {}".format(i)

    model['weights'] = alpha
    model['mu'] = mu.transpose(1, 0); print 'mu transpose', mu.transpose(1, 0)
    model['sigma'] = sig.transpose(1, 2, 0)

    return model


def gausspdf(X, mu, sigma):
    """
    Evaluate multivariate gauss.
    X are d x n input points, mu is the d-dimensional mean and
    sigma is the d x d covariance matrix.
    """
    mu_r, mu_g, mu_b = mu
    sig_r, sig_g, sig_b = np.array(map(lambda x: np.diag(x), sigma.transpose(2,0,1))).T

    h = np.arange(0,1,0.001)

    gauss_r = [1.0 / (np.sqrt(2.0*np.pi)*s) * np.exp(-(h-m)**2/(2.0*s**2)) for m, s in zip(mu_r, sig_r)]
    gauss_g = [1.0 / (np.sqrt(2.0*np.pi)*s) * np.exp(-(h-m)**2/(2.0*s**2)) for m, s in zip(mu_g, sig_g)]
    gauss_b = [1.0 / (np.sqrt(2.0*np.pi)*s) * np.exp(-(h-m)**2/(2.0*s**2)) for m, s in zip(mu_b, sig_b)]

    plt.figure(30); plt.clf()
    plt.title('gauss pdf for channel r')
    for g in gauss_r:
        plt.plot(h, g, "r-")
    plt.plot(h, np.sum(gauss_r, axis=0), "b-")

    plt.figure(31); plt.clf()
    plt.title('gauss pdf for channel g')
    for g in gauss_g:
        plt.plot(h, g, "r-")
    plt.plot(h, np.sum(gauss_g, axis=0), "b-")

    plt.figure(32); plt.clf()
    plt.title('gauss pdf for channel b')
    for g in gauss_b:
        plt.plot(h, g, "r-")
    plt.plot(h, np.sum(gauss_b, axis=0), "b-")
    plt.show()


def initialize_model(X, k):
    X = np.atleast_2d(X)
    d,n = X.shape
    model = dict()

    rand_int = np.round(np.random.rand(k)*X.shape[1]).astype('int')
    model['mu'] = X[:,rand_int]      # means are random samples from the data
    model['sigma'] = np.tile(np.atleast_3d(np.eye(d)*0.0025), (1,1,k)) # init sigma uniform
    model['weights'] = np.ones(k)/k # init weights uniform

    return model


class Image_Selector:
    def __init__(self, I):
        self.fig, self.ax = plt.subplots(1,1)
        self.ax.imshow(I, interpolation='none')

        self.RS = RectangleSelector(self.ax, self.line_select_callback,
                                    drawtype='box', useblit=True,
                                    button=[1, 3],  # don't use middle button
                                    minspanx=5, minspany=5,
                                    spancoords='pixels')

        self.fig.canvas.mpl_connect('button_release_event', self.on_button_release)
        self.fig.canvas.start_event_loop(timeout=-1)   # start blocking event loop

    def line_select_callback(self, eclick, erelease):
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        print '(%3.2f, %3.2f) --> (%3.2f, %3.2f)' % (x1, y1, x2, y2)
        self.selection_rect = np.round(np.array([[x1,y1],[x2,y2]]))
        sys.stdout.flush()

    def on_button_release(self, event):
        sys.stdout.flush()
        self.fig.canvas.stop_event_loop()    # break out of loop and continue program
