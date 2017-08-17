
import numpy as np
import scipy.sparse

def rgb2gray(I):
    return np.sum(I*np.array([0.299, 0.587, 0.114]), axis=2)


def make_derivatives_2D(shape):
    r"""
    Sparse matrix approximation of gradient operator on image plane by finite (forward) differences.
    
    Parameters
    ----------
    
    shape:
        image size (tuple of ints)
    
    Returns: 
    
    :Kx,Ky: sparse matrices for gradient in x- and y-direction
    """
    M = shape[0]
    N = shape[1]
    
    x,y = np.meshgrid(np.arange(0,N), np.arange(0,M))
    linIdx = np.ravel_multi_index((y,x), x.shape)    # linIdx[y,x] = linear index of (x,y) in an array of size MxN

    i = np.vstack( (np.reshape(linIdx[:,:-1], (-1,1) ), np.reshape(linIdx[:,:-1], (-1,1) )) )  # row indices
    j = np.vstack( (np.reshape(linIdx[:,:-1], (-1,1) ), np.reshape(linIdx[:,1:], (-1,1) )) )   # column indices
    v = np.vstack( (np.ones( (M*(N-1),1) )*-1, np.ones( (M*(N-1),1) )) )                       # values
    Kx = scipy.sparse.coo_matrix((v.flatten(),(i.flatten(),j.flatten())), shape=(M*N,M*N))

    i = np.vstack( (np.reshape(linIdx[:-1,:], (-1,1) ), np.reshape(linIdx[:-1,:], (-1,1) )) )
    j = np.vstack( (np.reshape(linIdx[:-1,:], (-1,1) ), np.reshape(linIdx[1:,:], (-1,1) )) )
    v = np.vstack( (np.ones( ((M-1)*N,1) )*-1, np.ones( ((M-1)*N,1) )) )
    Ky = scipy.sparse.coo_matrix((v.flatten(),(i.flatten(),j.flatten())), shape=(M*N,M*N))

    return Kx.tocsr(),Ky.tocsr()
    
def make_derivatives_hp_sym_2D(shape):
    r"""
    Sparse matrix approximation of gradient operator on image plane.
    Derivatives are computed using half-point symmetric finite differences.
    
    Parameters
    ----------
    
    shape:
        image size (tuple of ints)
    
    Returns: 
    
    :Kx,Ky: sparse matrices for gradient in x- and y-direction
    """
    M = shape[0]
    N = shape[1]
    
    x,y = np.meshgrid(np.arange(0,N), np.arange(0,M))
    linIdx = np.ravel_multi_index((y,x), x.shape)    # linIdx[y,x] = linear index of (x,y) in an array of size MxN

    i = np.vstack( (np.reshape(linIdx[:-1,:-1], (-1,1) ), np.reshape(linIdx[:-1,:-1], (-1,1) )) )  # row indices
    # print 'row indices ', i

    j = np.vstack( (np.reshape(linIdx[:-1,:-1], (-1,1) ), np.reshape(linIdx[:-1,1:], (-1,1) )) )   # column indices
    # print 'col indices ', j

    i = np.vstack( (i, np.reshape(linIdx[:-1,:-1], (-1,1) ), np.reshape(linIdx[:-1,:-1], (-1,1) )) )
    j = np.vstack( (j, np.reshape(linIdx[1:,:-1], (-1,1) ), np.reshape(linIdx[1:,:-1], (-1,1) )) )

    v = np.vstack( (np.ones( ((M-1)*(N-1),1) )*-0.5, np.ones( ((M-1)*(N-1),1) )*0.5) )             # values


    v = np.vstack( (v, np.ones( ((M-1)*(N-1),1) )*-0.5, np.ones( ((M-1)*(N-1),1) )*0.5) )


    Kx = scipy.sparse.coo_matrix((v.flatten(),(i.flatten(),j.flatten())), shape=(M*N,M*N))

    i = np.vstack( (np.reshape(linIdx[:-1,:-1], (-1,1) ), np.reshape(linIdx[:-1,:-1], (-1,1) )) )  # row indices
    j = np.vstack( (np.reshape(linIdx[:-1,:-1], (-1,1) ), np.reshape(linIdx[1:,:-1], (-1,1) )) ) # column indices
    i = np.vstack( (i, np.reshape(linIdx[:-1,:-1], (-1,1) ), np.reshape(linIdx[:-1,:-1], (-1,1) )) )
    j = np.vstack( (j, np.reshape(linIdx[:-1,1:], (-1,1) ), np.reshape(linIdx[1:,1:], (-1,1) )) )
    v = np.vstack( (np.ones( ((M-1)*(N-1),1) )*-0.5, np.ones( ((M-1)*(N-1),1) )*0.5) )
    v = np.vstack( (v, np.ones( ((M-1)*(N-1),1) )*-0.5, np.ones( ((M-1)*(N-1),1) )*0.5) )
    Ky = scipy.sparse.coo_matrix((v.flatten(),(i.flatten(),j.flatten())), shape=(M*N,M*N))

    return Kx.tocsc(),Ky.tocsc()
