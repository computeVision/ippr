import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

import sparsity

count_img = 0

#%% data loading

M = 28      # mnist images size
N = 28
num_samples = 20000     # if negative, use all samples

plt.close('all')

imgs = np.load('mnist_training_images.npy') / 255.0
labels = np.load('mnist_training_labels.npy')

if num_samples > 0:
    imgs = imgs[:,0:num_samples]
    labels = labels[0:num_samples]
else:
    num_samples = labels.size


#%% training

n_features = 512               # number of dictionary atoms
Lambda = 0.1
maxiter = 1000

D = np.random.randn(M*N, n_features)    # dictionary
C = np.zeros((n_features, num_samples))     # representation coefficients
#D = np.load('trained_dict_32.npy'); C = np.load('trained_coeff_32.npy'); n_features = C.shape[0]

D, C = sparsity.sparse_coding(imgs, M, N, D, C, Lambda, maxiter)

#np.save('trained_dict_'+str(n_features)+'.npy', D); np.save('trained_coeff_'+str(n_features)+'.npy', C)

sparsity.visualize_dictionary(imgs, num_samples, M, N, 0)
sparsity.visualize_dictionary(D, n_features, M, N, 1)

sparsity.visualize_reconstruction(D, C, M, N, 2, num_samples)

#%% classification

if num_samples > 0:
    C = C[:,0:num_samples]

classifier_sparse_coding = svm.LinearSVC(loss='hinge', verbose=10, max_iter=2000)
classifier_orig = svm.LinearSVC(loss='hinge', verbose=10, max_iter=2000)

# C_aug = C                 # augmented features
C_aug = np.vstack((C, C * C))
classifier_sparse_coding.fit(C_aug.T, labels)    # train
classifier_orig.fit(imgs.T, labels)

test_imgs = np.load('mnist_testing_images.npy') / 255.0
test_labels = np.load('mnist_testing_labels.npy')

# TODO: compute representation coefficients of test data
C_test = np.zeros((n_features, test_labels.size))
D_test = np.random.randn(M*N, n_features) # dictionary
C_test = sparsity.sparse_coding_test(test_imgs, M, N, D, C_test, Lambda, maxiter)
C_test_aug = np.vstack((C_test, C_test * C_test))

result_sparse = classifier_sparse_coding.predict(C_test_aug.T)
result_orig = classifier_orig.predict(test_imgs.T)
error_sparse = np.where(result_sparse!=test_labels)[0].size / float(test_labels.size) * 100
error_orig = np.where(result_orig!=test_labels)[0].size / float(test_labels.size) * 100

print (' ')
print ('Classification error on sparse code', error_sparse)
print ('Classification error on original data', error_orig)

sparsity.visualize_dictionary(test_imgs, 128, M, N, 4, labels=result_sparse, gt_labels=test_labels, shape=(8,16))
raw_input()