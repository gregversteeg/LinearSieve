print(__doc__)

# Authors: Vlad Niculae, Alexandre Gramfort
# License: BSD 3 clause

import logging
from time import time
import os
import sys

import numpy as np
from numpy.random import RandomState
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_olivetti_faces
from sklearn.cluster import MiniBatchKMeans
from sklearn import decomposition

sys.path.append('..')
import sieve

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

if not os.path.exists('faces'):
    os.makedirs('faces')
if not os.path.exists('faces/remainder'):
    os.makedirs('faces/remainder')

n_row, n_col = 2, 3
n_components = n_row * n_col
image_shape = (64, 64)
rng = RandomState(0)

###############################################################################
# Load faces data
dataset = fetch_olivetti_faces(shuffle=True, random_state=rng)
faces = dataset.data

n_samples, n_features = faces.shape

# global centering
faces_centered = faces - faces.mean(axis=0)

# local centering
faces_centered -= faces_centered.mean(axis=1).reshape(n_samples, -1)

print("Dataset consists of %d faces" % n_samples)


###############################################################################
def plot_gallery(title, images, n_col=n_col, n_row=n_row):
    plt.figure(figsize=(2. * n_col, 2.26 * n_row))
    plt.suptitle(title, size=16)
    for i, comp in enumerate(images):
        plt.subplot(n_row, n_col, i + 1)
        vmax = max(comp.max(), -comp.min())
        plt.imshow(comp.reshape(image_shape), cmap=plt.cm.gray,
                   interpolation='nearest',
                   vmin=-vmax, vmax=vmax)
        plt.xticks(())
        plt.yticks(())
    plt.subplots_adjust(0.01, 0.05, 0.99, 0.93, 0.04, 0.)


###############################################################################
# List of the different estimators, whether to center and transpose the
# problem, and whether the transformer uses the clustering API.
estimators = [
    ('Linear Sieve',
        sieve.Sieve(n_hidden=n_components), False),
    # ('Eigenfaces - RandomizedPCA',
    #     decomposition.RandomizedPCA(n_components=n_components, whiten=True), True),
    # ('Non-negative components - NMF',
    #     decomposition.NMF(n_components=n_components, init='nndsvda', beta=5.0,
    #                    tol=5e-3, sparseness='components'), False),
    # ('Independent components - FastICA',
    #     decomposition.FastICA(n_components=n_components, whiten=True), True),
    # ('Sparse comp. - MiniBatchSparsePCA',
    #     decomposition.MiniBatchSparsePCA(n_components=n_components, alpha=0.8,
    #                                   n_iter=100, batch_size=3, random_state=rng), True),
    # ('MiniBatchDictionaryLearning',
    #     decomposition.MiniBatchDictionaryLearning(n_components=15, alpha=0.1,
    #                                               n_iter=50, batch_size=3, random_state=rng), True),
    # ('Cluster centers - MiniBatchKMeans',
    #     MiniBatchKMeans(n_clusters=n_components, tol=1e-3, batch_size=20, max_iter=50, random_state=rng), True),
    # ('Factor Analysis components - FA',
    #    decomposition.FactorAnalysis(n_components=n_components, max_iter=2), True),
]


###############################################################################
# Plot a sample of the input data

plot_gallery("First centered Olivetti faces", faces_centered[:n_components])

###############################################################################
# Do the estimation and plot it

for name, estimator, center in estimators:
    print("Extracting the top %d %s..." % (n_components, name))
    t0 = time()
    data = faces
    if center:
        data = faces_centered
    estimator.fit(data)
    train_time = (time() - t0)
    print("done in %0.3fs" % train_time)
    if hasattr(estimator, 'cluster_centers_'):
        components_ = estimator.cluster_centers_
    elif hasattr(estimator, 'components_'):
        components_ = estimator.components_
    else:
        components_ = np.array([estimator.ws[i][:n_features] for i in range(n_components)])
    if hasattr(estimator, 'noise_variance_'):
        plot_gallery("Pixelwise variance",
                     estimator.noise_variance_.reshape(1, -1), n_col=1,
                     n_row=1)
    plot_gallery('%s - Train time %.1fs' % (name, train_time),
                 components_[:n_components])

    plt.savefig('faces/%s.pdf' % name)
    plt.clf()

n_components = 50
out = sieve.Sieve(n_hidden=n_components)
out.fit(data)
components_ = np.array([out.ws[i][:n_features] for i in range(n_components)])
plot_gallery('%s' % 'Linear Sieve Components', components_[:n_components],  n_col=10, n_row=5)
plt.savefig('faces/big_components.pdf')
plt.clf()
xs = []
for i in range(n_components):
    ys, xbar = out.transform(data, level=i, remainder=True)
    print xbar.shape
    xs.append(xbar[:, :n_features])
xs = np.array(xs)
print 'x shape', xs.shape
print 'alpha', out.alpha
for l in range(30):
    plot_gallery('Face %d' % l, xs[:, l, :], n_col=10, n_row=int(np.ceil(float(n_components)/10)))

    plt.savefig('faces/remainder/%d.pdf' % l)
    plt.close('all')