# The toughest case is actually when we have exact linear relationships
# The remainder information is subtracted out, but because of numerical error
# there is still an exact linear relationship in the noise
# Because the sieve is scale invariant, this relationship is picked up again
# at the next layer! 

import numpy as np
import sys

sys.path.append('..')
import sieve

np.set_printoptions(precision=2, suppress=True, linewidth=150)


def check_clusters(z, k, nr):
    for i in range(k):
        in_cluster = z[nr * i: nr * (i + 1)]
        out_cluster = np.hstack([z[:nr * i], z[nr * (i + 1):]])
        if not np.all(in_cluster == in_cluster[0]):
            return False
        if np.any(out_cluster == in_cluster[0]):
            return False
    return True


k, ns = 5, 50  # Number of signals and samples
nr = 2  # Number of copies of each signal
n = k * nr  # Number of variables
np.random.seed(0)
x = np.random.random((ns, k))
x = np.repeat(x, nr, axis=1)

out = sieve.Sieve(n_hidden=k).fit(x)
print out.ws.T
print 'ws', np.argmax(np.abs(out.ws[:, :n]), axis=0)
mis = np.array([mi[:n] for mi in out.mis])
print mis.T
print 'mis', np.argmax(mis, axis=0)
assert np.all(np.argmax(mis, axis=0) == np.argmax(np.abs(out.ws[:, :n]), axis=0))
assert check_clusters(np.argmax(mis, axis=0), k, nr)
