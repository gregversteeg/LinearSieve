import numpy as np
import sys
sys.path.append('..')
import sieve
from scipy.stats import kendalltau

kwargs = {'verbose': True, 'seed': 1}
np.random.seed(kwargs['seed'])
test_array_d = np.repeat([[0, 0, 0], [1, 1, 1]], 3, axis=0)


def test_discrete():
    out = sieve.Sieve(n_hidden=1, **kwargs)
    y = out.fit_transform(test_array_d)
    assert np.allclose(kendalltau(y[:, 0], test_array_d[:, 0])[0], 1)


def test_continuous():
    out = sieve.Sieve(n_hidden=1, **kwargs)
    x = test_set((5,), noise=0.001)
    y = out.fit_transform(x)
    assert np.allclose(np.abs(np.corrcoef(x[:, 0], y[:, 0])), 1)


def test_set(groups=(3, 2), noise=0.05, samples=1000):
    a = []
    for i, k in enumerate(groups):
        a.append(np.repeat(np.random.randn(samples, 1), k, axis=1))
    a = np.hstack(a)
    a += noise * np.random.normal(size=a.shape)
    a -= a.mean(axis=0)
    return a
