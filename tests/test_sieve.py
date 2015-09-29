import numpy as np
import sys
sys.path.append('..')
import sieve
from scipy.stats import kendalltau


kwargs = {'verbose': True, 'seed': 1}
np.random.seed(kwargs['seed'])
test_array_d = np.repeat([[0, 0, 0], [1, 1, 1]], 3, axis=0)
test_array_c = np.repeat(np.random.random((100, 1)), 3, axis=1)

# CONTINUOUS TESTS
def test_discrete():
    out = sieve.Sieve(n_hidden=1, **kwargs)
    y = out.fit_transform(test_array_d)
    assert np.allclose(kendalltau(y[:, 0], test_array_d[:, 0]), 1, atol=1e-3)


def test_discrete():
    out = sieve.Sieve(n_hidden=1, **kwargs)
    y = out.fit_transform(test_array_c)
    assert np.allclose(kendalltau(y[:, 0], test_array_c[:, 0]), 1, atol=1e-3)

