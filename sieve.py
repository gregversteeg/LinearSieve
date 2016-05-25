"""The Linear Information Sieve

   Copyright 2016 Greg Ver Steeg

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at
       http://www.apache.org/licenses/LICENSE-2.0
   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""

import numpy as np


class Sieve(object):
    """
    The Linear Information Sieve

    Conventions
    ----------
    Code follows sklearn naming/style (e.g. fit(X) to train).
    We use Einstein summation (np.einsum) for most matrix operations (very fast in numpy).
    The index convention is as follows:
        i = 1...n_variables (may be number of variables at level k)
        j = 1...n_hidden, used for indexing latent factors
        k = 1...n_hidden, used for sums over all levels
        l = 1...n_samples

    Parameters
    ----------
    n_hidden : int, default = 2
        The number of latent factors / layers of the sieve to use.

    max_iter : int, default = 5000
        The max. number of iterations for each latent factor.

    repeat : int, default = 1
        You can repeat each sieve optimization a few times and take the best result.

    noise : float default = 0.1
        To keep MI from being infinite, we imagine some fundamental measurement noise on Y. This number is
        arbitrary but it sets the scale of Y.

    verbose : int, optional
        The verbosity level. The default, zero, means silent mode. 1 outputs TC(X;Y) as you go
        2 output alpha matrix and MIs as you go.

    seed : integer or numpy.RandomState, optional
        A random number generator instance to define the state of the
        random permutations generator. If an integer is given, it fixes the
        seed. Defaults to the global numpy random number generator.

    Attributes
    ----------
    ws : float array
        The weights to calculate latent factors.

    moments : list of dicts
        The moments calculated at each level.

    mean_x
        Mean of the training data for each variable.

    tc_history
        Keeps track of convergence for each latent factor (If you used repeat, it shows repeated optimizations
        sequentially.)

    tcs
        The total correlation explained by each factor, or the total overall (tc = sum (tcs))

    References
    ----------
    [1] Greg Ver Steeg and Aram Galstyan. "The Information Sieve", ICML 2016.
    [2] Greg Ver Steeg, Shuyang Gao, Kyle Reing, and Aram Galstyan. "Sifting Common Information from Many Variables"
    """

    def __init__(self, n_hidden=2, max_iter=5000, noise=0.1, repeat=1, tol=1e-8, verbose=False, seed=None, **kwargs):
        self.n_hidden = n_hidden  # Number of latent factors to learn
        self.max_iter = max_iter  # Iterations at each layer
        self.noise = noise  # Fundamental limit on noise of Y
        self.repeat = repeat  # For each factor, repeat with different random IC and take best solution.
        self.tol = tol  # Check for convergence
        self.verbose = verbose
        np.random.seed(seed)  # Set seed for deterministic results
        self.kwargs = kwargs

        # Initialize these when we fit on data
        self.ws = []  # List of weight arrays to get y_k = w \dot x^{k=1}
        self.alpha = []  # Whether xi is transformed: x_i^k = x_i^{k-1} - alpha <x_i^{k-1} y_k> y_k
        self.moments = []  # list of dictionary of moments for each level
        self.mean_x = None  # Mean is subtracted out, save for prediction/inversion
        self.nv = None  # Number of variables in input data
        self.tc_history = [[0] for _ in range(self.n_hidden)]  # Keep track of TC convergence for each factor
        if verbose:
            np.set_printoptions(precision=3, suppress=True)
            print 'Linear Sieve with %d latent factors' % n_hidden

    def mi_j(self, j):
        """MIs for level j"""
        with np.errstate(all='ignore'):
            return np.where(self.moments[j]["X_i^2"] > 0, -0.5 * np.log1p(- self.moments[j]["r^2"]), 0)

    def tc_j(self, j):
        """TC at level j. Actually only a lower bound."""
        mis = self.mi_j(j)
        return np.sum(mis) - 0.5 * np.log(self.moments[j]["Y^2"]) + np.log(self.noise)  # The objective at each level

    @property
    def mis(self):
        """All MIs"""
        return [self.mi_j(j) for j in range(len(self.moments))]

    @property
    def tcs(self):
        """TCs at each level"""
        return [self.tc_j(j) for j in range(len(self.moments))]

    @property
    def tc(self):
        return sum(self.tcs)

    def fit_transform(self, x, **kwargs):
        self.fit(x)
        return self.transform(x, **kwargs)

    def fit(self, x):
        """Learn n_hidden latent factors that extract as much common information from X as possible."""
        x = np.asarray(x, dtype=float)
        ns, nv = x.shape
        self.nv = nv  # Number of variables in input data
        self.mean_x = np.ma.mean(np.ma.masked_array(x, np.isnan(x)), axis=0).data[np.newaxis,
                      :]  # exclude missing (np.nan)
        x = np.where(np.isnan(x), self.mean_x, x)  # Impute mean for missing values
        x = np.hstack([(x - self.mean_x), np.zeros((ns, self.n_hidden))])  # Allocate space and fill in
        self.ws = np.zeros((self.n_hidden, nv + self.n_hidden))

        for j in range(self.n_hidden):
            if self.verbose:
                print 'updating %d' % j
            self.moments.append({})  # Dictionary of moments for this level
            m = self.moments[j]  # Abbreviation
            nv_k = nv + j  # Number of variables on this level
            m["X_i^2"] = (np.einsum("li,li->i", x[:, :nv_k], x[:, :nv_k]) / (ns - 1)).clip(1e-10)  # Unbiased variance

            best_tc = -np.inf
            for _ in range(self.repeat):
                self.ws[j][:nv_k] = np.random.randn(nv_k) / np.sqrt(nv) * self.noise / np.sqrt(
                        m["X_i^2"])  # Random initialization
                self.update_parameters(x, j)  # Update moments and normalize w
                for i_loop in range(self.max_iter):
                    self.ws[j, :nv_k] = self.noise ** 2 * m["X_i Y"] / (
                    m["X_i^2"] * m["Y^2"] - m["X_i Y"] ** 2)  # Update w, Eq. 9 in paper
                    self.update_parameters(x, j)  # Update moments

                    self.tc_history[j].append(self.tc_j(j))
                    if self.verbose:
                        print 'TC = %0.5f' % self.tc_history[j][-1]
                    if np.abs(self.tc_history[j][-1] - self.tc_history[j][-2]) < self.tol:
                        break  # Stop if converged
                else:
                    if self.verbose:
                        print "Warning: Convergence not achieved in %d iterations. Increase max_iter." % self.max_iter
                if self.tc_j(j) > best_tc:
                    best_w = np.copy(self.ws[j, :nv_k])
                    best_tc = self.tc_j(j)
            self.ws[j][:nv_k] = best_w
            self.update_parameters(x, j)  # Update moments and normalize w

            x[:, nv_k] = np.dot(self.ws[j], x.T)  # This is just Y
            for i in range(nv_k):
                x[:, i] -= m['X_i Y'][i] / (m["Y^2"] - self.noise ** 2) * (
                x[:, nv_k] + self.noise * np.random.randn(ns))
        return self

    def transform(self, x, remainder=False, level=-1):
        """Transform an array of inputs, x, into an array of k latent factors, Y.
            Optionally, you can get the remainder information and/or stop at a specified level of the sieve."""
        x = np.asarray(x, dtype=float)
        ns, nv = x.shape
        assert self.nv == nv, "Incorrect number of variables in input, %d instead of %d" % (nv, self.nv)
        x = np.where(np.isnan(x), self.mean_x, x)  # Impute mean for missing values
        x = np.hstack([(x - self.mean_x), np.zeros((ns, self.n_hidden))])  # Allocate space and fill in
        ys = []
        for j in range(self.n_hidden):
            y = np.dot(self.ws[j], x.T)
            x[:, nv + j] = y
            for i in range(nv + j):
                x[:, i] -= self.moments[j]['X_i Y'][i] / (self.moments[j]["Y^2"] - self.noise ** 2) * x[:,
                                                                                                      nv + j]  # Don't add noise for transform, only in training
            ys.append(y)
            if j == np.mod(level, self.n_hidden):
                ys = np.vstack(ys).T
                if not remainder:
                    return ys
                else:
                    return ys, x[:, :nv + j + 1]

    def predict(self, ys):
        """Based on latent factors, Y, predict a lossy reconstruction of the original data, X."""
        nv = self.nv
        x = np.zeros((len(ys), nv))
        for j in range(self.n_hidden - 1, -1, -1):
            for i in range(nv + j):
                if i < nv:
                    x[:, i] += self.moments[j]['X_i Y'][i] / (self.moments[j]["Y^2"] - self.noise ** 2) * ys[:, j]
        return x + self.mean_x

    def invert(self, xbar):
        # Exactly invert bar X to recover X.
        raise NotImplementedError

    def update_parameters(self, x, j):
        """Update moments based on the weights."""
        nv_k = self.nv + j  # Number of variables on this level
        m = self.moments[j]  # Update moments, abbreviate for readability
        if np.max(np.abs(self.ws)) > 1e10:
            print 'Warning: if w is getting huge, a perfect linear rel. is causing divergence.'
            self.ws = self.ws.clip(-1e10, 1e10)
        y = np.dot(x, self.ws[j])
        m["Y^2"] = var(y) + self.noise ** 2  # <Y^2>
        m["X_i Y"] = np.dot(x.T, y)[:nv_k] / len(y)
        m["r^2"] = (m["X_i Y"] ** 2 / (m["X_i^2"] * m["Y^2"]))  # .clip(0, 1 - 1e-10)


def var(y):
    # 4 times faster than numpy.std!
    return np.dot(y, y) / len(y)
