"""Code for Multiple Coefficient Binning."""

import numpy as np
from numba import njit, prange
from scipy.stats import norm
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
from sklearn.utils.multiclass import check_classification_targets


@njit()
def _uniform_bins(timestamp_min, timestamp_max, n_timestamps, n_bins):
    bin_edges = np.empty((n_timestamps, n_bins - 1))
    for i in prange(n_timestamps):
        bin_edges[i] = np.linspace(
            timestamp_min[i], timestamp_max[i], n_bins + 1
        )[1:-1]
    return bin_edges


@njit()
def _digitize_1d(X, bins, n_samples, n_timestamps):
    X_digit = np.empty((n_samples, n_timestamps))
    for i in prange(n_timestamps):
        X_digit[:, i] = np.digitize(X[:, i], bins, right=True)
    return X_digit


@njit()
def _digitize_2d(X, bins, n_samples, n_timestamps):
    X_digit = np.empty((n_samples, n_timestamps))
    for i in prange(n_timestamps):
        X_digit[:, i] = np.digitize(X[:, i], bins[i], right=True)
    return X_digit


def _digitize(X, bins):
    n_samples, n_timestamps = X.shape
    if bins.ndim == 1:
        X_binned = _digitize_1d(X, bins, n_samples, n_timestamps)
    else:
        X_binned = _digitize_2d(X, bins, n_samples, n_timestamps)
    
    return X_binned.astype('int64')


class MultipleCoefficientBinning(BaseEstimator, TransformerMixin):
    def __init__(self, n_bins=4):
        self.n_bins = n_bins
        
    def fit(self, X, y):
        X, y = check_X_y(X, y, dtype='float64')
        check_classification_targets(y)
        n_samples, n_timestamps = X.shape
        self._n_timestamps_fit  = n_timestamps
        self._check_params(n_samples)
        self._check_constant(X)
        
        self.bin_edges_ = self._entropy_bins(X, y, n_timestamps, self.n_bins)
        
        return self
        
    def _entropy_bins(self, X, y, n_timestamps, n_bins):
        bins = np.empty((n_timestamps, n_bins - 1))
        clf  = DecisionTreeClassifier(criterion='entropy', max_leaf_nodes=n_bins)
        for i in range(n_timestamps):
            clf.fit(X[:, i][:, None], y)
            threshold = clf.tree_.threshold[clf.tree_.children_left != -1]
            if threshold.size < (n_bins - 1):
                raise ValueError("too many bins")
                
            bins[i] = threshold
        
        return np.sort(bins, axis=1)
    
    def transform(self, X):
        check_is_fitted(self, 'bin_edges_')
        X = check_array(X, dtype='float64')
        self._check_consistent_lengths(X)
        
        return _digitize(X, self.bin_edges_)

    def _check_params(self, n_samples):
        if not isinstance(self.n_bins, (int, np.integer)):
            raise TypeError("'n_bins' must be an integer.")
        
        if not 2 <= self.n_bins <= n_samples:
            raise ValueError(
                "'n_bins' must be greater than or equal to 2 and lower than "
                "or equal to n_samples (got {0}).".format(self.n_bins)
            )

    def _check_constant(self, X):
        if np.any(np.max(X, axis=0) - np.min(X, axis=0) == 0):
            raise ValueError("At least one timestamp is constant.")

    def _check_consistent_lengths(self, X):
        if self._n_timestamps_fit != X.shape[1]:
            raise ValueError(
                "The number of timestamps in X must be the same as "
                "the number of timestamps when `fit` was called "
                "({0} != {1}).".format(self._n_timestamps_fit, X.shape[1])
            )
