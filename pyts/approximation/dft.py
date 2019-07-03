"""Code for Discrete Fourier Transform."""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import f_classif
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
from math import ceil
from warnings import warn
from ..preprocessing import StandardScaler

def trim_to_length(X, n):
    # First remove the first complex coeff
    X = np.delete(X, [X.shape[1] // 2], axis=1)
    
    # Then (maybe) remove last complex coeff
    if X.shape[1] != n:
        X = X[:,:-1]
    
    return X


class DiscreteFourierTransform(BaseEstimator, TransformerMixin):
    
    def __init__(self, n_coefs=None, drop_sum=False, anova=False, norm_mean=False, norm_std=False):
        
        self.n_coefs   = n_coefs
        self.drop_sum  = drop_sum
        self.anova     = anova
        self.norm_mean = norm_mean
        self.norm_std  = norm_std
        
    def fit(self, X, y=None):
        _ = self.fit_transform(X, y)
        return self
    
    def _do_fft(self, X):
        n_samples, n_timestamps = X.shape
        
        X_fft  = np.fft.rfft(X)
        X_fft  = np.hstack([np.real(X_fft), np.imag(X_fft)])
        X_fft  = trim_to_length(X_fft, n_timestamps)
        
        if self.drop_sum:
            X_fft = X_fft[:, 1:]
        
        return X_fft
        
    def transform(self, X, chunked=False):
        check_is_fitted(self, 'support_')
        X = check_array(X, dtype='float64')
        
        X = StandardScaler(self.norm_mean, self.norm_std).fit_transform(X)
        
        X_fft = self._do_fft(X)
        
        out = X_fft[:, self.support_].copy()
        del X_fft
        return out
        
    def fit_transform(self, X, y=None):
        if self.anova:
            X, y = check_X_y(X, y, dtype='float64')
        else:
            X = check_array(X, dtype='float64')
        
        X = StandardScaler(self.norm_mean, self.norm_std).fit_transform(X)
        
        X_fft = self._do_fft(X)
        
        n_timestamps = X.shape[1]
        n_coefs      = self._check_params(n_timestamps)
        
        if self.anova:
            self.support_ = self._anova(X_fft, y, n_coefs, n_timestamps)
        else:
            self.support_ = np.arange(n_coefs)
        
        res = X_fft[:, self.support_].copy()
        del X_fft
        return res
        
    def _anova(self, X_fft, y, n_coefs, n_timestamps):
        if n_coefs < X_fft.shape[1]:
            non_constant = np.where(X_fft.var(axis=0) > 1e-8)[0]
            
            if non_constant.size == 0:
                raise ValueError("All the Fourier coefficients are constant.")
            elif non_constant.size < n_coefs:
                support = non_constant
            else:
                _, p = f_classif(X_fft[:, non_constant], y)
                support = non_constant[np.argsort(p)[:n_coefs]]
        else:
            support = np.arange(n_coefs)
        
        return support

    def _check_params(self, n_timestamps):
        if not ((isinstance(self.n_coefs,
                            (int, np.integer, float, np.floating)))
                or (self.n_coefs is None)):
            raise TypeError("'n_coefs' must be None, an integer or a float.")
        if isinstance(self.n_coefs, (int, np.integer)):
            if ((self.drop_sum)
                and (not (1 <= self.n_coefs <= (n_timestamps - 1)))):
                raise ValueError(
                    "If 'n_coefs' is an integer, it must be greater than or "
                    "equal to 1 and lower than or equal to (n_timestamps - 1) "
                    "if 'drop_sum=True'."
                )
            if not self.drop_sum and not (1 <= self.n_coefs <= n_timestamps):
                raise ValueError(
                    "If 'n_coefs' is an integer, it must be greater than or "
                    "equal to 1 and lower than or equal to n_timestamps "
                    "if 'drop_sum=False'."
                )
            n_coefs = self.n_coefs
        elif isinstance(self.n_coefs, (float, np.floating)):
            if not 0 < self.n_coefs <= 1:
                raise ValueError(
                    "If 'n_coefs' is a float, it must be greater "
                    "than 0 and lower than or equal to 1."
                )
            if self.drop_sum:
                n_coefs = ceil(self.n_coefs * (n_timestamps - 1))
            else:
                n_coefs = ceil(self.n_coefs * n_timestamps)
        else:
            n_coefs = (n_timestamps - 1) if self.drop_sum else n_timestamps
        return n_coefs
