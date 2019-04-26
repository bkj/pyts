"""Code for Word ExtrAction for time SEries cLassification."""

import sys
import numpy as np
from scipy.sparse import csc_matrix, hstack
from sklearn.utils.validation import check_array, check_X_y, check_is_fitted
from sklearn.utils.multiclass import check_classification_targets
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import chi2
from sklearn.pipeline import Pipeline
from joblib import Parallel, delayed

from ..utils import windowed_view
from ..approximation.dft import DiscreteFourierTransform
from ..approximation.mcb import MultipleCoefficientBinning

def _weasel_fit(X, y, sfa, chi2_threshold, window_size, window_step):
    n_samples, n_timestamps = X.shape
    n_windows  = ((n_timestamps - window_size + window_step) // window_step)
    
    X_windowed = windowed_view(X, window_size=window_size, window_step=window_step)
    X_windowed = X_windowed.reshape(n_samples * n_windows, window_size)
    
    X_sfa = sfa.fit_transform(X_windowed, np.repeat(y, n_windows))
    
    X_word = np.asarray([''.join([str(xx) for xx in X_sfa[i]]) for i in range(n_samples * n_windows)])
    X_word = X_word.reshape(n_samples, n_windows)
    X_bow  = np.asarray([' '.join(X_word[i]) for i in range(n_samples)])
    
    vec      = CountVectorizer(ngram_range=(1, 2))
    X_counts = vec.fit_transform(X_bow)
    
    chi2_stats, _ = chi2(X_counts, y)
    rel_feats     = np.where(chi2_stats > chi2_threshold)[0]
    
    return rel_feats, sfa, vec, X_counts[:, rel_feats]


def _weasel_transform(X, window_size, window_step, sfa, vec, rel_feats, chunk_size=1e8):
    n_samples, n_timestamps = X.shape
    n_windows  = ((n_timestamps - window_size + window_step) // window_step)
    
    def _transform_chunk(X_chunk):
        Xw = windowed_view(X_chunk, window_size=window_size, window_step=window_step)
        Xw = Xw.reshape(X_chunk.shape[0] * n_windows, window_size)
        return sfa.transform(Xw)
    
    Xw_sz = X[0,0].nbytes * (n_windows * window_size)
    X_sfa = [_transform_chunk(X_chunk) for X_chunk in np.array_split(X, 1 + Xw_sz // chunk_size)]
    X_sfa = np.vstack(X_sfa)
    
    X_word = np.asarray([''.join([str(xx) for xx in X_sfa[i]]) for i in range(n_samples * n_windows)])
    X_word = X_word.reshape(n_samples, n_windows)
    X_bow  = np.asarray([' '.join(X_word[i]) for i in range(n_samples)])
    
    X_counts = vec.transform(X_bow)
    
    return X_counts[:, rel_feats]


class WEASEL(BaseEstimator, TransformerMixin):
    def __init__(self, word_size=4, n_bins=4,
                 window_sizes=[0.1, 0.3, 0.5, 0.7, 0.9], window_steps=None,
                 anova=True, drop_sum=True, norm_mean=True, norm_std=True,
                 chi2_threshold=2, n_jobs=1, verbose=10):
        
        do_shuffle = True
        if do_shuffle:
            perm = np.random.permutation(len(window_sizes))
            window_sizes = [window_sizes[i] for i in perm]
            window_steps = [window_steps[i] for i in perm]
        
        self.word_size      = word_size
        self.n_bins         = n_bins
        self.window_sizes   = window_sizes
        self.window_steps   = window_steps
        self.anova          = anova
        self.drop_sum       = drop_sum
        self.norm_mean      = norm_mean
        self.norm_std       = norm_std
        self.n_bins         = n_bins
        self.chi2_threshold = chi2_threshold
        self.n_jobs         = n_jobs
        self.verbose        = verbose
    
    def fit(self, X, y):
        _ = self.fit_transform(X, y)
        return self
    
    def fit_transform(self, X, y):
        X, y = check_X_y(X, y)
        check_classification_targets(y)
        n_samples, n_timestamps = X.shape
        window_sizes, window_steps = self._check_params(n_timestamps)
        self._window_sizes = window_sizes
        self._window_steps = window_steps
        
        dft_params = {
            "n_coefs"    : self.word_size,
            "drop_sum"   : self.drop_sum,
            "anova"      : self.anova,
            "norm_mean"  : self.norm_mean,
            "norm_std"   : self.norm_std,
        }
        
        mcb_params = {
            "n_bins" : self.n_bins,
        }
        
        sfa = Pipeline([
            ('dft', DiscreteFourierTransform(**dft_params)),
            ('mcb', MultipleCoefficientBinning(**mcb_params))
        ])
        
        gen  = zip(window_sizes, window_steps)
        jobs = [delayed(_weasel_fit)(X, y, sfa, self.chi2_threshold, *args) for args in gen]
        if self.verbose: print('WEASEL: dispatching %d jobs' % len(jobs), file=sys.stderr)
        res  = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(jobs)
        
        self._rel_feats_list, self._sfa_list, self._vectorizer_list, X_features = zip(*res)
        
        return hstack(X_features)

    def transform(self, X):
        
        check_is_fitted(self, ['_rel_feats_list', '_sfa_list', '_vectorizer_list'])
        
        X = check_array(X)
        n_samples, n_timestamps = X.shape
        
        X_features = csc_matrix((n_samples, 0), dtype=np.int64)
        
        gen = zip(
            self._window_sizes,
            self._window_steps,
            self._sfa_list,
            self._vectorizer_list,
            self._rel_feats_list
        )
        
        jobs = [delayed(_weasel_transform)(X, *args) for args in gen]
        if self.verbose: print('WEASEL: dispatching %d jobs' % len(jobs), file=sys.stderr)
        X_features = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(jobs)
        return hstack(X_features)

    def _check_params(self, n_timestamps):
        if not isinstance(self.word_size, (int, np.integer)):
            raise TypeError("'word_size' must be an integer.")
        if not self.word_size >= 1:
            raise ValueError("'word_size' must be a positive integer.")

        if not isinstance(self.window_sizes, (list, tuple, np.ndarray)):
            raise TypeError("'window_sizes' must be array-like.")
        window_sizes = check_array(self.window_sizes, ensure_2d=False,
                                   dtype=None)
        if window_sizes.ndim != 1:
            raise ValueError("'window_sizes' must be one-dimensional.")
        if not issubclass(window_sizes.dtype.type, (np.integer, np.floating)):
            raise ValueError("The elements of 'window_sizes' must be integers "
                             "or floats.")
        if issubclass(window_sizes.dtype.type, np.floating):
            if not (np.min(window_sizes) > 0 and np.max(window_sizes) <= 1):
                raise ValueError(
                    "If the elements of 'window_sizes' are floats, they all "
                    "must be greater than 0 and lower than or equal to 1."
                )
            window_sizes = np.ceil(window_sizes * n_timestamps).astype('int64')
        if not np.max(window_sizes) <= n_timestamps:
            raise ValueError("All the elements in 'window_sizes' must be "
                             "lower than or equal to n_timestamps.")

        if self.drop_sum and not self.word_size < np.min(window_sizes):
            raise ValueError(
                "If 'drop_sum=True', 'word_size' must be lower than "
                "the minimum value in 'window_sizes'."
            )
        if not (self.drop_sum or self.word_size <= np.min(window_sizes)):
            raise ValueError(
                "If 'drop_sum=False', 'word_size' must be lower than or "
                "equal to the minimum value in 'window_sizes'."
            )

        if not ((self.window_steps is None)
                or isinstance(self.window_steps, (list, tuple, np.ndarray))):
            raise TypeError("'window_steps' must be None or array-like.")
        if self.window_steps is None:
            window_steps = window_sizes
        else:
            window_steps = check_array(self.window_steps, ensure_2d=False,
                                       dtype=None)
            if window_steps.ndim != 1:
                raise ValueError("'window_steps' must be one-dimensional.")
            if window_steps.size != window_sizes.size:
                raise ValueError("If 'window_steps' is not None, it must have "
                                 "the same size as 'window_sizes'.")
            if not issubclass(window_steps.dtype.type,
                              (np.integer, np.floating)):
                raise ValueError(
                    "If 'window_steps' is not None, the elements of "
                    "'window_steps' must be integers or floats."
                )
            if issubclass(window_steps.dtype.type, np.floating):
                if not ((np.min(window_steps) > 0
                         and np.max(window_steps) <= 1)):
                    raise ValueError(
                        "If the elements of 'window_steps' are floats, they "
                        "all must be greater than 0 and lower than or equal "
                        "to 1."
                    )
                window_steps = np.ceil(
                    window_steps * n_timestamps).astype('int64')
            if not ((np.min(window_steps) >= 1)
                    and (np.max(window_steps) <= n_timestamps)):
                raise ValueError("All the elements in 'window_steps' must be "
                                 "greater than or equal to 1 and lower than "
                                 "or equal to n_timestamps.")

        if not isinstance(self.chi2_threshold,
                          (int, np.integer, float, np.floating)):
            raise TypeError("'chi2_threshold' must be a float or an "
                            "integer.")
        if not self.chi2_threshold > 0:
            raise ValueError("'chi2_threshold' must be positive.")
        return window_sizes, window_steps
