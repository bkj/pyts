#!/usr/bin/env python

"""
    transformation/sfa.py
"""


from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted
from sklearn.base import BaseEstimator, TransformerMixin

from .dft import DiscreteFourierTransform
from .mcb import MultipleCoefficientBinning

class SymbolicFourierApproximation(BaseEstimator, TransformerMixin):
    def __init__(self, n_coefs=None, drop_sum=False, anova=False,
                 norm_mean=False, norm_std=False, n_bins=4):
        
        # DFT
        self.n_coefs   = n_coefs
        self.drop_sum  = drop_sum
        self.anova     = anova
        self.norm_mean = norm_mean
        self.norm_std  = norm_std
        
        # MCB
        self.n_bins    = n_bins
        
    def fit(self, X, y=None):
        _ = self.fit_transform(X, y)
        return self
        
    def transform(self, X):
        check_is_fitted(self, ['support_', 'bin_edges_'])
        return self._pipeline.transform(X)
        
    def fit_transform(self, X, y=None):
        dft = DiscreteFourierTransform(
            n_coefs=self.n_coefs,
            drop_sum=self.drop_sum,
            anova=self.anova,
            norm_mean=self.norm_mean,
            norm_std=self.norm_std,
        )
        
        mcb = MultipleCoefficientBinning(n_bins=self.n_bins)
        self._pipeline = Pipeline([('dft', dft), ('mcb', mcb)])
        
        X_sfa = self._pipeline.fit_transform(X, y)
        
        self.support_   = self._pipeline.named_steps['dft'].support_
        self.bin_edges_ = self._pipeline.named_steps['mcb'].bin_edges_
        
        return X_sfa
