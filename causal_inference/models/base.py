from typing import Union, Tuple
from abc import ABC, abstractmethod, abstractproperty
import pandas as pd
import numpy as np
from joblib import delayed, Parallel, effective_n_jobs


class BaseModel(ABC):

    @abstractmethod
    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        T: Union[pd.Series, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        **fit_kwargs,
    ):
        """Fit model"""
    @abstractproperty
    def ate(self) -> float:
        """Average treatment effect"""

    @abstractproperty
    def confidence_interval(self) -> Tuple[float, float]:
        """ATE confidence interval"""


class BootstrapCIModel(BaseModel):

    def __init__(
        self,
        bootstrap_samples: int = 1000,
        n_jobs: int = 1,
        bootstrap: bool = False,
    ):
        self._ate = None
        self._weight = None
        self._bootstrap_samples = bootstrap_samples
        self._n_jobs = effective_n_jobs(n_jobs)
        self._ci = (None, None)
        self._bootstrap = bootstrap

    def _compute_ate(self, X, T, y):
        raise NotImplementedError

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        T: Union[pd.Series, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        **fit_kwargs,
    ):
        self._ate = self._compute_ate(X, T, y)
        if self._bootstrap:
            self._run_bootstrap(X, T, y)
        return self

    def _safe_index(self, data, idx):
        if isinstance(data, pd.DataFrame):
            return data.iloc[idx, :]
        elif isinstance(data, pd.Series):
            return data.iloc[idx]
        return data[idx]

    def _run_bootstrap(self, X, T, y):
        samples = [
            np.random.choice(np.arange(X.shape[0]), X.shape[0], replace=True)
            for _ in range(self._bootstrap_samples)
        ]
        ates = (
            Parallel(n_jobs=self._n_jobs)(
                delayed(self._compute_ate)(
                   self._safe_index(X, sample_idx),
                   self._safe_index(T, sample_idx),
                   self._safe_index(y, sample_idx)
                )
                for sample_idx in samples
            )
        )
        self._ci = (np.percentile(ates, 2.5), np.percentile(ates, 97.5))

    @property
    def ate(self):
        return self._ate

    @property
    def confidence_interval(self) -> Tuple[float, float]:
        return self._ci
