from ..base import BootstrapCIModel
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression


class DoublyRobustEstimator(BootstrapCIModel):
    def __init__(
        self,
        propensity_model: BaseEstimator,
        bootstrap: bool = True,
        bootstrap_samples: int = 1000,
        n_jobs: int = -1,
    ):
        super().__init__(bootstrap_samples, n_jobs)
        self._bootstrap = bootstrap
        self._propensity_model = propensity_model
        self._weight = None

    def _compute_ate(self, X, T, y):

        mu0 = (
            LinearRegression()
            .fit(X[T == 0], y[T == 0])
            .predict(X[T == 0])
            .mean()
        )
        mu1 = (
            LinearRegression()
            .fit(X[T == 1], y[T == 1])
            .predict(X[T == 1])
            .mean()
        )

        self._propensity_model = self._propensity_model.fit(X, T)
        score = self._propensity_model.predict_proba(X)[:, 1]

        return (
            (np.mean((T * (y - mu1))/score) + mu1) -
            (np.mean(((1 - T) * (y - mu0))/(1 - score)) + mu0)
        )
