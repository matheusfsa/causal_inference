from ..base import BootstrapCIModel
import numpy as np
from sklearn.base import BaseEstimator


class PropensityModel(BootstrapCIModel):
    def __init__(
        self,
        model: BaseEstimator,
        bootstrap: bool = True,
        bootstrap_samples: int = 1000,
        n_jobs: int = -1,
    ):
        super().__init__(bootstrap_samples, n_jobs)
        self._bootstrap = bootstrap
        self._model = model
        self._weight = None

    def _compute_ate(self, X, T, y):
        self._model = self._model.fit(X, T)
        score = self._model.predict_proba(X)[:, 1]
        self._weight = (T - score)/(score * (1 - score))
        return np.mean(self._weight * y)
