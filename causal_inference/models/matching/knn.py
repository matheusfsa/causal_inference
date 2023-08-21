from ..base import BootstrapCIModel
from sklearn.neighbors import KNeighborsRegressor


class KNNModel(BootstrapCIModel):

    def __init__(
        self,
        bootstrap_samples: int = 1000,
        n_jobs: int = -1,
        bootstrap: bool = True
    ):
        super().__init__(bootstrap_samples, n_jobs, bootstrap)

    def _train_knn(self, X, y, **fit_kwargs):
        return KNeighborsRegressor(**fit_kwargs).fit(X, y)

    def _compute_ate(self, X, T, y):
        _m0 = self._train_knn(X[T == 0], y[T == 0])
        _m1 = self._train_knn(X[T == 1], y[T == 1])

        y0_pred = _m0.predict(X[T == 1])
        y1_pred = _m1.predict(X[T == 0])

        return (
            ((y[T == 1] - y0_pred).sum() - (y[T == 0] - y1_pred).sum()) /
            T.shape[0]
        )
