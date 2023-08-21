from ..base import BaseModel
from typing import Union, Tuple
from copy import copy
import pandas as pd
import numpy as np
from statsmodels.api import OLS
from statsmodels.regression.linear_model import RegressionResultsWrapper


class LinearModel(BaseModel):
    _model: RegressionResultsWrapper = None

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        T: Union[pd.Series, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        **fit_kwargs,
    ):
        if isinstance(X, pd.DataFrame):
            exog = copy(X)
        else:
            exog = pd.DataFrame(
                X, columns=[f'confounder_{i}' for i in range(X.shape[1])]
            )
        exog['treatment'] = T
        exog['intercept'] = 1
        self._model = OLS(y, exog, missing='drop').fit(**fit_kwargs)
        return self

    @property
    def ate(self):
        return self._model.params['treatment']

    @property
    def confidence_interval(self) -> Tuple[float, float]:
        ci_data = self._model.conf_int()
        ci = ci_data.loc['treatment', :]
        return ci[0], ci[1]
