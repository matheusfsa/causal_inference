from ..base import BaseModel
from typing import Union, Tuple
import pandas as pd
import numpy as np
from statsmodels.api import OLS
from statsmodels.regression.linear_model import RegressionResultsWrapper


class BiasedModel(BaseModel):
    _model: RegressionResultsWrapper = None

    def fit(
        self,
        T: Union[pd.Series, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        **fit_kwargs,
    ):
        exog = pd.DataFrame()
        exog['treatment'] = T
        exog['intercept'] = 1
        self._model = OLS(y, exog).fit(**fit_kwargs)
        return self

    @property
    def ate(self):
        return self._model.params['treatment']

    @property
    def confidence_interval(self) -> Tuple[float, float]:
        ci_data = self._model.conf_int()
        ci = ci_data.loc['treatment', :]
        return ci[0], ci[1]
