"""Forecaster abstract base class and Forecast dataclass.

Any new model (SARIMA, Prophet, LightGBM…) is a Forecaster subclass that
implements fit() and predict() — nothing else needs to change in the pipeline.

TODO (Phase 3): Implement concrete subclasses in baselines.py.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd


@dataclass
class Forecast:
    """Probabilistic forecast output.

    Attributes
    ----------
    point : pd.Series
        Point forecast indexed by date.
    lower : pd.Series
        Lower bound of the prediction interval, same index as point.
    upper : pd.Series
        Upper bound of the prediction interval, same index as point.
    level : float
        Nominal coverage of the prediction interval (e.g. 0.80 for 80 %).
    """

    point: pd.Series
    lower: pd.Series
    upper: pd.Series
    level: float


class Forecaster(ABC):
    """Abstract base for all forecasting models.

    Follows a fit / predict pattern so models are interchangeable.
    """

    @abstractmethod
    def fit(self, y: pd.Series) -> "Forecaster":
        """Fit the model on a univariate time series.

        Parameters
        ----------
        y : pd.Series
            Daily time series (DatetimeIndex, UTC).

        Returns
        -------
        Forecaster
            self, for method chaining.
        """
        ...

    @abstractmethod
    def predict(self, horizon: int, level: float = 0.80) -> Forecast:
        """Generate a probabilistic forecast.

        Parameters
        ----------
        horizon : int
            Number of future periods (days) to forecast.
        level : float
            Nominal prediction-interval coverage (0 < level < 1).

        Returns
        -------
        Forecast
            Point forecast plus lower/upper bounds.
        """
        ...