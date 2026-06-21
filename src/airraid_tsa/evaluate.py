"""Time-based evaluation: walk-forward, point + probabilistic metrics.

TODO (Phase 3):
- time_split(series, cutoff_date) -> (train, test).
- walk_forward_evaluate(forecaster, series, min_train, horizon) -> metrics DataFrame.
- Metrics to implement:
    - MAE, RMSE (point).
    - MASE (scaled vs seasonal-naive; <1 means useful).
    - Interval coverage (share of test points inside [lower, upper]).
    - Mean interval width (sharpness).
    - Pinball / quantile loss at interval quantiles.
- print_metrics_table(metrics) -> formatted table (printed + saved to outputs/).
"""

# TODO: implement evaluation pipeline.