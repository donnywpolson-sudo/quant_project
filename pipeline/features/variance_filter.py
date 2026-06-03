import logging
import numpy as np
import polars as pl
from sklearn.feature_selection import VarianceThreshold
logger = logging.getLogger(__name__)

def remove_constant_features(df: pl.DataFrame, feature_cols: list, threshold: float=1e-09) -> list:
    if len(feature_cols) == 0:
        return []
    # VarianceThreshold requires >1 sample to estimate variance.
    if df.height <= 1:
        return feature_cols
    X = df.select(feature_cols).fill_null(0.0).to_numpy().astype(np.float32)
    selector = VarianceThreshold(threshold=threshold)
    try:
        selector.fit(X)
    except ValueError:
        # All features have zero variance (e.g., tiny fold with constant data).
        # Return all columns — Ridge L2 regularization will zero their coefficients.
        logger.info('VarianceThreshold: no feature meets threshold; keeping all %d.', len(feature_cols))
        return list(feature_cols)
    keep_mask = selector.get_support()
    kept = [col for col, keep in zip(feature_cols, keep_mask) if keep]
    removed = len(feature_cols) - len(kept)
    if removed > 0:
        logger.info(f'Removed {removed} constant features. Remaining: {len(kept)}')
    return kept