import logging
import numpy as np
import polars as pl
logger = logging.getLogger(__name__)

def correlation_prune(df: pl.DataFrame, feature_cols: list, threshold: float=0.9) -> list:
    pass
    if df.height == 0 or len(feature_cols) == 0:
        return feature_cols
    logger.info(f'Running correlation pruning on {len(feature_cols)} features (threshold={threshold})...')
    X = df.select(feature_cols).fill_null(0.0).to_numpy().astype(np.float32)
    with np.errstate(divide='ignore', invalid='ignore'):
        corr = np.corrcoef(X, rowvar=False)
    keep = []
    dropped = set()
    for i, f in enumerate(feature_cols):
        if f in dropped:
            continue
        keep.append(f)
        for j in range(i + 1, len(feature_cols)):
            if feature_cols[j] in dropped:
                continue
            val = corr[i, j]
            if not np.isnan(val) and abs(val) > threshold:
                dropped.add(feature_cols[j])
    logger.info(f'Correlation pruning dropped {len(dropped)} features. Kept {len(keep)} features.')
    return keep