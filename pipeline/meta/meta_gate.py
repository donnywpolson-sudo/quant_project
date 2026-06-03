import numpy as np
from sklearn.linear_model import Ridge
from pipeline.common.config import config


def train_meta_model(
    train_X: np.ndarray,
    primary_pred_train: np.ndarray,
    meta_y_train: np.ndarray,
) -> Ridge:
    """
    Train a meta-label Ridge model on frozen features + primary model
    predictions to predict whether the primary model is correct.

    Args:
        train_X: frozen feature matrix (n_train, n_features).
        primary_pred_train: primary model's predicted direction (±1/0).
        meta_y_train: meta-label target (1=correct, 0=wrong, -1=abstain).

    Returns:
        Trained Ridge model.
    """
    meta_X = np.column_stack([
        train_X,
        primary_pred_train.reshape(-1, 1).astype(np.float32),
    ])
    mask = meta_y_train >= 0
    if mask.sum() < 20:
        return None
    model = Ridge(alpha=1.0, fit_intercept=True, random_state=config.SEED)
    model.fit(meta_X[mask], meta_y_train[mask].astype(np.float32))
    return model


def apply_meta_gate(
    df: "pl.DataFrame",
    meta_model: Ridge,
    test_X: np.ndarray,
    primary_pred_col: str = "primary_prediction",
    meta_threshold: float = 0.5,
) -> "pl.DataFrame":
    """
    Gate target_exec using meta-label model predictions.

    Zeroes out target_exec where meta-prob < meta_threshold,
    suppressing low-confidence trades.

    Args:
        df: DataFrame with primary_prediction and target_exec columns.
        meta_model: Trained Ridge for meta-labeling (None → pass-through).
        test_X: Test feature matrix.
        primary_pred_col: Column name for primary model directional prediction.
        meta_threshold: Minimum meta-prob to keep target_exec non-zero.

    Returns:
        df with gated target_exec and new meta_prob column.
    """
    import polars as pl

    if meta_model is None:
        return df.with_columns(pl.Series('meta_prob', np.zeros(df.height, dtype=np.float32)))

    primary_pred = df[primary_pred_col].to_numpy().astype(np.float32)
    meta_X = np.column_stack([
        test_X,
        primary_pred.reshape(-1, 1).astype(np.float32),
    ])
    meta_probs = meta_model.predict(meta_X).astype(np.float32)
    meta_probs = np.clip(meta_probs, 0.05, 0.95)

    if 'target_exec' in df.columns:
        gate_mask = meta_probs >= meta_threshold
        target_exec = df['target_exec'].to_numpy().astype(np.float32)
        target_exec[~gate_mask] = 0.0
        df = df.with_columns(
            pl.Series('target_exec', target_exec).cast(pl.Float32)
        )

    return df.with_columns(
        pl.Series('meta_prob', meta_probs).cast(pl.Float32)
    )
