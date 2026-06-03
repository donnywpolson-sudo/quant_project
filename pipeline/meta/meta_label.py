import polars as pl
from pipeline.common.config import config


def add_meta_label_target(
    df: pl.DataFrame,
    primary_prediction_col: str = 'primary_prediction',
    primary_target_col: str = 'target_tb',
) -> pl.DataFrame:
    """
    Meta-labeling target — predicts whether the primary model's
    directional forecast was correct.

    After the primary model (triple-barrier) assigns a direction
    (+1 / -1), the meta-label answers: "Would a trade in that
    direction have been profitable?"

    Label:
      1 — primary prediction direction matched realized outcome
      0 — primary prediction was wrong
     -1 — primary prediction was neutral (no trade)

    This is trained AFTER the primary model, using frozen features +
    the primary model's prediction as an additional input. The
    resulting meta-label probability gates the Ridge execution layer.

    PATCH 5: Vectorized — replaces O(n) Python for-loop with Polars
    expressions.
    """
    if primary_prediction_col not in df.columns:
        return df.with_columns(
            pl.lit(-1, dtype=pl.Int8).alias('target_meta')
        )

    if primary_target_col not in df.columns:
        return df.with_columns(
            pl.lit(-1, dtype=pl.Int8).alias('target_meta')
        )

    pred = pl.col(primary_prediction_col).cast(pl.Int8)
    actual = pl.col(primary_target_col).cast(pl.Int8)

    labels = (
        pl.when(pred == 0)
        .then(pl.lit(-1, dtype=pl.Int8))
        .when(pred == actual)
        .then(pl.lit(1, dtype=pl.Int8))
        .otherwise(pl.lit(0, dtype=pl.Int8))
    )

    return df.with_columns(labels.alias('target_meta'))
