import logging

import polars as pl

from pipeline.common.config import config

logger = logging.getLogger(__name__)


def add_target_15m(df: pl.DataFrame) -> pl.DataFrame:
    """
    Primary supervised label.

    On 1-minute bars, signal at bar t enters at open[t+1] and exits at
    open[t+16]. The regression label is:
        log(open[t+16] / open[t+1])
    That makes the label executable under execute_at=open[t+1] and avoids
    close[t] -> open[t+1] gap leakage.
    """
    horizon = int(getattr(config, "TARGET_15M_HORIZON", 15))
    eps = getattr(config, "EPS", 1e-9)
    entry_open = pl.col("open").shift(-1)
    exit_open = pl.col("open").shift(-(horizon + 1))
    forward_log_ret = (exit_open / entry_open.clip(eps, None)).log()

    multiplier = pl.col("contract_multiplier") if "contract_multiplier" in df.columns else pl.lit(1.0)
    tx_cost = float(getattr(config, "TX_COST_PER_ROUNDTURN", 0.0) or 0.0)
    commission = float(getattr(config, "COMMISSION_PER_CONTRACT", 0.0) or 0.0)
    commission_ret = (2.0 * commission) / (entry_open * multiplier).clip(eps, None)
    trade_threshold = pl.lit(tx_cost) + commission_ret

    return df.with_columns(
        [
            forward_log_ret.alias("target_15m_ret"),
            (forward_log_ret > 0).cast(pl.Int8).alias("target_15m_dir"),
            (
                pl.when(forward_log_ret > trade_threshold)
                .then(1)
                .when(forward_log_ret < -trade_threshold)
                .then(-1)
                .otherwise(0)
                .cast(pl.Int8)
                .alias("target_15m_trade_class")
            ),
        ]
    )


def add_target_daily_regime(df: pl.DataFrame) -> pl.DataFrame:
    """
    Daily context/filter column, not the model label.

    Uses existing causal HTF daily trend if present. If HTF expansion is off,
    emit neutral regime so downstream schemas stay stable.
    """
    if "htf_daily_trend_slope_10" in df.columns:
        regime = (
            pl.when(pl.col("htf_daily_trend_slope_10") > 0)
            .then(1)
            .when(pl.col("htf_daily_trend_slope_10") < 0)
            .then(-1)
            .otherwise(0)
        )
    else:
        regime = pl.lit(0)
    return df.with_columns(regime.cast(pl.Int8).alias("target_daily_regime"))


def drop_incomplete_target(df: pl.DataFrame) -> pl.DataFrame:
    filter_expr = None
    for target_col in ("target_15m_ret", "target_15m_dir", "target_15m_trade_class"):
        if target_col in df.columns:
            null_count = df[target_col].null_count()
            if null_count == df.height:
                logger.warning(
                    "[TARGET] %s is entirely null (%d/%d rows) - skipping",
                    target_col,
                    null_count,
                    df.height,
                )
                continue
            col_filter = pl.col(target_col).is_not_null()
            filter_expr = col_filter if filter_expr is None else filter_expr & col_filter
    if filter_expr is None:
        return df
    before = df.height
    df = df.filter(filter_expr)
    if df.height == 0 and before > 0:
        raise RuntimeError(
            "DROPNULL KILL: drop_incomplete_target collapsed %d rows to 0. "
            "Check target columns for full-null cascade." % before
        )
    return df
