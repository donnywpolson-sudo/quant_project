"""
hmm_filter.py — HMM Regime Filter Integration Layer.

Integrates HMM regime detection into the walkforward pipeline:
  - Runs 1H HMM inference in parallel with 5-min execution path.
  - Bridges regime probabilities to 5-min bars (forward-fill, causal).
  - Applies regime-aware trade gating in the execution layer.
  - Provides a drop-in `apply_hmm_filter()` entry point for walkforward.py.
  - Falls back gracefully when HMM fails (pass-through, log failure).
"""

import logging
from typing import Optional, Tuple

import numpy as np
import polars as pl

from pipeline.regime.hmm import HMMRegimeDetector, HMMConfig

logger = logging.getLogger(__name__)


class HMMRegimeFilter:
    """
    HMM Regime Filter for walkforward pipeline integration.

    Manages the dual-timeframe HMM detection lifecycle:
      1. Receives 1H data for regime detection.
      2. Computes filtered probabilities (forward algorithm, no look-ahead).
      3. Bridges to 5-min execution bars.
      4. Generates trade-gating masks for regime-aware execution.

    The filter tracks whether HMM is active, logs training history,
    and exposes fallback status for PSR validation.
    """

    def __init__(self, config: Optional[HMMConfig] = None):
        self.config = config or HMMConfig()
        self.detector = HMMRegimeDetector(self.config)
        self._initialized = False
        self._training_history: list = []
        self._last_1h_probs: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Initialize: fit HMM on historical 1H data before walkforward
    # ------------------------------------------------------------------
    def initialize(self, df_1h: pl.DataFrame) -> bool:
        """
        Fit the HMM on initial training 1H data.

        Call once before walkforward begins, using the first
        ``min_train_bars`` hours of data.

        Args:
            df_1h: Polars DataFrame with 1H OHLCV bars (initial training window).

        Returns:
            True if HMM fitted successfully, False if fallback active.
        """
        success = self.detector.fit(df_1h)
        self._initialized = success
        self._training_history.append({
            "phase": "initialize",
            "bars": df_1h.height,
            "success": success,
            "fallback": self.detector.fallback_active,
        })
        if not success:
            logger.warning(
                "HMM initialization failed. Regime filter will operate in "
                "pass-through mode (no regime gating)."
            )
        return success

    # ------------------------------------------------------------------
    # Process one fold: regime detect on 1H train data, bridge to 5min
    # ------------------------------------------------------------------
    def process_fold(
        self,
        df_1h_train: pl.DataFrame,
        df_1h_test: pl.DataFrame,
        df_5min_test: pl.DataFrame,
        retrain: bool = False,
    ) -> pl.DataFrame:
        """
        Process one walkforward fold with regime-aware filtering.

        Steps:
          1. Optionally retrain HMM on 1H training data.
          2. Compute filtered regime probabilities on 1H test data.
          3. Bridge regime probabilities to 5-min execution bars.
          4. Apply trade gating (regime-based execution mask).

        Args:
            df_1h_train: 1H training data for this fold (for periodic retraining).
            df_1h_test: 1H test data for this fold (for inference).
            df_5min_test: 5-min execution data for this fold (to be augmented).
            retrain: If True, retrain HMM on df_1h_train before filtering.

        Returns:
            df_5min_test augmented with hmm_regime_* columns and trade_gate.
        """
        # Retrain if requested
        if retrain and df_1h_train is not None and df_1h_train.height > 0:
            logger.info(
                f"Periodic HMM retrain on {df_1h_train.height} 1H bars."
            )
            success = self.detector.fit(df_1h_train)
            self._training_history.append({
                "phase": "retrain",
                "bars": df_1h_train.height,
                "success": success,
                "fallback": self.detector.fallback_active,
            })
            if not success:
                logger.warning("HMM retrain failed. Using existing model.")

        if not self._initialized and not self.detector.fallback_active:
            # Attempt to initialize on the combined train data
            init_df = df_1h_train if df_1h_train is not None else df_1h_test
            if init_df is not None and init_df.height >= self.config.min_train_bars:
                self.initialize(init_df)

        # Compute filtered probabilities on 1H test data
        if self.detector.hmm is not None:
            hourly_probs = self.detector.filter(df_1h_test, retrain_if_needed=False)
            self._last_1h_probs = hourly_probs
        else:
            # Fallback: uniform probabilities
            n = max(df_1h_test.height, 1)
            hourly_probs = np.ones((n, self.config.n_states)) / self.config.n_states
            self._last_1h_probs = hourly_probs

        # Bridge to 5-min — pass df_1h_test for timestamp-based alignment
        df_5min_test = self.detector.bridge_to_5min(
            hourly_probs, df_5min_test, df_1h_test
        )

        # Apply trade gate
        trade_gate = self.detector.compute_trade_gate(df_5min_test)
        df_5min_test = df_5min_test.with_columns(trade_gate.alias("hmm_trade_gate"))

        return df_5min_test

    # ------------------------------------------------------------------
    # Apply regime gating to execution signal
    # ------------------------------------------------------------------
    @staticmethod
    def apply_regime_gate(
        df_5min: pl.DataFrame,
        gate_column: str = "hmm_trade_gate",
    ) -> pl.DataFrame:
        """
        Zero out 'target_exec' where the HMM trade gate is False.

        This is called after the simulator computes target_exec but before
        position sizing, so the gate acts as a binary on/off for each bar.

        Args:
            df_5min: 5-min DataFrame with 'target_exec' and gate column.
            gate_column: Name of the boolean gate column.

        Returns:
            df_5min with gated target_exec.
        """
        if gate_column not in df_5min.columns:
            logger.warning(
                f"Gate column '{gate_column}' not found in DataFrame. "
                f"Skipping regime gate."
            )
            return df_5min

        df_5min = df_5min.with_columns(
            pl.when(pl.col(gate_column))
            .then(pl.col("target_exec"))
            .otherwise(pl.lit(0.0, dtype=pl.Float32))
            .alias("target_exec")
        )

        return df_5min

    # ------------------------------------------------------------------
    # Determine if HMM is contributing (for PSR comparison)
    # ------------------------------------------------------------------
    @property
    def is_active(self) -> bool:
        """True if HMM is actively filtering (not in fallback)."""
        return self._initialized and not self.detector.fallback_active

    @property
    def training_log(self) -> list:
        """Complete training/failure history for diagnostics."""
        return self._training_history

    @property
    def fallback_reason(self) -> Optional[str]:
        """Reason for fallback, if triggered."""
        if not self.detector.fallback_active:
            return None
        for entry in reversed(self._training_history):
            if entry.get("fallback"):
                return entry.get("reason")
        return "unknown"


# ============================================================================
# Standalone integration function for walkforward.py
# ============================================================================


def apply_hmm_filter(
    df_5min_base: pl.DataFrame,
    df_1h_train: Optional[pl.DataFrame],
    df_1h_test: pl.DataFrame,
    hmm_filter: Optional[HMMRegimeFilter] = None,
    config: Optional[HMMConfig] = None,
    retrain: bool = False,
) -> Tuple[pl.DataFrame, HMMRegimeFilter]:
    """
    Drop-in HMM regime filter for walkforward.py integration.

    Usage inside walkforward.py's process_fold()::

        from pipeline.regime.hmm_filter import apply_hmm_filter, HMMRegimeFilter

        hmm_filter = HMMRegimeFilter()
        hmm_filter.initialize(df_1h_init)

        # Inside process_fold, after base execution but before final PnL:
        df_result, hmm_filter = apply_hmm_filter(
            df_5min_base=df_executed,
            df_1h_train=df_1h_train,
            df_1h_test=df_1h_test,
            hmm_filter=hmm_filter,
            retrain=(fold_idx % 5 == 0),  # retrain every 5 folds
        )

    Args:
        df_5min_base: 5-min DataFrame after base execution (must have target_exec).
        df_1h_train: 1H training data for this fold.
        df_1h_test: 1H test data for this fold.
        hmm_filter: Existing filter instance (created on first call).
        config: HMM configuration (only used if hmm_filter is None).
        retrain: Whether to retrain HMM on this fold.

    Returns:
        (df_5min_gated, hmm_filter): Gated 5-min DataFrame and updated filter.
    """
    if hmm_filter is None:
        hmm_filter = HMMRegimeFilter(config)

    # If not yet initialized, do initial fit on training data
    if not hmm_filter._initialized and df_1h_train is not None:
        hmm_filter.initialize(df_1h_train)

    # Process the fold: bridge regime probs to 5-min and gate signal
    df_gated = hmm_filter.process_fold(
        df_1h_train=df_1h_train,
        df_1h_test=df_1h_test,
        df_5min_test=df_5min_base,
        retrain=retrain,
    )

    # Apply the regime gate (zero out target_exec where gate is False)
    df_gated = hmm_filter.apply_regime_gate(df_gated)

    return df_gated, hmm_filter