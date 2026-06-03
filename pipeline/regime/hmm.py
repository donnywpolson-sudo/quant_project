"""
hmm.py — 4-State Gaussian Hidden Markov Model for Regime Detection.

Architecture:
  - Input: 1-Hour OHLCV data (Detection Layer).
  - Model: 4-state Gaussian HMM with diagonal covariance.
  - Mapping: Post-fit state labeling via emission means/variances:
      0: Low Vol / Bullish   (high +mean, low var)
      1: Low Vol / Range     (near-zero mean, low var)
      2: High Vol / Correction (-mean, high var)
      3: High Vol / Expansion (+mean, high var)
  - Training: Periodic expanding window (weekly), NOT step-by-step.
  - Inference: Forward algorithm for recursive out-of-sample
    P(S_t | Data_{1:t}) with no look-ahead leakage.
  - Robustness: Fallback logging, stationarity checks, Z-score standardization.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
from numpy.linalg import LinAlgError
from scipy.special import logsumexp
import time as _time

try:
    from statsmodels.tsa.stattools import adfuller
    _HAS_ADFULLER = True
except ImportError:
    _HAS_ADFULLER = False

logger = logging.getLogger(__name__)

# ============================================================================
# Configuration
# ============================================================================


@dataclass
class HMMConfig:
    """Configuration for HMM regime detection."""

    n_states: int = 4
    n_iter: int = 100
    tol: float = 1e-4
    random_state: int = 42
    retrain_frequency_bars: int = 120  # ~weekly: 5d * 24h = 120 bars
    min_train_bars: int = 240  # minimum bars before first training
    covariance_type: str = "diag"  # diagonal covariance per state
    z_score_window: int = 60  # rolling window for Z-score standardization
    min_zscore_samples: int = 30  # minimum samples for Z-score
    fallback_default_regime: int = 1  # default to Low Vol / Range on failure
    max_retry_attempts: int = 3  # retries with different seeds on convergence failure
    feature_columns: tuple = field(
        default_factory=lambda: ("log_return", "volume_z", "range_pct",
                                 "skew_5", "momentum_10")
    )
    # Stationarity thresholds (ADF p-value must be below this)
    stationarity_pvalue_threshold: float = 0.05

    @property
    def state_labels(self) -> dict:
        """Human-readable labels for mapped states."""
        return {
            0: "LowVol_Bullish",
            1: "LowVol_Range",
            2: "HighVol_Correction",
            3: "HighVol_Expansion",
        }


# ============================================================================
# Core HMM Implementation (from scratch, no sklearn dependency)
# ============================================================================


class GaussianHMM:
    """
    Gaussian Hidden Markov Model with diagonal covariance.

    Implements:
      - Baum-Welch (EM) training with log-space forward-backward.
      - Forward algorithm for online recursive filtering.
      - Viterbi for most-likely state path.
    """

    def __init__(self, n_states: int = 4, random_state: int = 42,
                 tol: float = 1e-4, n_iter: int = 100):
        self.n_states = n_states
        self.random_state = random_state
        self.tol = tol
        self.n_iter = n_iter

        # Parameters to be learned
        self.startprob_: Optional[np.ndarray] = None   # (n_states,)
        self.transmat_: Optional[np.ndarray] = None     # (n_states, n_states)
        self.means_: Optional[np.ndarray] = None        # (n_states, n_features)
        self.covars_: Optional[np.ndarray] = None       # (n_states, n_features)

        self.n_features_: Optional[int] = None
        self._fitted = False

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------
    def _init_params(self, X: np.ndarray, rng: np.random.Generator) -> None:
        n_samples, n_features = X.shape
        self.n_features_ = n_features

        # Uniform initial distribution
        self.startprob_ = np.full(self.n_states, 1.0 / self.n_states)

        # Transition matrix: sticky diagonal (0.7 on diagonal, rest uniform)
        sticky = 0.7
        off_diag = (1.0 - sticky) / (self.n_states - 1)
        self.transmat_ = np.full((self.n_states, self.n_states), off_diag)
        np.fill_diagonal(self.transmat_, sticky)

        # Means: k-means-like initialization from data quantiles
        quantiles = np.linspace(0.1, 0.9, self.n_states)
        self.means_ = np.zeros((self.n_states, n_features))
        for f in range(n_features):
            self.means_[:, f] = np.quantile(X[:, f], quantiles)

        # Add small random perturbation
        self.means_ += rng.normal(0, 0.01, self.means_.shape) * np.std(X, axis=0)

        # Covariances: global variance as initial estimate
        global_var = np.var(X, axis=0) + 1e-6
        self.covars_ = np.tile(global_var, (self.n_states, 1))

    # ------------------------------------------------------------------
    # Observation log-likelihood (diagonal Gaussian)
    # ------------------------------------------------------------------
    def _compute_log_obs_likelihood(self, X: np.ndarray) -> np.ndarray:
        """Compute log P(X_t | S_t = i) for all t, i. Shape: (n_samples, n_states)."""
        n_samples = X.shape[0]
        log_prob = np.zeros((n_samples, self.n_states))

        for k in range(self.n_states):
            # Diagonal Gaussian log-likelihood
            diff = X - self.means_[k]  # (n_samples, n_features)
            cov = self.covars_[k] + 1e-10  # prevent log(0)
            # log N(x | mu, sigma^2) = -0.5 * [log(2*pi) + log(sigma^2) + (x-mu)^2/sigma^2]
            log_prob[:, k] = -0.5 * np.sum(
                np.log(2 * np.pi * cov) + (diff ** 2) / cov, axis=1
            )

        return log_prob

    # ------------------------------------------------------------------
    # Forward Algorithm
    # ------------------------------------------------------------------
    def _forward(self, log_frameprob: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Forward pass in log space.

        Args:
            log_frameprob: (n_samples, n_states) log observation probabilities.

        Returns:
            log_alpha: (n_samples, n_states) forward log-probabilities.
            log_likelihood: total log-likelihood.
        """
        n_samples = log_frameprob.shape[0]
        log_alpha = np.zeros((n_samples, self.n_states))
        log_startprob = np.log(self.startprob_ + 1e-300)
        log_transmat = np.log(self.transmat_ + 1e-300)

        # t = 0
        log_alpha[0] = log_startprob + log_frameprob[0]

        for t in range(1, n_samples):
            for j in range(self.n_states):
                # logsumexp over previous states + transition log-prob
                log_alpha[t, j] = logsumexp(
                    log_alpha[t - 1] + log_transmat[:, j]
                ) + log_frameprob[t, j]

            # Optional: rescale to prevent underflow (not strictly necessary in log space
            # but helps numerical stability for very long sequences)
            alpha_max = np.max(log_alpha[t])
            if alpha_max < -700:
                log_alpha[t] -= alpha_max

        log_likelihood = logsumexp(log_alpha[-1])
        return log_alpha, log_likelihood

    # ------------------------------------------------------------------
    # Backward Algorithm
    # ------------------------------------------------------------------
    def _backward(self, log_frameprob: np.ndarray) -> np.ndarray:
        """Backward pass in log space."""
        n_samples = log_frameprob.shape[0]
        log_beta = np.zeros((n_samples, self.n_states))
        log_transmat = np.log(self.transmat_ + 1e-300)

        # t = T-1 (log_beta = 0 = log(1))
        log_beta[-1] = 0.0

        for t in range(n_samples - 2, -1, -1):
            for i in range(self.n_states):
                log_beta[t, i] = logsumexp(
                    log_transmat[i] + log_frameprob[t + 1] + log_beta[t + 1]
                )

        return log_beta

    # ------------------------------------------------------------------
    # E-step: compute expected sufficient statistics
    # ------------------------------------------------------------------
    def _e_step(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns: gamma (state posteriors), xi (transition posteriors), log_lik."""
        n_samples = X.shape[0]
        log_frameprob = self._compute_log_obs_likelihood(X)
        log_alpha, log_lik = self._forward(log_frameprob)
        log_beta = self._backward(log_frameprob)

        # Gamma: log P(S_t = i | X) = log_alpha + log_beta - logsumexp over states
        log_gamma = log_alpha + log_beta
        log_gamma -= logsumexp(log_gamma, axis=1, keepdims=True)
        gamma = np.exp(log_gamma)

        # Xi: log P(S_t=i, S_{t+1}=j | X)
        log_xi = np.zeros((n_samples - 1, self.n_states, self.n_states))
        log_transmat = np.log(self.transmat_ + 1e-300)

        for t in range(n_samples - 1):
            for i in range(self.n_states):
                log_xi[t, i] = (
                    log_alpha[t, i]
                    + log_transmat[i]
                    + log_frameprob[t + 1]
                    + log_beta[t + 1]
                )
            log_xi[t] -= logsumexp(log_xi[t])

        xi = np.exp(log_xi)
        return gamma, xi, log_lik

    # ------------------------------------------------------------------
    # M-step: update parameters
    # ------------------------------------------------------------------
    def _m_step(self, X: np.ndarray, gamma: np.ndarray,
                xi: np.ndarray) -> None:
        n_samples, n_features = X.shape

        # Update startprob (use first gamma row, but can also average)
        self.startprob_ = gamma[0] / gamma[0].sum()

        # Update transmat
        xi_sum = xi.sum(axis=0)  # (n_states, n_states)
        self.transmat_ = xi_sum / xi_sum.sum(axis=1, keepdims=True)
        self.transmat_ = np.clip(self.transmat_, 1e-10, 1.0)
        self.transmat_ /= self.transmat_.sum(axis=1, keepdims=True)

        # Update means
        gamma_sum = gamma.sum(axis=0)  # (n_states,)
        self.means_ = (gamma.T @ X) / gamma_sum[:, np.newaxis]

        # Update covariances (diagonal)
        for k in range(self.n_states):
            diff = X - self.means_[k]
            weighted_sq = (gamma[:, k, np.newaxis] * (diff ** 2)).sum(axis=0)
            self.covars_[k] = weighted_sq / gamma_sum[k]
            self.covars_[k] = np.clip(self.covars_[k], 1e-8, None)

    # ------------------------------------------------------------------
    # Fit (Baum-Welch EM)
    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray) -> "GaussianHMM":
        """
        Fit the HMM via Baum-Welch EM.

        Args:
            X: (n_samples, n_features) observation matrix.

        Returns:
            self
        """
        X = np.asarray(X, dtype=np.float64)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        n_samples, n_features = X.shape
        if n_samples < self.n_states * 3:
            raise ValueError(
                f"Insufficient data: {n_samples} samples for {self.n_states} states. "
                f"Need at least {self.n_states * 3} samples."
            )

        rng = np.random.default_rng(self.random_state)
        self._init_params(X, rng)

        prev_log_lik = -np.inf
        t0 = _time.perf_counter()
        for iteration in range(self.n_iter):
            gamma, xi, log_lik = self._e_step(X)

            # Convergence check
            delta = abs(log_lik - prev_log_lik)
            if delta < self.tol:
                logger.debug(f"HMM converged at iteration {iteration + 1}, log_lik={log_lik:.2f}")
                break

            prev_log_lik = log_lik
            self._m_step(X, gamma, xi)

            # Per-iteration timing: use print() so it's visible even with WARNING log level
            if iteration > 0 and iteration % 10 == 0:
                elapsed = _time.perf_counter() - t0
                print(
                    f'[HMM-TIMING] iter={iteration + 1}/{self.n_iter} delta={delta:.6f} elapsed={elapsed:.1f}s',
                    flush=True,
                )
        else:
            elapsed = _time.perf_counter() - t0
            print(
                f'[HMM-TIMING] WARNING: did not converge in {self.n_iter} iterations '
                f'(last delta={delta:.6f}, elapsed={elapsed:.1f}s)',
                flush=True,
            )

        self._fitted = True
        return self

    # ------------------------------------------------------------------
    # Forward filtering (out-of-sample, recursive)
    # ------------------------------------------------------------------
    def forward_filter(self, X: np.ndarray) -> np.ndarray:
        """
        Compute filtered state probabilities P(S_t | Data_{1:t}) recursively.

        This is the key inference method: it processes one observation at a time
        using only past data, ensuring NO look-ahead bias.

        Args:
            X: (n_samples, n_features) observation matrix.

        Returns:
            posteriors: (n_samples, n_states) filtered probabilities.
        """
        if not self._fitted:
            raise RuntimeError("Model must be fitted before calling forward_filter().")

        X = np.asarray(X, dtype=np.float64)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        n_samples = X.shape[0]
        log_startprob = np.log(self.startprob_ + 1e-300)
        log_transmat = np.log(self.transmat_ + 1e-300)
        posteriors = np.zeros((n_samples, self.n_states))

        # Initialize with stationary distribution (approximate with startprob)
        prev_log_prob = log_startprob.copy()

        for t in range(n_samples):
            # Log observation likelihood for current observation
            log_obs = np.zeros(self.n_states)
            for k in range(self.n_states):
                diff = X[t] - self.means_[k]
                cov = self.covars_[k] + 1e-10
                log_obs[k] = -0.5 * np.sum(
                    np.log(2 * np.pi * cov) + (diff ** 2) / cov
                )

            # Prediction step: log P(S_t | Data_{1:t-1}) = logsumexp(prev + log_trans)
            log_pred = np.zeros(self.n_states)
            for j in range(self.n_states):
                log_pred[j] = logsumexp(prev_log_prob + log_transmat[:, j])

            # Update step: log P(S_t | Data_{1:t}) = log_pred + log_obs, then normalize
            log_post = log_pred + log_obs
            log_post -= logsumexp(log_post)
            posteriors[t] = np.exp(log_post)

            prev_log_prob = log_post

        return posteriors

    # ------------------------------------------------------------------
    # Viterbi decoding
    # ------------------------------------------------------------------
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Viterbi decoding for the most likely state sequence."""
        if not self._fitted:
            raise RuntimeError("Model not fitted.")

        X = np.asarray(X, dtype=np.float64)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        n_samples = X.shape[0]
        log_frameprob = self._compute_log_obs_likelihood(X)
        log_startprob = np.log(self.startprob_ + 1e-300)
        log_transmat = np.log(self.transmat_ + 1e-300)

        log_delta = np.zeros((n_samples, self.n_states))
        psi = np.zeros((n_samples, self.n_states), dtype=np.int32)

        log_delta[0] = log_startprob + log_frameprob[0]

        for t in range(1, n_samples):
            for j in range(self.n_states):
                temp = log_delta[t - 1] + log_transmat[:, j]
                psi[t, j] = np.argmax(temp)
                log_delta[t, j] = temp[psi[t, j]] + log_frameprob[t, j]

        state_sequence = np.zeros(n_samples, dtype=np.int32)
        state_sequence[-1] = np.argmax(log_delta[-1])
        for t in range(n_samples - 2, -1, -1):
            state_sequence[t] = psi[t + 1, state_sequence[t + 1]]

        return state_sequence


# ============================================================================
# Feature Engineering for HMM
# ============================================================================


def _compute_hmm_features(df_1h: "pl.DataFrame") -> np.ndarray:
    """
    Compute stationary, Z-score standardized features from 1H data.

    Returns:
        features: (n_samples, 5) numpy array with columns:
            [log_return, volume_z, range_pct, skew_5, momentum_10]
    """
    import polars as pl

    eps = 1e-9

    # 1. Log return
    log_close = pl.col("close").log()
    log_return = (log_close - log_close.shift(1)).fill_null(0.0)

    # 2. Volume Z-score (rolling, strictly past data via shift(1))
    vol_lagged = pl.col("volume").shift(1)
    vol_mean = vol_lagged.rolling_mean(window_size=20, min_periods=5)
    vol_std = vol_lagged.rolling_std(window_size=20, min_periods=5)
    volume_z = ((pl.col("volume") - vol_mean) / (vol_std + eps)).fill_null(0.0)

    # 3. Range percentage
    range_pct = ((pl.col("high") - pl.col("low")) / (pl.col("close") + eps)).fill_null(0.0)

    # 4. Skewness (rolling 5-bar)
    log_ret = log_close - log_close.shift(1)
    ret_mean_5 = log_ret.rolling_mean(window_size=5, min_periods=3)
    ret_std_5 = log_ret.rolling_std(window_size=5, min_periods=3)
    ret_z3 = ((log_ret - ret_mean_5) / (ret_std_5 + eps)) ** 3
    skew_5 = ret_z3.rolling_mean(window_size=5, min_periods=3).fill_null(0.0)

    # 5. Momentum (10-bar)
    momentum_10 = (log_close - log_close.shift(10)).fill_null(0.0)

    df_feat = df_1h.select([
        log_return.alias("log_return"),
        volume_z.alias("volume_z"),
        range_pct.alias("range_pct"),
        skew_5.alias("skew_5"),
        momentum_10.alias("momentum_10"),
    ])

    # Convert to numpy
    features = df_feat.to_numpy().astype(np.float64)
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    features = np.clip(features, -50.0, 50.0)

    return features


# ============================================================================
# Stationarity Check
# ============================================================================


def _check_stationarity(features: np.ndarray, threshold: float = 0.05) -> bool:
    """
    Check if all feature columns are stationary using ADF test.

    Returns True only if ALL columns pass the ADF test (p-value < threshold).
    On import failure (no statsmodels), logs a warning and returns True (skip check).
    """
    if not _HAS_ADFULLER:
        logger.warning(
            "statsmodels not available; skipping stationarity checks. "
            "Install with: pip install statsmodels"
        )
        return True

    n_samples, n_features = features.shape
    if n_samples < 30:
        logger.warning(f"Too few samples ({n_samples}) for ADF test; skipping.")
        return False

    all_stationary = True
    for f in range(n_features):
        try:
            result = adfuller(features[:, f], maxlag=min(20, n_samples // 4),
                              autolag="AIC")
            pvalue = result[1]
            if pvalue >= threshold:
                logger.warning(
                    f"Feature column {f} is non-stationary "
                    f"(ADF p-value={pvalue:.4f} >= {threshold}). "
                    f"Consider differencing or alternative transforms."
                )
                all_stationary = False
        except Exception as e:
            logger.warning(f"ADF test failed for feature {f}: {e}")
            all_stationary = False

    return all_stationary


# ============================================================================
# Z-Score Standardization (Expanding Window)
# ============================================================================


def _zscore_standardize(features: np.ndarray, window: int = 60,
                        min_samples: int = 30) -> np.ndarray:
    """
    Standardize features using expanding-window Z-score normalization.

    This avoids look-ahead by using only the mean/std up to time t.
    The first ``window`` observations use simple expanding statistics;
    after that, a rolling window of ``window`` is used for stability.

    Args:
        features: (n_samples, n_features) raw features.
        window: rolling window size for statistics.
        min_samples: minimum samples before computing Z-score.

    Returns:
        standardized: (n_samples, n_features) Z-scored features.
    """
    n_samples, n_features = features.shape
    standardized = np.zeros_like(features)
    eps = 1e-9

    # For the first min_samples, forward-fill with the first valid value
    for t in range(n_samples):
        start = max(0, t - window + 1)
        end = t
        seg = features[start:end]

        if seg.shape[0] < min_samples:
            standardized[t] = 0.0
            continue

        mean = seg.mean(axis=0)
        std = seg.std(axis=0)
        std = np.where(std < eps, 1.0, std)
        standardized[t] = (features[t] - mean) / std

    return standardized


# ============================================================================
# HMMRegimeDetector — Main Public Class
# ============================================================================


class HMMRegimeDetector:
    """
    Detects market regimes using a 4-state Gaussian HMM on 1-Hour data.

    The detector is designed for periodic retraining (e.g., weekly) and
    provides online filtered state probabilities via forward recursion
    (no look-ahead). Post-fit state mapping labels each latent state
    according to its emission characteristics.

    Usage::

        detector = HMMRegimeDetector(config=HMMConfig())
        detector.fit(df_1h_init)                         # initial training
        probs = detector.filter(df_1h_new)               # online inference
        df_5min = detector.bridge_to_5min(probs_1h, df_5min)  # dual-TF bridge
    """

    def __init__(self, config: Optional[HMMConfig] = None):
        self.config = config or HMMConfig()
        self.hmm: Optional[GaussianHMM] = None
        self._state_map: Optional[dict] = None  # latent_idx -> canonical regime
        self._last_train_idx: int = -1  # bar index of last training
        self._bars_since_train: int = 0
        self._train_history: list = []  # diagnostic log
        self._fallback_triggered: bool = False

        # Stored feature params for online Z-scoring
        self._feature_mean: Optional[np.ndarray] = None
        self._feature_std: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # State Mapping (post-fit)
    # ------------------------------------------------------------------
    def _map_states(self) -> dict:
        """
        Map latent HMM states to canonical regimes using emission parameters.

        Rules:
          - High mean (> 0.3 * max_abs_mean) + low var (< median var) → LowVol_Bullish (0)
          - Low abs mean (< 0.2 * max_abs_mean) + low var (< median var) → LowVol_Range (1)
          - Neg mean (< -0.1 * max_abs_mean) + high var (> median var) → HighVol_Correction (2)
          - Pos mean (> 0.1 * max_abs_mean) + high var (> median var) → HighVol_Expansion (3)
          - Remaining states assigned by best-fit criteria.
        """
        means = self.hmm.means_[:, 0]  # log_return emission mean
        vars_ = self.hmm.covars_[:, 0]  # log_return emission variance

        max_abs_mean = max(abs(means).max(), 1e-6)
        median_var = np.median(vars_)

        assigned = set()
        mapping = {}

        # Priority 1: LowVol_Bullish — high positive mean, low variance
        candidates_bull = [
            i for i in range(self.config.n_states)
            if means[i] > 0.3 * max_abs_mean and vars_[i] < median_var
        ]
        if candidates_bull:
            # Pick strongest positive mean
            idx = max(candidates_bull, key=lambda i: means[i])
            mapping[idx] = 0  # LowVol_Bullish
            assigned.add(idx)

        # Priority 2: LowVol_Range — near-zero mean, low variance
        candidates_range = [
            i for i in range(self.config.n_states)
            if i not in assigned
            and abs(means[i]) < 0.2 * max_abs_mean
            and vars_[i] < median_var
        ]
        if candidates_range:
            idx = min(candidates_range, key=lambda i: abs(means[i]))
            mapping[idx] = 1  # LowVol_Range
            assigned.add(idx)

        # Priority 3: HighVol_Correction — negative mean, high variance
        candidates_corr = [
            i for i in range(self.config.n_states)
            if i not in assigned
            and means[i] < -0.1 * max_abs_mean
            and vars_[i] > median_var
        ]
        if candidates_corr:
            idx = min(candidates_corr, key=lambda i: means[i])  # most negative
            mapping[idx] = 2  # HighVol_Correction
            assigned.add(idx)

        # Priority 4: HighVol_Expansion — positive mean, high variance
        candidates_exp = [
            i for i in range(self.config.n_states)
            if i not in assigned
            and means[i] > 0.1 * max_abs_mean
            and vars_[i] > median_var
        ]
        if candidates_exp:
            idx = max(candidates_exp, key=lambda i: means[i])
            mapping[idx] = 3  # HighVol_Expansion
            assigned.add(idx)

        # Remaining unassigned: assign by best-fit criteria
        remaining = [i for i in range(self.config.n_states) if i not in assigned]
        remaining_needed = [r for r in [0, 1, 2, 3] if r not in set(mapping.values())]

        for needed_regime in remaining_needed:
            if not remaining:
                break
            if needed_regime == 0:  # Bullish
                idx = max(remaining, key=lambda i: means[i] - vars_[i])
            elif needed_regime == 1:  # Range
                idx = min(remaining, key=lambda i: abs(means[i]))
            elif needed_regime == 2:  # Correction
                idx = min(remaining, key=lambda i: means[i])
            else:  # Expansion
                idx = max(remaining, key=lambda i: means[i] + vars_[i])
            mapping[idx] = needed_regime
            assigned.add(idx)
            remaining.remove(idx)

        # Final fallback: any still-unassigned
        for i in range(self.config.n_states):
            if i not in mapping:
                mapping[i] = self.config.fallback_default_regime

        logger.info(
            f"HMM state mapping: {mapping} | "
            f"means={dict(zip(range(self.config.n_states), np.round(means, 6)))} | "
            f"vars={dict(zip(range(self.config.n_states), np.round(vars_, 8)))}"
        )
        return mapping

    # ------------------------------------------------------------------
    # Fit (initial training)
    # ------------------------------------------------------------------
    def fit(self, df_1h: "pl.DataFrame") -> bool:
        """
        Initial fit of the HMM on 1-hour data.

        Args:
            df_1h: Polars DataFrame with 1H OHLCV data.

        Returns:
            True if fit succeeded, False if fallback was triggered.
        """
        features = _compute_hmm_features(df_1h)
        print(
            f'[HMM-TIMING] step=hmm_features rows={features.shape[0]} cols={features.shape[1]} '
            f'nan={int(np.sum(np.isnan(features)))} inf={int(np.sum(np.isinf(features)))} '
            f'unique_counts={[len(np.unique(features[:, i])) for i in range(features.shape[1])]}',
            flush=True,
        )

        if features.shape[0] < self.config.min_train_bars:
            logger.warning(
                f"Insufficient 1H bars ({features.shape[0]}) for HMM training. "
                f"Need >= {self.config.min_train_bars}. Using fallback."
            )
            self._fallback_triggered = True
            self._log_fallback("insufficient_bars", features.shape[0])
            return False

        # Stationarity check
        is_stationary = _check_stationarity(
            features, self.config.stationarity_pvalue_threshold
        )
        if not is_stationary:
            logger.warning(
                "Features are non-stationary. HMM regime detection may be unreliable. "
                "Proceeding with training but logging for monitoring."
            )
            self._log_fallback("non_stationary_features", None)

        # Z-score standardize
        features_std = _zscore_standardize(
            features, self.config.z_score_window, self.config.min_zscore_samples
        )

        # Fit with retry on convergence failure
        hmm = None
        for attempt in range(self.config.max_retry_attempts):
            seed = self.config.random_state + attempt * 100
            try:
                hmm = GaussianHMM(
                    n_states=self.config.n_states,
                    random_state=seed,
                    tol=self.config.tol,
                    n_iter=self.config.n_iter,
                )
                hmm.fit(features_std)
                self.hmm = hmm
                break
            except (ValueError, LinAlgError, RuntimeError) as e:
                logger.warning(
                    f"HMM training attempt {attempt + 1}/{self.config.max_retry_attempts} "
                    f"failed: {e}"
                )
                if attempt == self.config.max_retry_attempts - 1:
                    self._fallback_triggered = True
                    self._log_fallback("fit_failed_all_attempts", str(e))
                    return False

        # Map states
        self._state_map = self._map_states()
        self._last_train_idx = features.shape[0] - 1
        self._bars_since_train = 0

        self._train_history.append({
            "bars": features.shape[0],
            "log_lik": float(
                hmm._compute_log_obs_likelihood(features_std).sum()
            ) if hmm else None,
            "state_map": dict(self._state_map) if self._state_map else None,
        })

        logger.info(
            f"HMM trained successfully on {features.shape[0]} bars. "
            f"State map: {self._state_map}"
        )
        return True

    # ------------------------------------------------------------------
    # Periodic retrain check
    # ------------------------------------------------------------------
    def _should_retrain(self, n_new_bars: int) -> bool:
        """Check if retraining is needed based on bars since last training."""
        self._bars_since_train += n_new_bars
        return self._bars_since_train >= self.config.retrain_frequency_bars

    # ------------------------------------------------------------------
    # Filter (online forward recursion)
    # ------------------------------------------------------------------
    def filter(self, df_1h: "pl.DataFrame",
               retrain_if_needed: bool = False) -> np.ndarray:
        """
        Compute filtered regime probabilities P(S_t | Data_{1:t}).

        Uses forward algorithm recursively with NO future data — the
        probability at time t depends ONLY on observations through time t.

        Args:
            df_1h: Polars DataFrame with 1H OHLCV data.
            retrain_if_needed: If True, trigger retraining when the periodic
                               window threshold is reached.

        Returns:
            probs: (n_samples, 4) array of regime probabilities in canonical
                   order [LowVol_Bullish, LowVol_Range, HighVol_Correction,
                   HighVol_Expansion].
        """
        if self.hmm is None:
            logger.warning("HMM not fitted. Attempting to fit now.")
            success = self.fit(df_1h)
            if not success:
                # Return fallback uniform-ish probabilities
                n = df_1h.height
                probs = np.ones((n, self.config.n_states)) / self.config.n_states
                return probs

        # Check periodic retraining
        if retrain_if_needed and self._should_retrain(df_1h.height):
            logger.info(
                f"Periodic retrain triggered ({self._bars_since_train} bars "
                f"since last train)."
            )
            success = self.fit(df_1h)
            if not success:
                logger.warning("Retrain failed; using existing model.")

        # Compute features and standardize
        features = _compute_hmm_features(df_1h)
        features_std = _zscore_standardize(
            features, self.config.z_score_window, self.config.min_zscore_samples
        )

        # Forward filter
        raw_posteriors = self.hmm.forward_filter(features_std)

        # Remap from latent state index to canonical regime order
        probs = np.zeros((raw_posteriors.shape[0], self.config.n_states))
        for latent_idx, canonical_idx in self._state_map.items():
            probs[:, canonical_idx] = raw_posteriors[:, latent_idx]

        return probs

    # ------------------------------------------------------------------
    # Dual-Timeframe Bridge: 1H → 5min
    # ------------------------------------------------------------------
    def bridge_to_5min(
        self,
        hourly_probs: np.ndarray,
        df_5min: "pl.DataFrame",
        df_1h: "pl.DataFrame" = None,
    ) -> "pl.DataFrame":
        """
        Forward-fill hourly regime probabilities into 5-minute data.

        Each 5-minute bar gets the regime probability from the most recent
        1-hour bar (causal: no future information used). The regime
        probabilities are added as columns with prefix 'hmm_regime_'.

        Uses timestamp-based alignment when df_1h is provided (replacing a
        prior positional-indexing approach that misaligned on missing hours).
        Falls back to positional mapping when df_1h is None.

        Args:
            hourly_probs: (n_1h, 4) regime probabilities from filter().
            df_5min: 5-minute Polars DataFrame with 'ts_event' column.
            df_1h: 1-hour Polars DataFrame (used for timestamp alignment).

        Returns:
            df_5min with added columns: hmm_regime_0, ..., hmm_regime_3
        """
        import polars as pl

        n_hours = hourly_probs.shape[0]
        col_names = [f"hmm_regime_{i}" for i in range(self.config.n_states)]

        if df_1h is not None and df_1h.height > 0:
            # Timestamp-based causal alignment: probabilities from the 1H bar
            # starting at T are only available after that bar closes at T+1h.
            prob_cols = {
                col_names[i]: hourly_probs[:, i].astype(np.float32)
                for i in range(self.config.n_states)
            }
            ts_dtype = df_5min['ts_event'].dtype
            df_1h_probs = df_1h.select('ts_event').with_columns([
                pl.Series(name, prob_cols[name]) for name in col_names
            ]).with_columns(
                (pl.col('ts_event') + pl.duration(hours=1)).cast(ts_dtype).alias('ts_event')
            ).sort('ts_event')
            df_5min = df_5min.sort('ts_event').join_asof(
                df_1h_probs, on='ts_event', strategy='backward'
            )
            for col in col_names:
                if col in df_5min.columns:
                    df_5min = df_5min.with_columns(pl.col(col).fill_null(0.0))
            return df_5min

        # Fallback: positional mapping (legacy path when df_1h is unavailable)
        ts_5min = df_5min["ts_event"].to_numpy()
        ts_1h_floor = ts_5min.astype("datetime64[h]")

        unique_hours, inverse, counts = np.unique(
            ts_1h_floor, return_inverse=True, return_counts=True
        )

        n_unique_hours = len(unique_hours)
        if n_unique_hours > n_hours:
            logger.warning(
                f"5-min data has {n_unique_hours} unique hours but only "
                f"{n_hours} hourly prob rows. Truncating to available."
            )
            n_unique_hours = min(n_unique_hours, n_hours)

        aligned_probs = hourly_probs[-n_unique_hours:] if n_unique_hours <= n_hours else hourly_probs
        hour_index = np.clip(inverse, 0, aligned_probs.shape[0] - 1)
        probs_5min = aligned_probs[hour_index]

        for i in range(self.config.n_states):
            df_5min = df_5min.with_columns(
                pl.Series(col_names[i], probs_5min[:, i].astype(np.float32))
            )

        return df_5min

    # ------------------------------------------------------------------
    # Compute regime filter mask for execution
    # ------------------------------------------------------------------
    def compute_trade_gate(
        self,
        df_5min: "pl.DataFrame",
        allowed_regimes: Optional[list] = None,
        prohibited_regimes: Optional[list] = None,
    ) -> "pl.Series":
        """
        Generate a trade-gating mask from 5-min regime probabilities.

        Args:
            df_5min: 5-min DataFrame with hmm_regime_* columns.
            allowed_regimes: List of regime indices (0-3) where trading IS allowed.
                             Default: [0, 3] (LowVol_Bullish, HighVol_Expansion).
            prohibited_regimes: List of regime indices where trading is NOT allowed.
                                Takes precedence over allowed_regimes.

        Returns:
            Boolean Series: True where trading is allowed.
        """
        import polars as pl

        if allowed_regimes is None:
            allowed_regimes = [0, 3]  # default: only bullish and expansion
        if prohibited_regimes is None:
            prohibited_regimes = [2]  # default: no trading in correction

        # Get the dominant regime (highest probability) per bar
        prob_cols = [f"hmm_regime_{i}" for i in range(self.config.n_states)]
        available_cols = [c for c in prob_cols if c in df_5min.columns]

        if not available_cols:
            logger.warning("No hmm_regime_* columns found; returning all True.")
            return pl.Series("trade_gate", [True] * df_5min.height)

        # Argmax over probability columns
        probs_stacked = np.column_stack([
            df_5min[c].to_numpy() for c in available_cols
        ])
        dominant_regime = np.argmax(probs_stacked, axis=1)

        allow = np.ones(df_5min.height, dtype=bool)
        for regime in prohibited_regimes:
            if regime < len(available_cols):
                allow = allow & (dominant_regime != regime)

        # If allowed_regimes is not "all", further restrict
        if allowed_regimes:
            regime_allowed = np.zeros(df_5min.height, dtype=bool)
            for regime in allowed_regimes:
                if regime < len(available_cols):
                    regime_allowed = regime_allowed | (dominant_regime == regime)
            allow = allow & regime_allowed

        return pl.Series("trade_gate", allow)

    # ------------------------------------------------------------------
    # Fallback logging
    # ------------------------------------------------------------------
    def _log_fallback(self, reason: str, detail) -> None:
        """Log fallback trigger for monitoring and feature expansion."""
        entry = {
            "reason": reason,
            "detail": str(detail) if detail is not None else None,
            "fallback_regime": self.config.fallback_default_regime,
        }
        self._train_history.append({"fallback": True, **entry})
        logger.warning(
            f"HMM fallback triggered: {reason} | "
            f"detail={detail} | "
            f"default_regime={self.config.state_labels[self.config.fallback_default_regime]}"
        )

    @property
    def fallback_active(self) -> bool:
        return self._fallback_triggered
