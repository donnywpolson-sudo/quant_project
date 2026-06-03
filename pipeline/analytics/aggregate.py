import json
import polars as pl
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr
from pipeline.common.config import config

# Number of 5-minute bars in a trading year (~23h * 12 bars/h * 252 trading days)
# Overridable via config.ANNUAL_FACTOR if set before aggregate import.
ANNUAL_FACTOR = 69552
RISK_FREE_RATE = 0.0

# Column projection for loading backtest results: only read columns that
# are actually used downstream.  Eliminates SELECT * on parquet reads.
_BACKTEST_COLUMNS_OF_INTEREST = [
    'ts_event', 'pnl', 'position', 'benchmark_pnl',
    'prediction_prob', 'ret_exec', 'equity_curve',
    'return_on_equity', 'drawdown_pct',
]


def _bars_per_year() -> int:
    return int(getattr(config, 'ANNUAL_FACTOR', ANNUAL_FACTOR))


def _sharpe_pair(pnl: np.ndarray, bars_per_year: int) -> tuple[float, float]:
    if len(pnl) == 0:
        return 0.0, 0.0
    std = float(np.std(pnl))
    if std <= 1e-12:
        return 0.0, 0.0
    sharpe_per_bar = float(np.mean(pnl) / std)
    return sharpe_per_bar, float(sharpe_per_bar * np.sqrt(bars_per_year))


def _position_hit_turnover_metrics(
    pnl: np.ndarray,
    positions: np.ndarray | None,
    ret_exec: np.ndarray | None,
) -> dict:
    n = len(pnl)
    out = {
        'position_turnover': 0.0,
        'position_turnover_per_bar': 0.0,
        'position_change_events': 0,
        'bar_hit_rate_all_bars': 0.0,
        'bar_hit_rate_all_bars_n': n,
        'bar_hit_rate_active_bars': 0.0,
        'bar_hit_rate_active_bars_n': 0,
        'trade_hit_rate': None,
        'trade_hit_rate_n': 0,
    }
    if positions is None or len(positions) == 0:
        return out

    positions = np.asarray(positions, dtype=np.float64)
    pos_changes = np.abs(np.diff(positions, prepend=positions[0]))
    out['position_turnover'] = float(pos_changes.sum())
    out['position_turnover_per_bar'] = float(pos_changes.sum() / max(n, 1))
    out['position_change_events'] = int(np.sum(pos_changes > 1e-9))

    if ret_exec is not None and len(ret_exec) == n:
        ret_exec = np.asarray(ret_exec, dtype=np.float64)
        active = np.abs(positions) > 1e-12
        correct = np.zeros(n, dtype=bool)
        correct[active] = np.sign(positions[active]) == np.sign(ret_exec[active])
        out['bar_hit_rate_all_bars'] = float(correct.sum() / max(n, 1))
        out['bar_hit_rate_active_bars_n'] = int(active.sum())
        if active.sum() > 0:
            out['bar_hit_rate_active_bars'] = float(correct[active].mean())

    trade_pnl = []
    in_trade = False
    start = 0
    current_pos = 0.0
    for i, pos in enumerate(positions):
        if not in_trade and abs(pos) > 1e-12:
            in_trade = True
            start = i
            current_pos = pos
        elif in_trade and (abs(pos) <= 1e-12 or np.sign(pos) != np.sign(current_pos)):
            trade_pnl.append(float(np.sum(pnl[start:i])))
            in_trade = abs(pos) > 1e-12
            start = i
            current_pos = pos
    if in_trade:
        trade_pnl.append(float(np.sum(pnl[start:])))
    if trade_pnl:
        wins = np.sum(np.asarray(trade_pnl) > 0.0)
        out['trade_hit_rate'] = float(wins / len(trade_pnl))
        out['trade_hit_rate_n'] = int(len(trade_pnl))
    return out


def compute_backtest_metrics(df: pl.DataFrame) -> dict:
    ret_exec = df['ret_exec'] if 'ret_exec' in df.columns else None
    return compute_pro_metrics(
        df['pnl'],
        df['position'] if 'position' in df.columns else None,
        df['benchmark_pnl'] if 'benchmark_pnl' in df.columns else None,
        predictions_series=df['prediction_prob'].shift(1) if 'prediction_prob' in df.columns else None,
        targets_series=ret_exec,
        ret_exec_series=ret_exec,
    )


def _json_sanitize(value):
    if isinstance(value, dict):
        return {str(k): _json_sanitize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_sanitize(v) for v in value]
    if isinstance(value, tuple):
        return [_json_sanitize(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value


def build_metrics_report(df: pl.DataFrame, *, context: dict | None = None) -> dict:
    """Step 11 boundary: risk/alpha/execution metrics report."""
    required = ['ts_event', 'pnl', 'position', 'prediction_prob', 'ret_exec']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f'METRICS FAIL: missing columns {missing}')
    if df.height == 0:
        raise RuntimeError('METRICS FAIL: empty backtest result')

    metrics = compute_backtest_metrics(df)
    pnl = df['pnl'].to_numpy().astype(np.float64)
    pred = df['prediction_prob'].to_numpy().astype(np.float64)
    pos = df['position'].to_numpy().astype(np.float64)
    ret_exec = df['ret_exec'].to_numpy().astype(np.float64)

    if not np.isfinite(pnl).all():
        raise RuntimeError('METRICS FAIL: non-finite pnl')
    if not np.isfinite(pred).all():
        raise RuntimeError('METRICS FAIL: non-finite prediction_prob')
    if not np.isfinite(pos).all():
        raise RuntimeError('METRICS FAIL: non-finite position')
    if not np.isfinite(ret_exec).all():
        raise RuntimeError('METRICS FAIL: non-finite ret_exec')

    bars_per_year = int(metrics['bars_per_year'])
    gross_metrics = None
    cost_drag = None
    if 'gross_pnl' in df.columns:
        gross = df['gross_pnl'].to_numpy().astype(np.float64)
        if not np.isfinite(gross).all():
            raise RuntimeError('METRICS FAIL: non-finite gross_pnl')
        gross_sharpe_per_bar, gross_sharpe_ann = _sharpe_pair(gross, bars_per_year)
        gross_metrics = {
            'total_pnl': round(float(gross.sum()), 6),
            'sharpe_per_bar': round(gross_sharpe_per_bar, 8),
            'sharpe_annualized': round(gross_sharpe_ann, 3),
        }
        cost_drag = round(float(gross.sum() - pnl.sum()), 6)

    pred_dist = {
        'min': round(float(pred.min()), 8),
        'max': round(float(pred.max()), 8),
        'mean': round(float(pred.mean()), 8),
        'std': round(float(pred.std()), 8),
        'gt_0_55_frac': round(float((pred > 0.55).mean()), 8),
        'lt_0_45_frac': round(float((pred < 0.45).mean()), 8),
    }
    active = np.abs(pos) > 1e-12
    position_dist = {
        'active_bars': int(active.sum()),
        'active_frac': round(float(active.mean()), 8),
        'long_bars': int((pos > 1e-12).sum()),
        'short_bars': int((pos < -1e-12).sum()),
        'flat_bars': int((np.abs(pos) <= 1e-12).sum()),
        'max_abs_position': round(float(np.abs(pos).max()), 8),
    }

    report = {
        'status': 'OK',
        'context': context or {},
        'rows': df.height,
        'ts_min': str(df['ts_event'].min()),
        'ts_max': str(df['ts_event'].max()),
        'net': metrics,
        'gross': gross_metrics,
        'cost_drag': cost_drag,
        'prediction_distribution': pred_dist,
        'position_distribution': position_dist,
        'diagnostics': {
            'spearman_ic': metrics.get('spearman_ic'),
            'spearman_ic_pvalue': metrics.get('spearman_ic_pvalue'),
            'bar_hit_rate_active_bars': metrics.get('bar_hit_rate_active_bars'),
            'bar_hit_rate_active_bars_n': metrics.get('bar_hit_rate_active_bars_n'),
            'position_turnover': metrics.get('position_turnover'),
            'position_change_events': metrics.get('position_change_events'),
            'trade_hit_rate': metrics.get('trade_hit_rate'),
            'trade_hit_rate_n': metrics.get('trade_hit_rate_n'),
        },
    }
    print(
        f'[METRICS] rows={df.height} net_sharpe={metrics["sharpe_annualized"]:.3f} '
        f'pnl={metrics["total_pnl"]:.2f} ic={metrics.get("spearman_ic")} '
        f'turnover={metrics.get("position_turnover")}',
        flush=True,
    )
    return _json_sanitize(report)


def compute_ic(predictions: pl.Series, targets: pl.Series) -> dict:
    """Compute Information Coefficient using Spearman rank correlation.

    Args:
        predictions: Model prediction probabilities (or any continuous signal).
        targets: Realized forward returns or target labels.

    Returns:
        dict with 'spearman_ic' (float) rounded to 4 decimals, or None if
        the computation fails (e.g., constant arrays).
    """
    try:
        pred = predictions.to_numpy().astype(np.float64)
        targ = targets.to_numpy().astype(np.float64)
        mask = np.isfinite(pred) & np.isfinite(targ)
        if mask.sum() < 3:
            return {'spearman_ic': None}
        corr, pvalue = spearmanr(pred[mask], targ[mask])
        return {
            'spearman_ic': round(float(corr), 4),
            'spearman_ic_pvalue': round(float(pvalue), 6),
        }
    except Exception:
        return {'spearman_ic': None}


def compute_pro_metrics(
    pnl_series: pl.Series,
    positions_series: pl.Series = None,
    benchmark_series: pl.Series = None,
    predictions_series: pl.Series = None,
    targets_series: pl.Series = None,
    ret_exec_series: pl.Series = None,
) -> dict:
    pnl = pnl_series.to_numpy().astype(np.float64)
    eps = 1e-12
    bars_per_year = _bars_per_year()

    # --- Core PnL metrics ---
    total_pnl = float(pnl.sum())

    # PnL is dollar-scaled. Percent returns must be normalized by account equity.
    avg_pnl = float(pnl.mean())
    std_pnl = float(pnl.std())
    sharpe_per_bar, sharpe = _sharpe_pair(pnl, bars_per_year)

    # --- Sortino ratio ---
    downside = pnl[pnl < 0]
    if len(downside) > 0:
        downside_std = float(downside.std())
        sortino = (avg_pnl / (downside_std + eps)) * np.sqrt(bars_per_year)
    else:
        sortino = np.inf if avg_pnl > 0 else 0.0

    # --- Drawdown ---
    cum_pnl = np.cumsum(pnl)
    running_max = np.maximum.accumulate(cum_pnl)
    drawdown = cum_pnl - running_max
    max_drawdown = float(drawdown.min())
    starting_equity = float(getattr(config, 'EQUITY', 100000.0))
    if starting_equity <= 0:
        starting_equity = 100000.0
    equity_curve = starting_equity + cum_pnl
    running_equity_max = np.maximum.accumulate(equity_curve)
    drawdown_pct = equity_curve / np.maximum(running_equity_max, eps) - 1.0
    max_drawdown_pct = float(drawdown_pct.min()) if len(drawdown_pct) else 0.0
    total_return_on_equity = total_pnl / starting_equity
    total_return_pct = total_return_on_equity * 100.0

    # --- Calmar ratio ---
    annualized_return = avg_pnl * bars_per_year
    calmar = annualized_return / (abs(max_drawdown) + eps)
    annualized_return_on_equity = (avg_pnl * bars_per_year) / starting_equity
    calmar_ratio_pct = annualized_return_on_equity / (abs(max_drawdown_pct) + eps)

    # --- Trade / hit-rate statistics ---
    win_rate = 0.0
    avg_win = 0.0
    avg_loss = 0.0
    profit_factor = 0.0
    avg_holding_bars = 0.0
    positions = positions_series.to_numpy().astype(np.float64) if positions_series is not None and len(positions_series) > 0 else None
    ret_exec = ret_exec_series.to_numpy().astype(np.float64) if ret_exec_series is not None and len(ret_exec_series) == len(pnl) else None
    position_stats = _position_hit_turnover_metrics(pnl, positions, ret_exec)

    if positions is not None:
        pos_changes = np.abs(np.diff(positions, prepend=0))
        entry_idx = np.where(pos_changes > 0)[0]
        if len(entry_idx) > 1:
            holding = np.diff(entry_idx)
            avg_holding_bars = float(holding.mean())

        # Trade-level PnL attribution
        trade_pnl = []
        current_pos = 0
        entry_bar = 0
        cum_pnl_local = np.cumsum(pnl)
        for i in range(len(pnl)):
            if positions[i] != current_pos:
                if current_pos != 0:
                    trade_pnl.append(cum_pnl_local[i] - cum_pnl_local[entry_bar])
                current_pos = positions[i]
                entry_bar = i
        if current_pos != 0:
            trade_pnl.append(cum_pnl_local[-1] - cum_pnl_local[entry_bar])

        trade_pnl = np.array(trade_pnl, dtype=np.float32)
        if len(trade_pnl) > 0:
            gains = trade_pnl[trade_pnl > 0]
            losses = trade_pnl[trade_pnl < 0]
            win_rate = float(len(gains) / max(len(trade_pnl), 1))
            avg_win = float(gains.mean()) if len(gains) > 0 else 0.0
            avg_loss = float(losses.mean()) if len(losses) > 0 else 0.0
            if len(losses) > 0 and abs(losses.sum()) > eps:
                profit_factor = float(gains.sum() / abs(losses.sum()))
            else:
                profit_factor = np.inf if gains.sum() > 0 else 0.0

    # --- Benchmark metrics ---
    benchmark_sharpe = None
    benchmark_maxdd = None
    correlation = None
    if benchmark_series is not None:
        bench = benchmark_series.to_numpy().astype(np.float64)
        bench_avg = float(bench.mean())
        bench_std = float(bench.std())
        if bench_std > eps:
            benchmark_sharpe = (bench_avg / bench_std) * np.sqrt(bars_per_year)
        else:
            benchmark_sharpe = 0.0

        bench_cum = np.cumsum(bench)
        bench_max = np.maximum.accumulate(bench_cum)
        benchmark_maxdd = float((bench_cum - bench_max).min())

        if (
            len(pnl) == len(bench)
            and np.isfinite(pnl).all()
            and np.isfinite(bench).all()
            and float(np.std(pnl)) > eps
            and float(np.std(bench)) > eps
        ):
            try:
                corr_matrix = np.corrcoef(pnl, bench)
                correlation = float(corr_matrix[0, 1])
            except Exception:
                correlation = 0.0

    # --- Information Coefficient ---
    ic_result = {}
    if predictions_series is not None and targets_series is not None:
        ic_result = compute_ic(predictions_series, targets_series)

    return {
        'total_pnl': round(total_pnl, 6),
        'total_return_percent': round(total_return_pct, 4),
        'starting_equity': round(starting_equity, 2),
        'total_return_on_equity': round(total_return_on_equity, 6),
        'total_return_on_equity_percent': round(total_return_on_equity * 100.0, 4),
        'annualized_return_on_equity': round(annualized_return_on_equity, 6),
        'bars_per_year': bars_per_year,
        'sharpe_per_bar': round(sharpe_per_bar, 8),
        'sharpe_annualized': round(sharpe, 3) if np.isfinite(sharpe) else 0.0,
        'sortino_annualized': round(sortino, 3) if np.isfinite(sortino) else 0.0,
        'calmar_ratio': round(calmar, 3),
        'calmar_ratio_pct': round(calmar_ratio_pct, 3) if np.isfinite(calmar_ratio_pct) else 0.0,
        'max_drawdown': round(max_drawdown, 6),
        'max_drawdown_pct': round(max_drawdown_pct, 6),
        'bar_hit_rate_all_bars': round(position_stats['bar_hit_rate_all_bars'], 6),
        'bar_hit_rate_all_bars_n': int(position_stats['bar_hit_rate_all_bars_n']),
        'bar_hit_rate_active_bars': round(position_stats['bar_hit_rate_active_bars'], 6),
        'bar_hit_rate_active_bars_n': int(position_stats['bar_hit_rate_active_bars_n']),
        'trade_hit_rate': None if position_stats['trade_hit_rate'] is None else round(position_stats['trade_hit_rate'], 6),
        'trade_hit_rate_n': int(position_stats['trade_hit_rate_n']),
        'win_rate': round(win_rate, 4),
        'profit_factor': round(profit_factor, 4) if np.isfinite(profit_factor) else 'inf',
        'avg_win': round(avg_win, 8),
        'avg_loss': round(avg_loss, 8),
        'ratio_avg_win_loss': round(avg_win / (abs(avg_loss) + eps), 3),
        'number_of_trades': int(position_stats['position_change_events']),
        'position_change_events': int(position_stats['position_change_events']),
        'avg_holding_bars': round(avg_holding_bars, 1),
        'position_turnover': round(position_stats['position_turnover'], 6),
        'position_turnover_per_bar': round(position_stats['position_turnover_per_bar'], 8),
        'turnover': round(position_stats['position_turnover_per_bar'], 4),
        'spearman_ic': ic_result.get('spearman_ic'),
        'spearman_ic_pvalue': ic_result.get('spearman_ic_pvalue'),
        'benchmark_sharpe': round(benchmark_sharpe, 3) if benchmark_sharpe is not None else None,
        'benchmark_max_drawdown': round(benchmark_maxdd, 6) if benchmark_maxdd is not None else None,
        'correlation_with_benchmark': round(correlation, 4) if correlation is not None else None,
    }


def load_all_backtests(artifacts_root='output') -> dict:
    """Load all backtest parquet files with explicit column projection.

    Uses the ``columns`` parameter of ``pl.read_parquet`` to read only the
    six columns of interest rather than SELECT *, and filters to files that
    actually contain the required columns.
    """
    root = Path(artifacts_root)
    results = {}
    active_profile = str(getattr(config, 'ACTIVE_PROFILE', '') or '')
    profile_suffix = f'_{active_profile}' if active_profile else ''
    use_profile_filter = bool(
        profile_suffix and list(root.glob(f'*/*_split_*{profile_suffix}/backtest_results*.parquet'))
    )

    # Search for backtest_results*.parquet in split-aware directories:
    #   <root>/<market>/<year>_split_<idx>/backtest_results_hmm.parquet
    #   <root>/<market>/<year>_split_<idx>/backtest_results.parquet
    for f in sorted(root.glob('*/*_split_*/backtest_results*.parquet')):
        try:
            if use_profile_filter and not f.parent.name.endswith(profile_suffix):
                continue
            # Projection: read only columns we'll actually use downstream.
            # Polars will read the parquet footer to discover which of the
            # requested columns exist; missing columns are silently ignored
            # when ``columns`` is a list (pre-0.20 behaviour), so we verify
            # afterwards instead.
            df = pl.read_parquet(f, columns=_BACKTEST_COLUMNS_OF_INTEREST)

            if 'pnl' not in df.columns or 'ts_event' not in df.columns:
                continue

            keep = ['ts_event']
            for col in ['pnl', 'position', 'benchmark_pnl',
                        'prediction_prob', 'ret_exec', 'equity_curve',
                        'return_on_equity', 'drawdown_pct']:
                if col in df.columns:
                    keep.append(col)
            df = df.select(keep).sort('ts_event')
            market = f.parent.parent.name
            year_dir = f.parent.name  # e.g. "2024_split_1" or "2024"
            year = year_dir.split('_')[0] if '_split_' in year_dir else year_dir
            split_tag = year_dir if '_split_' in year_dir else None
            results.setdefault(market, []).append((year, df))
        except Exception:
            continue

    return results


def aggregate_market(dfs: list) -> pl.DataFrame:
    return pl.concat([df for _, df in dfs]).sort('ts_event')


def compute_year_breakdown(year_dfs: list) -> list:
    breakdown = []
    for year, df in year_dfs:
        predictions = df['prediction_prob'].shift(1) if 'prediction_prob' in df.columns else None
        targets = df['ret_exec'] if 'ret_exec' in df.columns else None
        metrics = compute_pro_metrics(
            df['pnl'],
            df['position'] if 'position' in df.columns else None,
            df['benchmark_pnl'] if 'benchmark_pnl' in df.columns else None,
            predictions_series=predictions,
            targets_series=targets,
            ret_exec_series=targets,
        )
        metrics['year'] = year
        breakdown.append(metrics)
    return breakdown


def _build_combined_pnl(results: dict) -> pl.Series:
    """
    Build a combined-market total-PnL series by concatenating all market
    PnL series and summing per-timestamp using group_by aggregation.

    This avoids N+1 join chains and the suffix/collision issues inherent
    in outer joins over 3+ frames.

    Returns a pl.Series of combined pnl values, or None if fewer than 2
    market series are available.
    """
    pnl_frames = []
    for market, year_dfs in results.items():
        market_df = aggregate_market(year_dfs).select(['ts_event', 'pnl'])
        pnl_frames.append(market_df)

    if len(pnl_frames) < 2:
        return None

    # Stack all market series and sum PnL per timestamp
    stacked = pl.concat(pnl_frames)
    combined = stacked.group_by('ts_event', maintain_order=True).agg(
        pl.col('pnl').sum().alias('total_pnl')
    )
    return combined['total_pnl']


def run_aggregation(artifacts_root='output'):
    root = Path(artifacts_root)
    output_dir = root / 'aggregated'
    output_dir.mkdir(parents=True, exist_ok=True)
    results = load_all_backtests(artifacts_root)
    if not results:
        print('No backtest results found.')
        return

    all_series = []

    for market, year_dfs in results.items():
        combined = aggregate_market(year_dfs)
        if combined.is_empty():
            continue
        predictions = combined['prediction_prob'].shift(1) if 'prediction_prob' in combined.columns else None
        targets = combined['ret_exec'] if 'ret_exec' in combined.columns else None
        metrics = compute_pro_metrics(
            combined['pnl'],
            combined['position'] if 'position' in combined.columns else None,
            combined['benchmark_pnl'] if 'benchmark_pnl' in combined.columns else None,
            predictions_series=predictions,
            targets_series=targets,
            ret_exec_series=targets,
        )
        metrics.update({
            'market': market,
            'num_years': len(year_dfs),
            'total_rows': combined.height,
            'years_breakdown': compute_year_breakdown(year_dfs),
        })
        out = output_dir / f'{market}_metrics.json'
        with open(out, 'w') as f:
            json.dump(metrics, f, indent=2)
        ic_str = (
            f"IC={metrics['spearman_ic']}"
            if metrics.get('spearman_ic') is not None
            else "IC=N/A"
        )
        print(
            f"Saved {out} | sharpe_annualized={metrics['sharpe_annualized']} "
            f"(bars_per_year={metrics['bars_per_year']}) | "
            f"bar_hit_rate_active={metrics['bar_hit_rate_active_bars']} "
            f"n={metrics['bar_hit_rate_active_bars_n']} | {ic_str} | "
            f"PnL={metrics['total_pnl']}"
        )
        all_series.append(combined['pnl'])

    # Combined multi-market report — single outer-join chain (eliminates N+1)
    if len(results) > 1:
        total_pnl = _build_combined_pnl(results)
        if total_pnl is not None:
            total_metrics = compute_pro_metrics(total_pnl)
            total_metrics.update({
                'description': 'Sum of all markets',
                'markets': sorted(results.keys()),
            })
            out = output_dir / 'all_markets.json'
            with open(out, 'w') as f:
                json.dump(total_metrics, f, indent=2)
            print(
                f"Saved combined report | sharpe_annualized={total_metrics['sharpe_annualized']} "
                f"(bars_per_year={total_metrics['bars_per_year']})"
            )


ANNUAL_FACTOR = 69552  # 23h * 12 bars/h * 252 days


def calculate_metrics(file_path: str):
    """Standalone metrics calculator for a single backtest parquet file."""
    import polars as pl
    import numpy as np

    try:
        df = pl.read_parquet(file_path)
    except Exception as e:
        print(f'Error reading file: {e}')
        return
    if 'pnl' not in df.columns:
        print("No 'pnl' column found. Ensure backtest_results.parquet contains execution output.")
        return
    metrics = compute_backtest_metrics(df)
    pnl = df['pnl'].to_numpy()
    total_pnl = pnl.sum()
    avg_pnl = pnl.mean()
    std_pnl = pnl.std()
    if std_pnl > 0:
        sharpe = metrics['sharpe_annualized']
    else:
        sharpe = 0.0
    cum_pnl = np.cumsum(pnl)
    running_max = np.maximum.accumulate(cum_pnl)
    drawdown = cum_pnl - running_max
    max_drawdown = drawdown.min()
    starting_equity = float(getattr(config, 'EQUITY', 100000.0))
    equity_curve = starting_equity + cum_pnl
    equity_running_max = np.maximum.accumulate(equity_curve)
    drawdown_pct = equity_curve / np.maximum(equity_running_max, 1e-12) - 1.0
    max_drawdown_pct = drawdown_pct.min()
    total_roe = total_pnl / starting_equity if starting_equity > 0 else 0.0
    gross_total_pnl = None
    gross_sharpe = None
    if 'gross_pnl' in df.columns:
        gross = df['gross_pnl'].to_numpy()
        gross_total_pnl = gross.sum()
        gross_std = gross.std()
        _, gross_sharpe = _sharpe_pair(gross.astype(np.float64), metrics['bars_per_year'])
    if 'position' in df.columns:
        turnover = metrics['position_turnover']
    else:
        turnover = 0.0

    bar_hit_rate_all = metrics['bar_hit_rate_all_bars']
    bar_hit_rate_active = metrics['bar_hit_rate_active_bars']

    spearman_ic = None
    if 'prediction_prob' in df.columns and 'ret_exec' in df.columns:
        from scipy.stats import spearmanr as _spearmanr
        try:
            pred = df['prediction_prob'].shift(1).to_numpy().astype(np.float64)
            targ = df['ret_exec'].to_numpy().astype(np.float64)
            mask = np.isfinite(pred) & np.isfinite(targ)
            if mask.sum() >= 3:
                ic_val, _ = _spearmanr(pred[mask], targ[mask])
                spearman_ic = round(float(ic_val), 4)
        except Exception:
            pass

    corr = 0.0
    if 'prediction' in df.columns and 'target_15m_ret' in df.columns:
        pred_old = df['prediction'].to_numpy()
        target = df['target_15m_ret'].to_numpy()
        mask = ~(np.isnan(pred_old) | np.isnan(target))
        if mask.sum() > 1:
            corr = np.corrcoef(pred_old[mask], target[mask])[0, 1]

    benchmark_sharpe = None
    benchmark_maxdd = None
    if 'benchmark_pnl' in df.columns:
        bench_pnl = df['benchmark_pnl'].to_numpy()
        bench_avg = bench_pnl.mean()
        bench_std = bench_pnl.std()
        benchmark_sharpe = (bench_avg / bench_std * np.sqrt(metrics['bars_per_year'])) if bench_std > 0 else 0.0
        bench_cum = np.cumsum(bench_pnl)
        bench_running_max = np.maximum.accumulate(bench_cum)
        benchmark_maxdd = (bench_cum - bench_running_max).min()
    print('\n' + '=' * 50)
    print('            PERFORMANCE REPORT')
    print('=' * 50)
    print(f'Total PnL:            {total_pnl:12.4f}')
    print(f'Avg PnL per bar:      {avg_pnl:12.6f}')
    print(f'Std PnL per bar:      {std_pnl:12.6f}')
    print(f'Sharpe per bar:       {metrics["sharpe_per_bar"]:12.6f}')
    print(f'Sharpe annualized:    {sharpe:12.3f}  bars_per_year={metrics["bars_per_year"]}')
    if gross_total_pnl is not None:
        print(f'Gross Total PnL:      {gross_total_pnl:12.4f}')
        print(f'Gross Sharpe (ann.):  {gross_sharpe:12.3f}')
    print(f'Starting Equity:      {starting_equity:12.2f}')
    print(f'Return on Equity:     {total_roe * 100.0:12.4f}%')
    print(f'Max Drawdown:         {max_drawdown:12.4f}')
    print(f'Max Drawdown %:       {max_drawdown_pct * 100.0:12.4f}%')
    print(f'Position Turnover:    {turnover:12.6f}')
    print(f'Prediction-Target Corr:{corr:12.4f}')
    print(f'Bar Hit Rate All:     {bar_hit_rate_all:12.4f}  denominator={metrics["bar_hit_rate_all_bars_n"]}')
    print(f'Bar Hit Rate Active:  {bar_hit_rate_active:12.4f}  denominator={metrics["bar_hit_rate_active_bars_n"]}')
    if metrics.get('trade_hit_rate') is not None:
        print(f'Trade Hit Rate:       {metrics["trade_hit_rate"]:12.4f}  denominator={metrics["trade_hit_rate_n"]}')
    if spearman_ic is not None:
        print(f'Spearman IC:           {spearman_ic:12.4f}')
    else:
        print(f'Spearman IC:           {"N/A":>12}')
    if benchmark_sharpe is not None:
        print(f'Benchmark Sharpe:     {benchmark_sharpe:12.3f}')
        print(f'Benchmark MaxDD:      {benchmark_maxdd:12.4f}')
    print('=' * 50)
