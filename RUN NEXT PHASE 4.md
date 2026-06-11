Use GPT-5.5 with extra-high reasoning.

Mode:

* Plan mode on.
* Pursue goal on.
* Include IDE/repo context.
* Inspect actual files before editing.
* Do not rely only on the pasted prompt if repo state disagrees.

Working style:

* Be adversarial about leakage, causality, schema drift, and stale docs.
* Preserve existing passing behavior.
* Do not do unrelated refactors.
* Do not build future phases.
* Do not modify generated data/reports except by running required validation commands.
* Do not commit generated artifacts.

Before editing:

1. Run `git status --short`.
2. Inspect the relevant files.
3. Produce a short implementation plan.
4. Then execute the plan.

After editing:

1. Run required validation commands.
2. Report changed files.
3. Report tests/commands run.
4. Report warnings/failures.
5. Report final `git status --short`.



Implement Phase 4 only: Baseline + L0 Regime Feature Matrix.

Repo:
C:\Users\donny\Desktop\quant_project

Create/update:

* scripts/phase4_features/**init**.py
* scripts/phase4_features/build_baseline_features.py
* tests/phase4_features/**init**.py
* tests/phase4_features/test_build_baseline_features.py

Input:

* data/labeled/{market}/{year}.parquet

Output:

* data/feature_matrices/baseline/{market}/{year}.parquet
* data/feature_matrices/baseline/feature_cols.json
* data/feature_matrices/baseline/target_cols.json
* data/feature_matrices/baseline/metadata_cols.json
* data/feature_matrices/baseline/excluded_cols.json
* reports/features_baseline/baseline_feature_manifest.json
* reports/features_baseline/baseline_feature_report.json
* reports/features_baseline/feature_registry.json
* reports/features_baseline/feature_correlation_report.csv

Do not build WFA, models, predictions, execution, metrics, gates, feature selection, expanded features, or Phase 5+.
Do not modify Phase 2/3 except for import/profile compatibility if tests are broken.
Do not commit generated data/reports.

Core rules:

* L0 1-minute OHLCV only.
* Features use completed bar t only.
* All feature columns start with feature_.
* Rolling features group by session_segment_id and never cross sessions.
* Prior-session features use only completed prior sessions.
* No target, future-path, label, audit, source, or Databento ID columns in feature_cols.
* Optional unavailable long-lookback/intermarket/prior-session features should become NaN and be reported, not fail the whole build.

Validity:

* feature_input_valid false if causal_valid false, required OHLCV missing, is_synthetic true, roll_window_flag true, boundary_session_flag true, or valid_ohlcv false.
* training_row_valid = feature_input_valid and target_valid.
* If feature_row_valid is kept, it must mean feature_input_valid only.
* Do not use target_valid, target_invalid_reason, mae/mfe, fade labels, trend labels, revert labels, or any future-path label field to decide feature availability.

Lookback / rolling validity:

* Lag and rolling features must not compute through invalid rows.
* For any feature using prior bars, if the lookback window touches feature_input_valid false rows, set that feature to NaN.
* feature_ret_1 is NaN if t-1 is synthetic, invalid OHLCV, roll_window, boundary, or outside the same session_segment_id.
* Rolling N features require the full N-bar window inside the same session_segment_id and only valid raw/input rows.
* Synthetic forward-filled prices must not create returns, volatility, range, VWAP, compression, trend, effort/result, shock, or intermarket features.

Session / opening range:

* Current-session high/low/mid/range/VWAP features use only session start through completed bar t.
* Never use final full-session high, low, close, range, or VWAP.
* Opening range features are ready only after first 30 valid session minutes complete.
* If the opening-range window contains synthetic, invalid OHLCV, roll-window, or boundary rows, set opening-range distance/breakout/open-drive features to NaN/false and report missingness.

Registries:

* feature_cols
* target_cols
* metadata_cols
* excluded_cols
* feature_families mapping feature -> family

Feature family names must include:

* baseline_ohlcv
* fade_safety_trend_danger
* breakout_rejection
* range_chop
* session_structure
* volatility_volume
* higher_timeframe_prior_session
* time_buckets
* tier1_intermarket
* effort_result
* trend_day_open_drive
* auction_acceptance
* shock_decay
* tier1_cross_market_regime

Forbidden feature columns include:

* any target_* column
* ts, market, year, session_id, session_date, session_segment_id
* raw_row_present, is_synthetic, causal_valid, valid_ohlcv, inside_session, boundary_session_flag
* feature_input_valid, training_row_valid, target_valid, target_invalid_reason
* mae_ticks_15m, mfe_ticks_15m
* fade_long_success_15m, fade_short_success_15m
* trend_danger_up_30m, trend_danger_down_30m
* revert_to_vwap_30m, revert_to_session_mid_30m
* rtype, publisher_id, instrument_id, symbol
* source_path, source_file_hash, source_row_number
* raw_schema_variant, timestamp_source, metadata_available
* roll_detection_available, roll_detection_source, roll_policy_status
* synthetic_gap_id, synthetic_gap_size_minutes, synthetic_gap_reason
* data_quality_status, data_quality_degraded, session_data_quality_degraded, trainable_data_quality

Feature families and columns:

1. Baseline OHLCV:
   feature_ret_1, feature_ret_5, feature_ret_10, feature_ret_20, feature_log_ret_1, feature_range_norm, feature_true_range, feature_ewma_vol_20, feature_volume_z_20, feature_close_position_in_range, feature_body_to_range, feature_upper_wick_ratio, feature_lower_wick_ratio, feature_minutes_since_session_open, feature_minutes_until_session_close, feature_session_progress, feature_minute_of_day_sin, feature_minute_of_day_cos, feature_day_of_week

2. Fade-safety / trend-danger:
   feature_efficiency_ratio_15, feature_efficiency_ratio_30, feature_efficiency_ratio_60, feature_directional_bar_ratio_15, feature_directional_bar_ratio_30, feature_consecutive_up_bars, feature_consecutive_down_bars, feature_trend_persistence_30, feature_signed_trend_persistence_30

3. Breakout / rejection:
   feature_prior_high_20_dist, feature_prior_low_20_dist, feature_breakout_above_20, feature_breakout_below_20, feature_failed_breakout_above_20, feature_failed_breakout_below_20, feature_close_back_inside_range_20, feature_upper_wick_rejection, feature_lower_wick_rejection

Use prior rolling ranges excluding current bar:

* prior_high_20 = high over bars t-20 to t-1.
* prior_low_20 = low over bars t-20 to t-1.
* failed_breakout_above_20 = high[t] > prior_high_20 and close[t] <= prior_high_20.
* failed_breakout_below_20 mirrors downside logic.

4. Range / chop:
   feature_realized_range_30, feature_realized_range_60, feature_range_compression_30_vs_120, feature_chop_ratio_30, feature_inside_bar_count_20, feature_overlap_ratio_20

5. Session structure:
   feature_session_open_dist, feature_session_high_dist, feature_session_low_dist, feature_session_mid_dist, feature_session_vwap_dist, feature_session_range_percentile, feature_opening_range_30_ready, feature_opening_range_30_high_dist, feature_opening_range_30_low_dist, feature_opening_range_30_breakout_up, feature_opening_range_30_breakout_down

6. Volatility / volume:
   feature_realized_vol_15, feature_realized_vol_60, feature_vol_expansion_15_vs_60, feature_large_bar_count_30, feature_shock_bar_flag, feature_bars_since_shock, feature_volume_z_60, feature_volume_surge_with_range, feature_volume_surge_without_progress, feature_range_per_volume, feature_volume_climax_flag, feature_bars_since_volume_climax

7. Higher-timeframe / prior-session:
   feature_5m_ret_3, feature_15m_ret_4, feature_60m_trend_slope, feature_daily_open_dist, feature_prior_session_high_dist, feature_prior_session_low_dist, feature_prior_session_close_dist, feature_prior_session_range_pct, feature_overnight_gap_ticks

5m/15m/60m features must use completed data only. Prefer rolling 1m-derived windows unless resampling is explicitly closed on completed bars with no future minutes.

8. Time buckets:
   feature_time_bucket_globex_open, feature_time_bucket_europe, feature_time_bucket_us_open, feature_time_bucket_midday, feature_time_bucket_power_hour, feature_first_30m_flag, feature_last_30m_flag

9. Tier-1 intermarket features, if CL/ES/ZN labeled files exist:
   feature_rel_ret_vs_ES_15, feature_rel_ret_vs_ZN_15, feature_rel_ret_vs_CL_15, feature_corr_vs_ES_60, feature_corr_vs_ZN_60, feature_corr_vs_CL_60, feature_es_zn_divergence_30, feature_cl_es_divergence_30

10. Effort vs result / absorption:
    feature_effort_result_30, feature_absorption_proxy_30, feature_exhaustion_proxy_30, feature_volume_per_tick_progress_30, feature_range_without_close_progress_30

Definitions:

* close_progress_30 = abs(close[t] - close[t-30])
* range_sum_30 = rolling sum of true_range over last 30 valid bars
* volume_sum_30 = rolling sum of volume over last 30 valid bars
* effort_result_30 = close_progress_30 / max(range_sum_30, eps)
* volume_per_tick_progress_30 = volume_sum_30 / max(close_progress_ticks_30, 1)
* range_without_close_progress_30 = range_sum_30 / max(close_progress_30, tick_size)
* absorption_proxy_30 should be high when volume/range is high but close progress is weak
* exhaustion_proxy_30 should be high when volume/range is high after directional extension but current close progress stalls

11. Trend-day / open-drive risk:
    feature_open_drive_up, feature_open_drive_down, feature_open_drive_strength_30, feature_session_one_wayness, feature_vwap_side_persistence, feature_bars_above_vwap_30, feature_bars_below_vwap_30, feature_pullback_shallowness_30

Definitions:

* open-drive features use only first 30 valid session minutes and only become populated after that window is complete
* open_drive_up/down indicate strong directional move away from session open during first 30 valid minutes
* open_drive_strength_30 = abs(close[t] - session_open) / max(session_range_so_far, tick_size), after first 30 valid minutes
* session_one_wayness = abs(close[t] - session_open) / max(session_high_so_far - session_low_so_far, tick_size)
* vwap_side_persistence = fraction of last 30 valid bars closing on the same side of session VWAP
* bars_above_vwap_30 / bars_below_vwap_30 count last 30 valid bars relative to session VWAP
* pullback_shallowness_30 measures whether pullbacks are small relative to current session directional extension

12. Auction acceptance / failed retest:
    feature_session_range_extension_up, feature_session_range_extension_down, feature_session_acceptance_above_mid, feature_session_acceptance_below_mid, feature_failed_retest_session_high, feature_failed_retest_session_low

Definitions:

* session_high_so_far/session_low_so_far use only bars through completed bar t
* range_extension_up = max(0, close[t] - prior_session_high) / tick_size when prior session exists
* range_extension_down = max(0, prior_session_low - close[t]) / tick_size when prior session exists
* session_acceptance_above_mid = fraction of last 30 valid closes above session_mid_so_far
* session_acceptance_below_mid = fraction of last 30 valid closes below session_mid_so_far
* failed_retest_session_high is true when price retests/breaks session_high_so_far reference and closes back below it without acceptance
* failed_retest_session_low mirrors downside logic
* never use final full-session high/low

13. Shock decay:
    feature_shock_direction, feature_bars_since_up_shock, feature_bars_since_down_shock, feature_post_shock_retrace_pct, feature_post_shock_continuation_pct, feature_post_shock_range_decay

Definitions:

* shock threshold must be based on past-only rolling true-range statistics
* shock_direction = sign(close[t] - close[t-1]) when shock_bar_flag true, else 0
* bars_since_up_shock and bars_since_down_shock count valid bars since last directional shock within session
* post_shock_retrace_pct measures how much of the latest shock move has retraced using only bars after the shock through t
* post_shock_continuation_pct measures extension beyond the shock extreme using only bars after the shock through t
* post_shock_range_decay compares recent true_range after shock to shock true_range using only completed bars

14. Tier-1 cross-market regime:
    feature_tier1_direction_agreement_15, feature_tier1_return_dispersion_15, feature_tier1_risk_on_score_30, feature_es_zn_risk_regime_30, feature_cl_es_macro_divergence_30

Definitions:

* Use exact timestamp joins only across CL, ES, ZN.
* No forward-fill.
* No self-vs-self features.
* Do not read other-market target_valid or target/label columns.
* For each market row, use other-market OHLCV/causal validity only.
* tier1_direction_agreement_15 = fraction of available tier-1 markets with same 15-bar return sign
* tier1_return_dispersion_15 = cross-sectional std of 15-bar returns across available tier-1 markets
* tier1_risk_on_score_30 combines ES strength, ZN weakness, and CL direction using only current/past completed bars
* es_zn_risk_regime_30 captures ES versus ZN directional relationship
* cl_es_macro_divergence_30 captures CL versus ES directional divergence

Intermarket guard:

* Join other markets by exact ts only.
* No forward-fill.
* Do not create self-vs-self features.
* For ES rows, skip/NaN ES-vs-ES features; same for CL and ZN.
* Other-market columns allowed: ts, open, high, low, close, volume, causal_valid, valid_ohlcv, is_synthetic, roll_window_flag, boundary_session_flag, session_segment_id.
* Do not read other-market target_valid or target/label columns.
* If other market row is missing or invalid, set intermarket features NaN and report missing rate.

Reports:

* row counts by file
* feature_input_valid and training_row_valid counts
* target_valid counts
* NaN count/pct by feature
* feature family counts
* forbidden-column leakage check
* intermarket missing rates if attempted
* correlation pairs with abs(corr) >= 0.98, computed on training_row_valid rows only
* warning_count, failure_count, failures

Tests:

* profile aliases resolve
* ret_1 uses close[t] / close[t-1] only
* rolling features do not cross session_segment_id or invalid lookback rows
* breakout uses prior range excluding current bar
* session VWAP uses only bars up to t
* session high/low/mid/range use session-so-far only
* opening range unavailable before first 30 valid minutes
* effort/result features become NaN if any lookback row is invalid/synthetic
* open-drive features unavailable before first 30 valid session minutes
* open-drive window invalid if it contains synthetic/invalid rows
* session_one_wayness uses session-so-far only
* acceptance features use session midpoint/VWAP through t only
* failed retest does not use final session high/low
* shock decay uses only bars after latest shock and before/equal t
* tier1 cross-market features use exact timestamp joins only
* tier1 cross-market features do not use self-vs-self or target columns
* feature_input_valid does not depend on target_valid
* row can have feature_input_valid=true and target_valid=false
* training_row_valid = feature_input_valid and target_valid
* invalid causal/synthetic/roll/boundary/invalid-OHLCV rows invalidate feature_input_valid
* target/label/audit/source columns excluded from feature_cols
* 5m/15m/60m features use completed data only
* registry JSONs exist in data/feature_matrices/baseline/
* reports are written

Validation:
python -m scripts.phase4_features.build_baseline_features --profile tier_1_core
python -m pytest -q
git status --short

Stop after reporting changed files, validation results, output row counts, feature count, warnings/failures, and git status.
