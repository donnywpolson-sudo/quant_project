from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import polars as pl
from pipeline.common.config import config
from pipeline.common.io.atomic import atomic_write_canonical_parquet
import logging
logger = logging.getLogger(__name__)


DATA_ROOT = Path("data")
RAW_DATA_ROOT = DATA_ROOT / "raw"
VALIDATED_DATA_ROOT = DATA_ROOT / "validated"
SESSION_NORMALIZED_DATA_ROOT = DATA_ROOT / "session_normalized"
CAUSALLY_GATED_NORMALIZED_DATA_ROOT = DATA_ROOT / "causally_gated_normalized"
LABELED_DATA_ROOT = DATA_ROOT / "labeled"
FEATURES_BASELINE_DATA_ROOT = DATA_ROOT / "features_baseline"
FEATURE_MATRICES_BASELINE_ROOT = DATA_ROOT / "feature_matrices" / "baseline"
FEATURE_MATRICES_EXPANDED_ROOT = DATA_ROOT / "feature_matrices" / "expanded"
FROZEN_FEATURES_ROOT = DATA_ROOT / "frozen_features"

REPORTS_ROOT = Path("reports")
VALIDATION_REPORTS_ROOT = REPORTS_ROOT / "validation"
SESSION_NORMALIZATION_REPORTS_ROOT = REPORTS_ROOT / "session_normalization"
CAUSAL_GATING_REPORTS_ROOT = REPORTS_ROOT / "causal_gating"
WFA_REPORTS_ROOT = REPORTS_ROOT / "wfa"
METRICS_REPORTS_ROOT = REPORTS_ROOT / "metrics"

ARTIFACTS_ROOT = Path("artifacts")
MODELS_ARTIFACTS_ROOT = ARTIFACTS_ROOT / "models"
SCALERS_ARTIFACTS_ROOT = ARTIFACTS_ROOT / "scalers"
SELECTORS_ARTIFACTS_ROOT = ARTIFACTS_ROOT / "selectors"
RUN_MANIFESTS_ARTIFACTS_ROOT = ARTIFACTS_ROOT / "run_manifests"
BACKTESTS_ARTIFACTS_ROOT = ARTIFACTS_ROOT / "backtests"


def write_canonical_parquet(data: pl.DataFrame | pa.Table, path: str):
    if isinstance(data, pl.DataFrame):
        df = data
    else:
        df = pl.from_arrow(data)
    row_group_size = getattr(config, 'ROW_GROUP_SIZE', 65536)
    atomic_write_canonical_parquet(df, path, row_group_size=row_group_size)
    logger.info('Successfully wrote canonical parquet to %s with %d columns.',
                path, len(df.columns))
