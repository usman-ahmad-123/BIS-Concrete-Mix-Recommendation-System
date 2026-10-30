
"""BIS-oriented concrete mix candidate recommendation prototype.

This application:
- cleans the standard concrete-strength dataset without imputing invented mixes;
- evaluates CatBoost with an 80:20 formulation-group-held-out split;
- generates candidate recommendations only from the held-out formulations;
- saves validation and full-data models separately;
- ranks candidates by smallest non-negative strength surplus, then binder content;
- presents engineering observations rather than unsupported defect probabilities.

It is NOT a certified IS 10262 mix-design calculator or substitute for trial batches.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    shap = None
    SHAP_AVAILABLE = False


# ============================================================
# 1. PAGE AND PROJECT CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="BIS Concrete Mix Recommendation System",
    page_icon="🏗️",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models" / "tuned"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

VALIDATION_MODEL_PATH = MODEL_DIR / "CatBoost_BIS_validation_group_holdout.pkl"
FULL_MODEL_PATH = MODEL_DIR / "CatBoost_BIS_full_data.pkl"
MODEL_METADATA_PATH = MODEL_DIR / "CatBoost_BIS_model_metadata.json"

FEATURE_NAMES = [
    "Cement (component 1)(kg in a m^3 mixture)",
    "Blast Furnace Slag (component 2)(kg in a m^3 mixture)",
    "Fly Ash (component 3)(kg in a m^3 mixture)",
    "Water  (component 4)(kg in a m^3 mixture)",
    "Superplasticizer (component 5)(kg in a m^3 mixture)",
    "Coarse Aggregate  (component 6)(kg in a m^3 mixture)",
    "Fine Aggregate (component 7)(kg in a m^3 mixture)",
    "Age (day)",
]

INGREDIENT_NAMES = FEATURE_NAMES[:-1]
AGE_COLUMN = FEATURE_NAMES[-1]

DISPLAY_NAMES = {
    FEATURE_NAMES[0]: "Cement",
    FEATURE_NAMES[1]: "Blast Furnace Slag",
    FEATURE_NAMES[2]: "Fly Ash",
    FEATURE_NAMES[3]: "Water",
    FEATURE_NAMES[4]: "Superplasticizer",
    FEATURE_NAMES[5]: "Coarse Aggregate",
    FEATURE_NAMES[6]: "Fine Aggregate",
    FEATURE_NAMES[7]: "Curing Age",
}

DISPLAY_UNITS = {
    FEATURE_NAMES[0]: "kg/m³",
    FEATURE_NAMES[1]: "kg/m³",
    FEATURE_NAMES[2]: "kg/m³",
    FEATURE_NAMES[3]: "kg/m³",
    FEATURE_NAMES[4]: "kg/m³",
    FEATURE_NAMES[5]: "kg/m³",
    FEATURE_NAMES[6]: "kg/m³",
    FEATURE_NAMES[7]: "days",
}


# ============================================================
# 2. DATASET DISCOVERY AND CLEANING
# ============================================================

def normalize_name(value: Any) -> str:
    """Normalize headers to tolerate extra spaces and case changes."""
    return " ".join(str(value).strip().lower().split())


def find_dataset() -> Path:
    """Find the dataset in common project locations."""
    candidates = [
        BASE_DIR / "Raw_data" / "Concrete_Data.csv",
        BASE_DIR / "Raw_data" / "concrete_data.csv",
        BASE_DIR / "Datasets" / "Raw_data" / "Concrete_Data.csv",
        BASE_DIR / "Datasets" / "Concrete_Data.csv",
        BASE_DIR / "Concrete_Data.csv",
        BASE_DIR / "Raw_data" / "Concrete_Data.xlsx",
    ]
    for path in candidates:
        if path.is_file():
            return path
    checked = "\n".join(str(p) for p in candidates)
    raise FileNotFoundError(
        "Concrete dataset not found. Place Concrete_Data.csv in Raw_data/. "
        f"Locations checked:\n{checked}"
    )


@st.cache_data(show_spinner=False)
def load_dataset(
    dataset_path: str,
    dataset_mtime: float
) -> tuple[pd.DataFrame, pd.Series, dict[str, Any]]:
    """Load, validate and clean data. Invalid rows are dropped, not imputed.

    The path and modification time are cache keys so replacing the CSV
    causes Streamlit to refresh the cleaned dataset and both models.
    """
    path = Path(dataset_path)
    _ = dataset_mtime

    if path.suffix.lower() in {".xlsx", ".xls"}:
        raw = pd.read_excel(path)
    else:
        raw = pd.read_csv(path)

    if raw.empty:
        raise ValueError("The dataset is empty.")

    normalized_to_original: dict[str, str] = {}

    for col in raw.columns:
        key = normalize_name(col)

        if key in normalized_to_original:
            raise ValueError(
                f"Duplicate/ambiguous column headers after whitespace normalization: {col}"
            )

        normalized_to_original[key] = col

    source_features: list[str] = []

    for feature in FEATURE_NAMES:
        key = normalize_name(feature)

        if key not in normalized_to_original:
            raise ValueError(
                f"Required predictor column is missing: {feature}. "
                "Check that this is the standard Concrete Compressive Strength dataset."
            )

        source_features.append(normalized_to_original[key])

    target_candidates = [
        col for col in raw.columns
        if "compressive strength" in normalize_name(col)
        and col not in source_features
    ]

    if len(target_candidates) != 1:
        raise ValueError(
            "Could not identify exactly one target column containing 'compressive strength'. "
            f"Candidates found: {target_candidates}"
        )

    target_source = target_candidates[0]

    X_raw = raw[source_features].copy()
    X_raw.columns = FEATURE_NAMES
    y_raw = raw[target_source].copy()

    # Coerce invalid text to NaN so it is explicitly counted and removed.
    X = X_raw.apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(y_raw, errors="coerce")

    missing_mask = X.isna().any(axis=1) | y.isna()

    finite_x = np.isfinite(X.to_numpy(dtype=float)).all(axis=1)
    finite_y = np.isfinite(y.to_numpy(dtype=float))

    nonfinite_mask = ~(finite_x & finite_y)

    # Broad physical-domain checks only; no arbitrary ingredient upper limits.
    nonnegative_mask = (X[INGREDIENT_NAMES] >= 0).all(axis=1)
    positive_cement_mask = X[FEATURE_NAMES[0]] > 0
    positive_age_mask = X[AGE_COLUMN] > 0

    positive_binder_mask = (
        X[FEATURE_NAMES[0]]
        + X[FEATURE_NAMES[1]]
        + X[FEATURE_NAMES[2]]
    ) > 0

    invalid_domain_mask = ~(
        nonnegative_mask
        & positive_cement_mask
        & positive_age_mask
        & positive_binder_mask
        & (y > 0)
    )

    valid_mask = ~(
        missing_mask
        | nonfinite_mask
        | invalid_domain_mask
    )

    X_clean = X.loc[valid_mask].reset_index(drop=True)
    y_clean = y.loc[valid_mask].reset_index(drop=True).astype(float)

    if len(X_clean) < 30:
        raise ValueError(
            f"Only {len(X_clean)} valid rows remain. "
            "At least 30 are required for this prototype."
        )

    if X_clean[FEATURE_NAMES].isna().any().any():
        raise ValueError(
            "Unexpected missing values remain after cleaning."
        )

    quality = {
        "dataset_path": str(path),
        "target_column": str(target_source),
        "raw_rows": int(len(raw)),
        "valid_rows": int(len(X_clean)),
        "dropped_rows": int((~valid_mask).sum()),
        "missing_or_non_numeric_rows": int(missing_mask.sum()),
        "non_finite_rows": int(nonfinite_mask.sum()),
        "invalid_domain_rows": int(invalid_domain_mask.sum()),
        "unique_formulations": int(
            pd.MultiIndex.from_frame(
                X_clean[INGREDIENT_NAMES].round(8)
            ).nunique()
        ),
    }

    return X_clean, y_clean, quality


# ============================================================
# 3. GROUP-HELD-OUT VALIDATION AND MODEL STORAGE
# ============================================================

@st.cache_resource(show_spinner="Training group-held-out CatBoost models...")
def train_and_validate(
    X: pd.DataFrame,
    y: pd.Series,
) -> tuple[
    CatBoostRegressor,
    CatBoostRegressor,
    pd.DataFrame,
    pd.Series,
    dict[str, Any]
]:
    """Split by recipe formulation so every age of a recipe stays in one split."""

    if len(X) != len(y):
        raise ValueError("Feature and target row counts differ.")

    # Age is intentionally excluded:
    # all ages of the same recipe stay together.
    formulation_index = pd.MultiIndex.from_frame(
        X[INGREDIENT_NAMES].round(8)
    )

    groups, unique_groups = pd.factorize(
        formulation_index,
        sort=True
    )

    if len(unique_groups) < 5:
        raise ValueError(
            "Too few unique ingredient formulations for group-held-out validation."
        )

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=42,
    )

    train_idx, test_idx = next(
        splitter.split(X, y, groups=groups)
    )

    X_train = X.iloc[train_idx].reset_index(drop=True)
    y_train = y.iloc[train_idx].reset_index(drop=True)

    X_test = X.iloc[test_idx].reset_index(drop=True)
    y_test = y.iloc[test_idx].reset_index(drop=True)

    train_group_count = int(
        np.unique(groups[train_idx]).size
    )

    test_group_count = int(
        np.unique(groups[test_idx]).size
    )

    # This model is used for held-out evaluation AND candidate scoring.
    # Candidate formulations are drawn only from X_test, so their recipes
    # were not present in this model's training groups.
    validation_model = CatBoostRegressor(
        iterations=700,
        depth=6,
        learning_rate=0.05,
        loss_function="RMSE",
        random_seed=42,
        verbose=False,
        allow_writing_files=False,
        thread_count=-1,
    )

    validation_model.fit(
        X_train,
        y_train
    )

    test_predictions = np.asarray(
        validation_model.predict(X_test),
        dtype=float
    )

    if not np.isfinite(test_predictions).all():
        raise ValueError(
            "Validation model returned non-finite test predictions."
        )

    metrics = {
        "r2": float(
            r2_score(y_test, test_predictions)
        ),
        "rmse": float(
            np.sqrt(
                mean_squared_error(y_test, test_predictions)
            )
        ),
        "mae": float(
            mean_absolute_error(y_test, test_predictions)
        ),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "train_formulations": train_group_count,
        "test_formulations": test_group_count,
        "total_formulations": int(len(unique_groups)),
        "split": "GroupShuffleSplit 80:20 by seven-ingredient formulation",
        "random_state": 42,
    }

    # A second, separate model is fitted on all valid data for deployment.
    # It is NOT used to score the held-out candidate recommendations below.
    full_model = CatBoostRegressor(
        iterations=700,
        depth=6,
        learning_rate=0.05,
        loss_function="RMSE",
        random_seed=42,
        verbose=False,
        allow_writing_files=False,
        thread_count=-1,
    )

    full_model.fit(
        X,
        y
    )

    for name, fitted_model in [
        ("validation model", validation_model),
        ("full-data model", full_model),
    ]:

        fitted_names = list(
            getattr(fitted_model, "feature_names_", []) or []
        )

        if len(fitted_names) != len(FEATURE_NAMES):
            raise ValueError(
                f"{name} has {len(fitted_names)} features; expected exactly 8."
            )

        if fitted_names != FEATURE_NAMES:
            raise ValueError(
                f"{name} feature names/order do not match the app schema."
            )

    # Persist both models independently.
    # Never overwrite the user's old model file.
    joblib.dump(
        validation_model,
        VALIDATION_MODEL_PATH
    )

    joblib.dump(
        full_model,
        FULL_MODEL_PATH
    )

    MODEL_METADATA_PATH.write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    return (
        validation_model,
        full_model,
        X_test,
        y_test,
        metrics
    )


# ============================================================
# 4. INITIALIZE DATA AND MODELS
# ============================================================

st.title(
    "🏗️ BIS-Oriented Concrete Mix Recommendation System"
)

st.markdown(
    """
    Search historically tested concrete formulations for candidate mixes
    that meet a user-entered **target mean compressive strength**.

    The model predicts strength; engineering checks and laboratory trials
    remain mandatory.
    """
)

st.warning(
    "Research and decision-support prototype only. This app does not produce "
    "a certified BIS mix design, establish code compliance, or guarantee defect-free concrete."
)

try:

    dataset_file = find_dataset()

    X, y, data_quality = load_dataset(
        str(dataset_file),
        dataset_file.stat().st_mtime,
    )

    (
        validation_model,
        full_model,
        X_test,
        y_test,
        metrics
    ) = train_and_validate(X, y)

except Exception as exc:

    st.error(
        f"Application initialization failed: {exc}"
    )

    st.stop()


# ============================================================
# 5. DATA QUALITY SUMMARY
# ============================================================

with st.expander(
    "Dataset processing and data-quality report"
):

    q1, q2, q3 = st.columns(3)

    q1.metric(
        "Raw rows",
        f"{data_quality['raw_rows']:,}"
    )

    q2.metric(
        "Valid rows",
        f"{data_quality['valid_rows']:,}"
    )

    q3.metric(
        "Dropped rows",
        f"{data_quality['dropped_rows']:,}"
    )

    st.write(
        f"**Dataset:** `{data_quality['dataset_path']}`"
    )

    st.write(
        f"**Target column:** `{data_quality['target_column']}`"
    )

    st.write(
        f"**Unique ingredient formulations:** "
        f"{data_quality['unique_formulations']:,}"
    )

    st.write(
        "Rows with missing/non-numeric values: "
        f"{data_quality['missing_or_non_numeric_rows']}"
    )

    st.write(
        f"Rows with non-finite values: {data_quality['non_finite_rows']}"
    )

    st.write(
        "Rows failing basic physical-domain checks: "
        f"{data_quality['invalid_domain_rows']}"
    )

    st.caption(
        "Invalid records are removed rather than imputed: fabricated ingredient "
        "or strength values would create synthetic mixes and could distort the "
        "recommendation search. Overlapping rejection categories are not additive."
    )


# ============================================================
# 6. GROUP-HELD-OUT MODEL PERFORMANCE
# ============================================================

st.markdown("---")

st.header(
    "📈 Model Performance — Group-Held-Out Test Set"
)

st.caption(
    "Metrics are calculated on unseen formulation groups, not a random row split. "
    "All records belonging to the same seven-ingredient recipe (including its "
    "different curing ages) are kept entirely in either training or testing."
)

m1, m2, m3 = st.columns(3)

m1.metric(
    "Held-out R²",
    f"{metrics['r2']:.4f}"
)

m2.metric(
    "Held-out RMSE",
    f"{metrics['rmse']:.2f} MPa"
)

m3.metric(
    "Held-out MAE",
    f"{metrics['mae']:.2f} MPa"
)

st.caption(
    f"Training: {metrics['train_rows']} rows / "
    f"{metrics['train_formulations']} formulations  |  "
    f"Test: {metrics['test_rows']} rows / "
    f"{metrics['test_formulations']} formulations  |  "
    f"Split: 80:20, random_state={metrics['random_state']}"
)

with st.expander(
    "Model configuration and saved artifacts"
):

    st.write(
        "Algorithm: CatBoostRegressor"
    )

    st.write(
        "Features: 8 (seven ingredient quantities + age)"
    )

    st.write(
        "Validation model — trained on training groups only:"
    )

    st.code(
        str(VALIDATION_MODEL_PATH)
    )

    st.write(
        "Full-data model — trained on all valid dataset rows:"
    )

    st.code(
        str(FULL_MODEL_PATH)
    )

    st.write(
        "Validation metadata:"
    )

    st.code(
        str(MODEL_METADATA_PATH)
    )

    st.info(
        "Candidate predictions below use the validation model, not the full-data model. "
        "The full-data model is saved separately for later deployment after the validation "
        "strategy and external performance have been reviewed."
    )


# ============================================================
# 7. PROJECT INPUTS
# ============================================================

st.markdown("---")

st.header(
    "🎯 Project Requirements"
)

st.write(
    "Enter the target mean strength independently. It is not automatically inferred "
    "from the selected grade; determine it through the approved mix-design procedure."
)

left, right = st.columns(2)

with left:

    reference_grade = st.selectbox(
        "Characteristic grade reference (optional)",
        [
            "Not specified",
            "M10",
            "M15",
            "M20",
            "M25",
            "M30",
            "M35",
            "M40",
            "M45",
            "M50"
        ],
        index=0,
        help=(
            "Reference label only. It does not set or calculate "
            "the target mean strength."
        ),
    )

    target_mean_strength = st.number_input(
        "Target mean compressive strength (MPa)",
        min_value=1.0,
        max_value=150.0,
        value=30.0,
        step=0.5,
        format="%.1f",
        help=(
            "Enter the target mean strength determined for the project. "
            "This value is screened directly."
        ),
    )

    # Only ages actually present in the held-out formulation groups are offered.
    available_ages = sorted(
        pd.to_numeric(
            X_test[AGE_COLUMN],
            errors="coerce"
        ).dropna().unique().tolist()
    )

    if not available_ages:
        st.error(
            "No valid historically tested curing ages are available in the held-out groups."
        )
        st.stop()

    default_age = (
        28 if 28 in available_ages
        else available_ages[0]
    )

    selected_age = st.selectbox(
        "Historically tested curing age (days)",
        available_ages,
        index=available_ages.index(default_age),
        format_func=lambda value: f"{value:g} days",
        help=(
            "Choices are actual ages observed among held-out formulations in this dataset."
        ),
    )


with right:

    construction_purpose = st.selectbox(
        "Purpose of construction",
        [
            "Residential building",
            "Commercial building",
            "High-rise building",
            "Bridge construction",
            "Road / pavement",
            "Foundation",
            "Precast concrete",
            "Mass concrete",
            "Water-retaining structure",
            "Other structural application",
        ],
    )

    exposure_condition = st.selectbox(
        "Exposure condition",
        [
            "Not specified",
            "Mild",
            "Moderate",
            "Severe",
            "Very severe",
            "Extreme"
        ],
    )

    placement_method = st.selectbox(
        "Placement method",
        [
            "Conventional placement",
            "Pumped concrete",
            "Congested reinforcement",
            "Precast production",
            "Mass concrete placement",
        ],
    )


st.subheader(
    "Fresh-concrete and site requirements"
)

s1, s2, s3 = st.columns(3)

with s1:

    target_slump = st.number_input(
        "Required slump (mm)",
        min_value=0,
        max_value=250,
        value=100,
        step=5
    )

with s2:

    nominal_aggregate_size = st.selectbox(
        "Nominal maximum aggregate size (mm)",
        [10.0, 12.5, 20.0, 40.0],
        index=2,
        format_func=lambda value: f"{value:g} mm",
    )

with s3:

    transport_minutes = st.number_input(
        "Transport + placement duration (min)",
        min_value=0,
        max_value=360,
        value=60,
        step=5
    )


st.caption(
    "Purpose, exposure, placement method, slump and aggregate size are captured "
    "for engineering review. The source dataset has no corresponding labels/features, "
    "so these inputs do not alter CatBoost predictions."
)


with st.expander(
    "Site weather observations (not model predictors)"
):

    w1, w2, w3 = st.columns(3)

    with w1:

        ambient_temperature = st.number_input(
            "Ambient temperature (°C)",
            min_value=-10.0,
            max_value=60.0,
            value=30.0,
            step=0.5,
            format="%.1f"
        )

    with w2:

        relative_humidity = st.number_input(
            "Relative humidity (%)",
            min_value=0.0,
            max_value=100.0,
            value=50.0,
            step=1.0,
            format="%.1f"
        )

    with w3:

        wind_speed = st.number_input(
            "Wind speed (km/h)",
            min_value=0.0,
            max_value=150.0,
            value=10.0,
            step=0.5,
            format="%.1f"
        )

    st.caption(
        "Weather values are recorded for the engineering observation module only. "
        "They are not inputs to CatBoost and are not converted into defect probabilities."
    )


# ============================================================
# 8. DOMAIN CHECKS AND CANDIDATE GENERATION
# ============================================================

st.markdown("---")

st.header(
    "🧪 Candidate Mix Search"
)

observed_strength_min = float(y.min())
observed_strength_max = float(y.max())

if not (
    observed_strength_min
    <= target_mean_strength
    <= observed_strength_max
):

    st.error(
        f"Target mean strength {target_mean_strength:.1f} MPa is outside the "
        f"observed dataset target range "
        f"({observed_strength_min:.1f}–{observed_strength_max:.1f} MPa). "
        "Do not extrapolate this prototype beyond its training data."
    )

    st.stop()


# Candidates are actual rows from the held-out formulations,
# at an age actually tested.

heldout_candidates = X_test.loc[
    np.isclose(
        X_test[AGE_COLUMN].astype(float),
        float(selected_age)
    )
].copy()

if heldout_candidates.empty:

    st.warning(
        "No held-out formulation has a test record at the selected age. "
        "Choose another age from the offered historical age list."
    )

    st.stop()


# One unique recipe per tested age.
# Retain only seven ingredients plus age.

heldout_candidates = (
    heldout_candidates[FEATURE_NAMES]
    .drop_duplicates(subset=FEATURE_NAMES)
    .reset_index(drop=True)
)


# Score using the validation model, which has never seen
# these ingredient formulations during training.

try:

    heldout_predictions = np.asarray(
        validation_model.predict(
            heldout_candidates[FEATURE_NAMES]
        ),
        dtype=float,
    )

except Exception as exc:

    st.error(
        f"Held-out candidate prediction failed: {exc}"
    )

    st.stop()


if not np.isfinite(heldout_predictions).all():

    st.error(
        "The validation model returned invalid candidate predictions."
    )

    st.stop()


candidate_df = heldout_candidates.copy()

candidate_df["Predicted Strength (MPa)"] = (
    heldout_predictions
)

candidate_df["Strength Surplus (MPa)"] = (
    candidate_df["Predicted Strength (MPa)"]
    - target_mean_strength
)

candidate_df["Total Binder (kg/m³)"] = (
    candidate_df[FEATURE_NAMES[0]]
    + candidate_df[FEATURE_NAMES[1]]
    + candidate_df[FEATURE_NAMES[2]]
)

candidate_df["Water/Binder Ratio"] = (
    candidate_df[FEATURE_NAMES[3]]
    / candidate_df["Total Binder (kg/m³)"].replace(
        0,
        np.nan
    )
)

candidate_df["Fine Aggregate Fraction"] = (
    candidate_df[FEATURE_NAMES[6]]
    / (
        candidate_df[FEATURE_NAMES[5]]
        + candidate_df[FEATURE_NAMES[6]]
    ).replace(
        0,
        np.nan
    )
)

candidate_df = (
    candidate_df
    .replace([np.inf, -np.inf], np.nan)
    .dropna()
    .copy()
)


# ============================================================
# 9. STRENGTH FILTER AND RANKING
# ============================================================

# A feasible candidate must meet the user-entered target mean strength.

feasible = candidate_df.loc[
    candidate_df["Predicted Strength (MPa)"]
    >= target_mean_strength
].copy()

has_feasible_candidates = not feasible.empty

if not has_feasible_candidates:

    st.warning(
        "No unseen historical formulation at this tested age meets the entered target mean strength. "
        "The app will not fabricate new ingredient combinations or silently lower the target."
    )

    st.write(
        "The closest held-out formulations are shown for inspection only. "
        "They are below the target and are NOT recommendations."
    )

    feasible = candidate_df.sort_values(
        "Predicted Strength (MPa)",
        ascending=False
    ).head(5).copy()

else:

    # Required ranking:
    # smallest non-negative strength surplus first;
    # total binder content is the tie-breaker.

    feasible = feasible.sort_values(
        by=[
            "Strength Surplus (MPa)",
            "Total Binder (kg/m³)"
        ],
        ascending=[True, True],
        kind="mergesort",
    ).reset_index(drop=True)


if has_feasible_candidates:

    st.success(
        f"Found {len(feasible)} unseen historical formulation(s) meeting "
        f"the target mean strength of {target_mean_strength:.1f} MPa "
        f"at {selected_age:g} days."
    )

else:

    st.error(
        "No mix is recommended for this target and age. "
        "Review the project requirement or obtain additional validated data."
    )


st.caption(
    "The displayed formulations come from held-out recipe groups. They were not used to train "
    "the validation model that predicts their strength. Passing the model threshold does not "
    "make a candidate an approved mix proportion."
)


# ============================================================
# 10. COMPACT RECOMMENDATION TABLE
# ============================================================

if has_feasible_candidates:

    st.subheader(
        "Ranked candidate shortlist"
    )

else:

    st.subheader(
        "Closest candidate shortlist (below target)"
    )


feasible.insert(
    0,
    "Candidate",
    [
        f"Mix {i + 1}"
        for i in range(len(feasible))
    ]
)

summary_table = feasible[
    [
        "Candidate",
        "Predicted Strength (MPa)",
        "Strength Surplus (MPa)",
        "Total Binder (kg/m³)",
        "Water/Binder Ratio",
    ]
].copy()

summary_table = summary_table.rename(
    columns={
        "Predicted Strength (MPa)": "Pred. MPa",
        "Strength Surplus (MPa)": "Surplus MPa",
        "Total Binder (kg/m³)": "Binder kg/m³",
        "Water/Binder Ratio": "W/B",
    }
)

st.dataframe(
    summary_table.round(3),
    use_container_width=True,
    hide_index=True,
    column_config={
        "Candidate": st.column_config.TextColumn(
            "Mix",
            width="small"
        ),
        "Pred. MPa": st.column_config.NumberColumn(
            "Pred. MPa",
            format="%.2f",
            width="small"
        ),
        "Surplus MPa": st.column_config.NumberColumn(
            "Surplus",
            format="%.2f",
            width="small"
        ),
        "Binder kg/m³": st.column_config.NumberColumn(
            "Binder",
            format="%.1f",
            width="small"
        ),
        "W/B": st.column_config.NumberColumn(
            "W/B",
            format="%.3f",
            width="small"
        ),
    },
)


selected_candidate_number = st.selectbox(
    "View ingredient proportions for",
    options=list(range(len(feasible))),
    format_func=lambda index: (
        f"Mix {index + 1} — "
        f"{feasible.iloc[index]['Predicted Strength (MPa)']:.2f} MPa"
    ),
)

selected_mix = feasible.iloc[selected_candidate_number]


if has_feasible_candidates:

    st.subheader(
        f"Mix {selected_candidate_number + 1}: ingredient recommendation"
    )

else:

    st.subheader(
        f"Mix {selected_candidate_number + 1}: ingredient quantities for inspection only"
    )


ingredient_output = pd.DataFrame({
    "Ingredient": [
        DISPLAY_NAMES[col]
        for col in INGREDIENT_NAMES
    ],
    "Recommended quantity (kg/m³)": [
        float(selected_mix[col])
        for col in INGREDIENT_NAMES
    ],
})


mix_left, mix_right = st.columns([1.2, 1])

with mix_left:

    st.dataframe(
        ingredient_output.round(2),
        use_container_width=True,
        hide_index=True,
        height=285,
    )

with mix_right:

    st.metric(
        "Predicted strength",
        f"{selected_mix['Predicted Strength (MPa)']:.2f} MPa"
    )

    st.metric(
        "Target mean strength",
        f"{target_mean_strength:.2f} MPa"
    )

    st.metric(
        "Strength surplus",
        f"{selected_mix['Strength Surplus (MPa)']:.2f} MPa"
    )

    st.metric(
        "Water/binder ratio",
        f"{selected_mix['Water/Binder Ratio']:.3f}"
    )


# Export full candidate schedule, including all ingredient values.

export_columns = [
    "Candidate",
    *INGREDIENT_NAMES,
    "Predicted Strength (MPa)",
    "Strength Surplus (MPa)",
    "Total Binder (kg/m³)",
    "Water/Binder Ratio",
    "Fine Aggregate Fraction",
]

export_df = feasible[export_columns].copy()

export_df = export_df.rename(
    columns={
        DISPLAY_NAMES.get(c, c): c
        for c in export_df.columns
    }
)

export_df.columns = [
    "Candidate" if col == "Candidate"
    else f"{DISPLAY_NAMES[col]} ({DISPLAY_UNITS[col]})"
    if col in DISPLAY_NAMES
    else col
    for col in export_df.columns
]


st.download_button(
    (
        "⬇️ Download ranked candidate mixes (CSV)"
        if has_feasible_candidates
        else "⬇️ Download closest held-out mixes (CSV)"
    ),
    data=export_df.to_csv(index=False).encode("utf-8"),
    file_name="BIS_group_heldout_mix_candidates.csv",
    mime="text/csv",
)


# ============================================================
# 11. ENGINEERING OBSERVATIONS — NO FABRICATED PROBABILITIES
# ============================================================

st.markdown("---")

st.header(
    "🧰 Fresh-concrete defect prevention observations"
)

st.warning(
    "The dataset and CatBoost model do not contain measured bleeding, segregation, "
    "plastic-shrinkage or slump-retention outcomes. The items below are engineering "
    "observations and trial-batch checks—not ML classifications, pass/fail certificates "
    "or defect probabilities."
)


st.subheader(
    "Recorded site conditions"
)

weather_col1, weather_col2, weather_col3 = st.columns(3)

weather_col1.metric(
    "Temperature",
    f"{ambient_temperature:.1f} °C"
)

weather_col2.metric(
    "Relative humidity",
    f"{relative_humidity:.1f}%"
)

weather_col3.metric(
    "Wind speed",
    f"{wind_speed:.1f} km/h"
)


st.info(
    "Use the recorded temperature, humidity and wind conditions in a suitable site-specific "
    "evaporation assessment immediately before placing concrete. This application does not "
    "calculate evaporation rate or plastic-shrinkage probability."
)


observations = pd.DataFrame([
    {
        "Issue": "Bleeding",
        "Review / field check": (
            "Measure bleeding on trial concrete; examine water content, "
            "paste volume, fines, grading and admixture compatibility."
        ),
        "Evidence required": (
            "Fresh-concrete bleeding test and trial-batch observations."
        ),
    },
    {
        "Issue": "Segregation",
        "Review / field check": (
            "Check aggregate grading, cohesion, consistency, discharge, "
            "vibration and pumping behaviour."
        ),
        "Evidence required": (
            "Visual stability assessment and placement/pumping trial where relevant."
        ),
    },
    {
        "Issue": "Plastic shrinkage",
        "Review / field check": (
            "Assess evaporation conditions; plan wind/sun protection, "
            "finishing sequence and prompt curing."
        ),
        "Evidence required": (
            "Site weather/evaporation assessment and curing/finishing plan."
        ),
    },
    {
        "Issue": "Workability loss",
        "Review / field check": (
            "Test slump/consistency at discharge and after the expected "
            "transport/placement duration."
        ),
        "Evidence required": (
            "Slump-retention and cement-admixture compatibility trials."
        ),
    },
    {
        "Issue": "Durability",
        "Review / field check": (
            "Verify exposure-specific limits, cover, binder type, "
            "water/binder ratio and curing."
        ),
        "Evidence required": (
            "Material certificates, design checks and required durability tests."
        ),
    },
])

st.dataframe(
    observations,
    use_container_width=True,
    hide_index=True
)

st.caption(
    f"Context supplied: {construction_purpose}; "
    f"exposure={exposure_condition}; "
    f"placement={placement_method}; "
    f"slump target={target_slump} mm; "
    f"nominal aggregate size={nominal_aggregate_size:g} mm; "
    f"transport/placement duration={transport_minutes} min. "
    "These values are not trained defect-model inputs."
)


# ============================================================
# 12. SHAP EXPLANATION FOR SELECTED HELD-OUT CANDIDATE
# ============================================================

st.markdown("---")

st.header(
    "🧠 SHAP explanation — selected candidate strength"
)

st.caption(
    "SHAP explains the validation model's predicted compressive strength for this selected "
    "held-out candidate. It does not explain or predict fresh-concrete defects."
)


if SHAP_AVAILABLE:

    try:

        selected_model_input = pd.DataFrame(
            [
                [
                    float(selected_mix[col])
                    for col in FEATURE_NAMES
                ]
            ],
            columns=FEATURE_NAMES,
        )

        shap_explainer = shap.TreeExplainer(
            validation_model
        )

        shap_result = shap_explainer.shap_values(
            selected_model_input
        )

        shap_values = np.asarray(shap_result)

        if shap_values.ndim == 2:

            selected_shap = shap_values[0]

        elif shap_values.ndim == 1:

            selected_shap = shap_values

        else:

            raise ValueError(
                f"Unexpected SHAP dimensions: {shap_values.shape}"
            )

        if len(selected_shap) != len(FEATURE_NAMES):

            raise ValueError(
                "SHAP feature count does not match the eight model inputs."
            )

        shap_df = pd.DataFrame({
            "Feature": [
                DISPLAY_NAMES[col]
                for col in FEATURE_NAMES
            ],
            "SHAP value": selected_shap,
        })

        shap_df["Absolute contribution"] = (
            shap_df["SHAP value"].abs()
        )

        shap_df = shap_df.sort_values(
            "Absolute contribution",
            ascending=True
        )

        fig, ax = plt.subplots(
            figsize=(9, 4.5)
        )

        ax.barh(
            shap_df["Feature"],
            shap_df["SHAP value"]
        )

        ax.axvline(
            0,
            linewidth=1
        )

        ax.set_xlabel(
            "SHAP contribution to predicted strength (MPa)"
        )

        ax.set_title(
            "Selected candidate — local feature contributions"
        )

        fig.tight_layout()

        st.pyplot(fig)

        plt.close(fig)

        with st.expander(
            "View exact SHAP contributions"
        ):

            st.dataframe(
                shap_df.sort_values(
                    "Absolute contribution",
                    ascending=False
                ),
                use_container_width=True,
                hide_index=True,
            )

        st.caption(
            "Positive SHAP values push the model prediction above its expected value; "
            "negative values push it below. These are model attributions, not causal effects."
        )

    except Exception as exc:

        st.warning(
            f"SHAP explanation is unavailable: {exc}"
        )

else:

    st.info(
        "SHAP is not installed. Install it with: python -m pip install shap"
    )


# ============================================================
# 13. BIS VERIFICATION AND REQUIRED MATERIAL DATA
# ============================================================

st.markdown("---")

st.header(
    "📚 BIS verification and material-data requirements"
)

st.markdown(
    "The following standards are relevant starting points. Check the BIS catalogue for the "
    "current edition, amendments and project-specific applicability before design or construction."
)


bis_rows = [
    (
        "IS 10262:2019",
        "Concrete mix proportioning — guidelines",
        "Mix design procedure and proportioning checks"
    ),
    (
        "IS 456:2000",
        "Plain and reinforced concrete — code of practice",
        "Structural grade, durability and exposure requirements"
    ),
    (
        "IS 383:2016",
        "Coarse and fine aggregate for concrete — specification",
        "Aggregate quality, grading and suitability"
    ),
    (
        "IS 9103:1999",
        "Concrete admixtures — specification",
        "Admixture qualification and compatibility"
    ),
    (
        "IS 1199 (Part 2):2018",
        "Fresh concrete — determination of consistency",
        "Slump/consistency testing"
    ),
    (
        "IS 1199 (Part 5):2018",
        "Making and curing test specimens",
        "Preparation and curing of test specimens"
    ),
    (
        "IS 516 (Part 1/Sec 1):2021",
        "Hardened concrete — compressive, flexural and split tensile strength",
        "Strength testing; verify current applicable edition"
    ),
]

bis_table = pd.DataFrame(
    bis_rows,
    columns=[
        "Indian Standard",
        "Subject",
        "Application in the workflow"
    ],
)

st.dataframe(
    bis_table,
    use_container_width=True,
    hide_index=True
)


st.subheader(
    "Material and project information not available in the current ML dataset"
)

required_material_data = [
    (
        "Cement",
        "Cement type/grade, specific gravity, source and certificate"
    ),
    (
        "Fine aggregate",
        "Grading/sieve analysis, zone, specific gravity, absorption and moisture"
    ),
    (
        "Coarse aggregate",
        "Grading, nominal size, specific gravity, absorption, moisture and shape"
    ),
    (
        "Water",
        "Water quality and suitability for concrete"
    ),
    (
        "Mineral admixtures",
        "Fly-ash/slag source, specific gravity, fineness and conformity certificates"
    ),
    (
        "Chemical admixture",
        "Type, manufacturer dosage range, solids content and compatibility trials"
    ),
    (
        "Design inputs",
        "Standard deviation, exposure class, minimum cement/binder, maximum water/binder ratio and durability criteria"
    ),
    (
        "Fresh concrete",
        "Slump, air content, density, bleeding, segregation and retention trial results"
    ),
    (
        "Hardened concrete",
        "Trial-batch compressive strength at specified ages and required durability results"
    ),
]

material_table = pd.DataFrame(
    required_material_data,
    columns=[
        "Input",
        "Information required before final mix approval"
    ],
)

st.dataframe(
    material_table,
    use_container_width=True,
    hide_index=True
)


st.markdown(
    "[BIS Standards Portal](https://standards.bis.gov.in/) · "
    "[BIS Know Your Standard](https://www.bis.gov.in/Know-Your-Standard/?lang=en)"
)

st.warning(
    "The candidate list is not an IS 10262 absolute-volume calculation. It does not apply "
    "moisture/absorption corrections, calculate code-based target mean strength from standard "
    "deviation, or verify the IS 456 durability limits. Final proportions require a qualified "
    "mix-design engineer, material characterization and laboratory trial batches."
)


# ============================================================
# 14. MODEL ARTIFACT SUMMARY AND FOOTER
# ============================================================

st.markdown("---")

st.header(
    "💾 Model storage"
)

store1, store2 = st.columns(2)

with store1:

    st.write(
        "**Group-held-out validation model**"
    )

    st.code(
        str(VALIDATION_MODEL_PATH)
    )

    st.caption(
        "Trained on the 80% training formulation groups; "
        "used for held-out candidate scoring."
    )

with store2:

    st.write(
        "**Full-data model**"
    )

    st.code(
        str(FULL_MODEL_PATH)
    )

    st.caption(
        "Trained on all cleaned records; stored separately "
        "and not used for held-out candidate scoring."
    )


st.caption(
    "BIS-oriented Concrete Mix Recommendation Prototype | "
    "CatBoost | Group-held-out validation | SHAP"
)
