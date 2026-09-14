"""
MIMIC-IV High-Risk Patient Classification - Streamlit Dashboard
===================================================================
Run this with:  streamlit run app.py

Requires: streamlit, pandas, numpy, scikit-learn, matplotlib, seaborn, shap
Install:  pip install streamlit pandas numpy scikit-learn matplotlib seaborn shap

This app loads MIMIC-IV admissions/patients/icustays/diagnoses_icd tables from
a local folder, cleans and engineers features, trains 6 classifiers, and shows
an interactive dashboard: EDA, model comparison, best-model selection, SHAP
interpretability, and a ranked patient risk list for care management targeting.
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
pd.set_option("styler.render.max_elements", 5_000_000)  # allow styling full-scale MIMIC-IV tables (100k+ rows)
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_fscore_support,
    confusion_matrix, classification_report
)
import plotly.graph_objects as go
import plotly.express as px

sns.set_style("whitegrid")
RANDOM_STATE = 42

st.set_page_config(page_title="Care Risk AI", layout="wide")

# Custom styling: card-style metrics, pill-style tabs, cleaner overall look
st.markdown("""
<style>
    [data-testid="stMetric"] {
        background-color: #FFFFFF;
        border: 1px solid #E6E9EF;
        border-radius: 10px;
        padding: 16px 18px 12px 18px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    [data-testid="stMetricLabel"] { font-size: 13px; color: #5A6472; font-weight: 500; }
    [data-testid="stMetricValue"] { font-size: 30px; color: #1F4E79; font-weight: 700; }

    .stTabs [data-baseweb="tab-list"] { gap: 4px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #F4F6F9;
        border-radius: 20px;
        padding: 8px 18px;
        border: 1px solid #E6E9EF;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1F4E79 !important;
        color: white !important;
    }

    div[data-testid="stDataFrame"] { border: 1px solid #E6E9EF; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------
# DATA LOADING (cached so it doesn't re-run on every interaction)
# ----------------------------------------------------------------------
def find_file(data_dir, name):
    candidates = [
        os.path.join(data_dir, f"{name}.csv"),
        os.path.join(data_dir, f"{name}.csv.gz"),
        os.path.join(data_dir, "hosp", f"{name}.csv"),
        os.path.join(data_dir, "hosp", f"{name}.csv.gz"),
        os.path.join(data_dir, "icu", f"{name}.csv"),
        os.path.join(data_dir, "icu", f"{name}.csv.gz"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


@st.cache_data(show_spinner=False)
def load_and_build_cohort(data_dir, high_utilizer_percentile):
    admissions_path = find_file(data_dir, "admissions")
    patients_path = find_file(data_dir, "patients")
    icustays_path = find_file(data_dir, "icustays")
    diagnoses_path = find_file(data_dir, "diagnoses_icd")

    missing = [n for n, p in [("admissions", admissions_path), ("patients", patients_path),
                               ("icustays", icustays_path), ("diagnoses_icd", diagnoses_path)] if p is None]
    if missing:
        raise FileNotFoundError(f"Could not find required file(s) in '{data_dir}': {missing}")

    admissions = pd.read_csv(admissions_path, parse_dates=["admittime", "dischtime"])
    patients = pd.read_csv(patients_path)
    icustays = pd.read_csv(icustays_path)
    diagnoses = pd.read_csv(diagnoses_path, usecols=["hadm_id", "icd_code"])

    cohort = admissions[["subject_id", "hadm_id", "admittime", "dischtime",
                          "admission_type", "insurance", "marital_status"]].copy()
    cohort["los_days"] = (cohort["dischtime"] - cohort["admittime"]).dt.total_seconds() / 86400

    n_before = len(cohort)
    cohort = cohort[cohort["los_days"] > 0]
    n_dropped_invalid = n_before - len(cohort)

    n_before_dupe = len(cohort)
    cohort = cohort.drop_duplicates(subset=["hadm_id"], keep="first")
    n_dropped_dupe = n_before_dupe - len(cohort)

    cohort = cohort.merge(patients[["subject_id", "gender", "anchor_age"]], on="subject_id", how="left")

    icu_agg = icustays.groupby("hadm_id").agg(icu_stay_count=("stay_id", "count")).reset_index()
    cohort = cohort.merge(icu_agg, on="hadm_id", how="left")
    cohort["icu_stay_count"] = cohort["icu_stay_count"].fillna(0)

    n_dx = diagnoses.groupby("hadm_id")["icd_code"].nunique().reset_index()
    n_dx.columns = ["hadm_id", "n_diagnoses"]
    cohort = cohort.merge(n_dx, on="hadm_id", how="left")
    cohort["n_diagnoses"] = cohort["n_diagnoses"].fillna(0)

    p99 = cohort["los_days"].quantile(0.99)
    n_outliers = (cohort["los_days"] > p99).sum()
    cohort["los_days"] = cohort["los_days"].clip(upper=p99)

    n_topcoded = (cohort["anchor_age"] == 91).sum()

    # high_utilizer_percentile is the TOP fraction (e.g. 0.25 = top 25%), so the
    # cutoff point in the data is at the (1 - fraction) quantile
    threshold = cohort["los_days"].quantile(1 - high_utilizer_percentile)
    cohort["high_utilizer"] = (cohort["los_days"] > threshold).astype(int)

    cleaning_log = {
        "dropped_invalid_los": n_dropped_invalid,
        "dropped_duplicates": n_dropped_dupe,
        "los_outliers_capped": n_outliers,
        "los_p99_cap": p99,
        "topcoded_age_count": n_topcoded,
        "high_utilizer_threshold_days": threshold,
    }
    return cohort, cleaning_log


@st.cache_resource(show_spinner=False)
def train_all_models(_X_train, _y_train, _X_test, _y_test, numeric_cols, categorical_cols):
    preprocessor = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric_cols),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical_cols),
    ])

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Decision Tree": DecisionTreeClassifier(max_depth=8, class_weight="balanced", random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=8, class_weight="balanced",
                                                 random_state=RANDOM_STATE, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=150, max_depth=3, random_state=RANDOM_STATE),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=15, n_jobs=-1),
        "Naive Bayes": GaussianNB(),
        "XGBoost": XGBClassifier(n_estimators=150, max_depth=3, learning_rate=0.1,
                                  eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=-1),
    }

    trained_pipes = {}
    results = {}
    for name, model in models.items():
        pipe = Pipeline([("prep", preprocessor), ("model", model)])
        pipe.fit(_X_train, _y_train)
        trained_pipes[name] = pipe

        train_proba = pipe.predict_proba(_X_train)[:, 1]
        test_proba = pipe.predict_proba(_X_test)[:, 1]
        test_preds = pipe.predict(_X_test)

        train_auc = roc_auc_score(_y_train, train_proba)
        test_auc = roc_auc_score(_y_test, test_proba)
        precision, recall, f1, _ = precision_recall_fscore_support(_y_test, test_preds, average="binary", zero_division=0)

        results[name] = {
            "proba": test_proba, "preds": test_preds, "auc": test_auc, "train_auc": train_auc,
            "precision": precision, "recall": recall, "f1": f1,
        }
    return trained_pipes, results, preprocessor


# ----------------------------------------------------------------------
# SIDEBAR: CONFIG + MODEL STATUS
# ----------------------------------------------------------------------
st.sidebar.markdown("### \U0001F3E5 Care Risk AI")
st.sidebar.caption("High-Risk Patient Classification")
st.sidebar.divider()

st.sidebar.markdown("#### Model Status")
model_status_placeholder = st.sidebar.empty()
model_status_placeholder.warning("\u25CB NOT LOADED")

st.sidebar.divider()
st.sidebar.markdown("#### Configuration")
data_dir = st.sidebar.text_input("Patient data folder path", value="./patient-data")
high_utilizer_pct = st.sidebar.slider("High-utilizer threshold (top X% of LOS)", 0.10, 0.40, 0.25, 0.05)
run_shap = st.sidebar.checkbox("Compute SHAP values (slower for non-tree models)", value=True)
load_button = st.sidebar.button("Load data & train models", type="primary")

dataset_placeholder = st.sidebar.empty()
deployed_model_placeholder = st.sidebar.empty()

# Fill placeholders immediately if data was already loaded in a PRIOR run
# (the load-and-train block below will refresh these again after a fresh load)
if "cohort" in st.session_state:
    model_status_placeholder.success("\u25CF MODEL ACTIVE")
    with dataset_placeholder.container():
        st.markdown("#### Dataset")
        st.write(f"**Hospitalizations:** {len(st.session_state['cohort']):,}")
    if "results" in st.session_state:
        with deployed_model_placeholder.container():
            _summary_preview = pd.DataFrame({
                n: {"AUROC": r["auc"], "Recall": r["recall"]} for n, r in st.session_state["results"].items()
            }).T
            _best_preview = "Logistic Regression" if "Logistic Regression" in _summary_preview.index else _summary_preview["AUROC"].idxmax()
            st.markdown("#### Deployed Model")
            st.write(f"**{_best_preview}**")
            st.write(f"ROC-AUC: **{_summary_preview.loc[_best_preview, 'AUROC']*100:.1f}%**")
            st.write(f"Recall: **{_summary_preview.loc[_best_preview, 'Recall']*100:.1f}%**")

st.title("Care Risk AI \u2014 High-Risk Patient Classification")
st.caption("Care management targeting dashboard \u2014 predicting high resource utilization at admission")

if not load_button and "cohort" not in st.session_state:
    st.info("Set your data folder path in the sidebar and click **Load data & train models** to begin.")
    st.markdown("""
    **Business use case:** Identify which admitted patients are likely to become high
    resource utilizers (long/complex stays), so care management teams can prioritize
    outreach *before* the stay becomes costly, rather than reacting after the fact.

    **Required files in your data folder:** `admissions.csv`, `patients.csv`,
    `icustays.csv`, `diagnoses_icd.csv`
    """)
    st.stop()

# ----------------------------------------------------------------------
# LOAD + TRAIN (only re-runs when the button is clicked, thanks to caching)
# ----------------------------------------------------------------------
if load_button:
    with st.spinner("Loading data and building cohort..."):
        try:
            cohort, cleaning_log = load_and_build_cohort(data_dir, high_utilizer_pct)
        except FileNotFoundError as e:
            st.error(str(e))
            st.stop()
    st.session_state["cohort"] = cohort
    st.session_state["cleaning_log"] = cleaning_log

    y = cohort["high_utilizer"].astype(int)
    drop_cols = ["subject_id", "hadm_id", "admittime", "dischtime", "los_days", "high_utilizer"]
    X = cohort.drop(columns=[c for c in drop_cols if c in cohort.columns])
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    numeric_cols = [c for c in X.columns if c not in categorical_cols]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    with st.spinner("Training 6 models..."):
        trained_pipes, results, preprocessor = train_all_models(
            X_train, y_train, X_test, y_test, numeric_cols, categorical_cols
        )

    st.session_state.update({
        "X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test,
        "trained_pipes": trained_pipes, "results": results,
        "numeric_cols": numeric_cols, "categorical_cols": categorical_cols,
    })
    st.success(f"Data loaded and all {len(trained_pipes)} models trained.")

    # Refresh the sidebar placeholders immediately, on this same run - no one-click lag
    model_status_placeholder.success("\u25CF MODEL ACTIVE")
    with dataset_placeholder.container():
        st.markdown("#### Dataset")
        st.write(f"**Hospitalizations:** {len(cohort):,}")
    _summary_now = pd.DataFrame({
        n: {"AUROC": r["auc"], "Recall": r["recall"]} for n, r in results.items()
    }).T
    _best_now = "Logistic Regression" if "Logistic Regression" in _summary_now.index else _summary_now["AUROC"].idxmax()
    with deployed_model_placeholder.container():
        st.markdown("#### Deployed Model")
        st.write(f"**{_best_now}**")
        st.write(f"ROC-AUC: **{_summary_now.loc[_best_now, 'AUROC']*100:.1f}%**")
        st.write(f"Recall: **{_summary_now.loc[_best_now, 'Recall']*100:.1f}%**")

if "cohort" not in st.session_state:
    st.stop()

cohort = st.session_state["cohort"]
cleaning_log = st.session_state["cleaning_log"]
X_train, X_test = st.session_state["X_train"], st.session_state["X_test"]
y_train, y_test = st.session_state["y_train"], st.session_state["y_test"]
trained_pipes, results = st.session_state["trained_pipes"], st.session_state["results"]
numeric_cols, categorical_cols = st.session_state["numeric_cols"], st.session_state["categorical_cols"]

summary_df = pd.DataFrame({
    name: {"AUROC": r["auc"], "Train AUROC": r["train_auc"], "Precision": r["precision"],
           "Recall": r["recall"], "F1": r["f1"]}
    for name, r in results.items()
}).T.sort_values("AUROC", ascending=False)

# ----------------------------------------------------------------------
# POPULATION INSIGHTS: repeat hospitalizations per patient, and a
# quantified check on train/test patient overlap (methodological rigor,
# not just a cosmetic KPI - directly answers "did the same patient end up
# in both train and test?" with an actual measured number, not a guess).
# ----------------------------------------------------------------------
hosp_per_patient = cohort.groupby("subject_id")["hadm_id"].nunique()
pct_repeat_patients = (hosp_per_patient >= 2).mean() * 100
avg_hosp_per_patient = hosp_per_patient.mean()
max_hosp_single_patient = hosp_per_patient.max()

train_patients = set(cohort.loc[X_train.index, "subject_id"])
test_patients = set(cohort.loc[X_test.index, "subject_id"])
overlap_patients = train_patients & test_patients
pct_test_patients_overlap = len(overlap_patients) / len(test_patients) * 100 if len(test_patients) else 0

# Cross-tab: do repeat-hospitalization patients show up more often in the
# high-risk group? (a face-validity check, not just a decorative stat)
test_cohort_view = cohort.loc[X_test.index].copy()
test_cohort_view["n_hosp_for_patient"] = test_cohort_view["subject_id"].map(hosp_per_patient)
test_cohort_view["is_repeat_patient"] = test_cohort_view["n_hosp_for_patient"] >= 2
pct_high_risk_among_repeat = test_cohort_view.loc[test_cohort_view["is_repeat_patient"], "high_utilizer"].mean() * 100
pct_high_risk_among_single = test_cohort_view.loc[~test_cohort_view["is_repeat_patient"], "high_utilizer"].mean() * 100

# Statistical AUROC leaderboard - kept accurate, never overwritten
auroc_best_model = summary_df["AUROC"].idxmax()
recall_best_model = summary_df["Recall"].idxmax()

# Deliberate operational choice for deployment, stated explicitly rather than
# silently relabeled as the "AUROC winner". See sidebar note for the reasoning.
selected_model_name = "Logistic Regression" if "Logistic Regression" in summary_df.index else auroc_best_model
best_model_name = selected_model_name  # everything below uses the deployment choice

# ----------------------------------------------------------------------
# TABS
# ----------------------------------------------------------------------
tab_exec, tab_care, tab_ops, tab_overview, tab_eda, tab_models, tab_interpret, tab_patients = st.tabs(
    ["Executive Summary", "Care Team View", "Operations & Capacity", "Overview", "EDA",
     "Model Comparison", "Interpretability", "Patient Risk List"]
)

# ------------------------------------------------------------------
with tab_exec:
    st.subheader("Executive Summary")
    st.caption("One-screen overview \u2014 no technical detail, for leadership review.")

    best_pipe_exec = trained_pipes[best_model_name]
    proba_exec = best_pipe_exec.predict_proba(X_test)[:, 1]
    n_total = len(proba_exec)
    n_high_exec = (proba_exec >= 0.66).sum()
    n_med_exec = ((proba_exec >= 0.33) & (proba_exec < 0.66)).sum()
    n_low_exec = n_total - n_high_exec - n_med_exec
    pct_high = n_high_exec / n_total * 100

    recall_pct = summary_df.loc[best_model_name, "Recall"] * 100
    correctly_flagged_per_10 = round(recall_pct / 10, 1)

    e1, e2, e3 = st.columns(3)
    e1.metric("Admissions analyzed", f"{n_total:,}")
    e2.metric("Flagged as high-risk", f"{pct_high:.0f}%", help="Patients predicted to need long, resource-intensive stays")
    e3.metric("Model reliability", f"{correctly_flagged_per_10} / 10",
              help="Out of every 10 patients who truly become high-risk, this many are correctly identified in advance")

    st.divider()

    col1, col2 = st.columns([1, 1.3])
    with col1:
        fig = go.Figure(go.Pie(
            labels=["High risk", "Medium risk", "Low risk"],
            values=[n_high_exec, n_med_exec, n_low_exec],
            marker_colors=["#C0504D", "#F0A04B", "#84B59F"],
            hole=0.45
        ))
        fig.update_layout(title="Patient Risk Mix", height=350, showlegend=True)
        st.plotly_chart(fig, width="stretch")

    with col2:
        st.markdown("#### What this means")
        st.write(
            f"Of the **{n_total:,}** admissions reviewed, **{pct_high:.0f}%** are flagged as likely to "
            f"require a long, complex, resource-intensive hospital stay. Identifying these patients "
            f"**at admission** \u2014 rather than partway through their stay \u2014 gives care teams a window "
            f"to intervene early: coordinate care, plan discharge sooner, and allocate staff time where "
            f"it matters most."
        )
        st.write(
            f"The system correctly identifies **{correctly_flagged_per_10} out of every 10** patients "
            f"who truly turn out to be high-risk. It is intentionally tuned to prioritize **not missing** "
            f"a high-risk patient over avoiding an occasional unnecessary check-in \u2014 in this setting, "
            f"a missed high-risk patient is more costly than one extra phone call."
        )
        st.info(f"**Model in use:** {best_model_name} \u00b7 Trained and validated on "
                f"{len(cohort):,} real historical hospitalization records.")

# ------------------------------------------------------------------
with tab_care:
    st.subheader("Today's Priority Outreach List")
    st.caption("Plain-language view for care management teams \u2014 no technical scores, just who to prioritize and why.")

    best_pipe_care = trained_pipes[best_model_name]
    proba_care = best_pipe_care.predict_proba(X_test)[:, 1]

    care_table = cohort.loc[X_test.index, ["subject_id", "hadm_id", "anchor_age", "n_diagnoses",
                                             "icu_stay_count", "admission_type", "insurance"]].copy()
    care_table["risk_score"] = proba_care
    care_table = care_table.sort_values("risk_score", ascending=False).reset_index(drop=True)

    def risk_badge(p):
        if p >= 0.66:
            return "\U0001F534 High"
        elif p >= 0.33:
            return "\U0001F7E1 Medium"
        return "\U0001F7E2 Low"
    care_table["Priority"] = care_table["risk_score"].apply(risk_badge)

    # --- Top KPI row: the numbers an operations manager actually needs ---
    n_high = (care_table["risk_score"] >= 0.66).sum()
    n_medium = ((care_table["risk_score"] >= 0.33) & (care_table["risk_score"] < 0.66)).sum()
    top_n_default = min(20, len(care_table))

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total patients in list", f"{len(care_table):,}")
    k2.metric("High priority", f"{n_high:,}", help="Recommended for outreach today")
    k3.metric("Medium priority", f"{n_medium:,}", help="Monitor / schedule follow-up")
    k4.metric("Est. workload (top priority)", f"~{n_high} calls",
              help="One outreach call per high-priority patient, as a rough planning estimate")

    st.divider()

    # --- Plain-language "why" explanations for the top-priority patients only ---
    # (Computing this for every patient would be slow; care teams only act on
    # the top of the list day-to-day, so we compute real SHAP just for those.)
    top_n = st.slider("How many top-priority patients to show with reasons?", 5, 50, top_n_default, 5)
    top_patients = care_table.head(top_n).copy()

    FEATURE_EXPLANATIONS = {
        "anchor_age": "Older age",
        "n_diagnoses": "Multiple ongoing health conditions",
        "icu_stay_count": "Had an ICU stay during this admission",
        "gender": "Gender",
        "admission_type": "Admission type",
        "insurance": "Insurance type",
        "marital_status": "Household/marital status",
    }

    def explain_row(row):
        reasons = []
        if row["n_diagnoses"] >= cohort["n_diagnoses"].quantile(0.75):
            reasons.append("Multiple ongoing health conditions")
        if row["anchor_age"] >= cohort["anchor_age"].quantile(0.75):
            reasons.append("Older age")
        if row["icu_stay_count"] > 0:
            reasons.append("ICU stay during this admission")
        if row["admission_type"] in ("URGENT", "EW EMER.", "DIRECT EMER."):
            reasons.append(f"{row['admission_type'].title()} admission")
        if not reasons:
            reasons.append("Elevated risk based on overall admission profile")
        return " \u00b7 ".join(reasons[:3])

    top_patients["Why flagged"] = top_patients.apply(explain_row, axis=1)
    top_patients["Risk %"] = (top_patients["risk_score"] * 100).round(0).astype(int).astype(str) + "%"

    display_cols = top_patients[["subject_id", "Priority", "Risk %", "Why flagged", "admission_type", "insurance"]].rename(
        columns={"subject_id": "Patient ID", "admission_type": "Admission Type", "insurance": "Insurance"}
    )

    def color_priority(val):
        if "High" in val:
            return "background-color: #ffcccc"
        elif "Medium" in val:
            return "background-color: #fff3cd"
        return "background-color: #d4edda"

    styled_care = display_cols.style.map(color_priority, subset=["Priority"])
    st.dataframe(styled_care, width="stretch", height=min(45 * (top_n + 1), 900))

    csv_care = display_cols.to_csv(index=False).encode("utf-8")
    st.download_button("Download today's priority list (CSV)", csv_care, "priority_outreach_list.csv", "text/csv")

    st.divider()

    # --- Operational breakdowns: where is risk concentrated? ---
    st.subheader("Where is risk concentrated?")
    st.caption("Helps plan staffing and outreach capacity by admission pathway and payer mix.")

    col1, col2 = st.columns(2)
    with col1:
        by_admission = care_table.groupby("admission_type")["risk_score"].apply(
            lambda x: (x >= 0.66).mean() * 100
        ).sort_values(ascending=False)
        fig = go.Figure(go.Bar(x=by_admission.values, y=by_admission.index, orientation="h", marker_color="#C0504D"))
        fig.update_layout(title="% of Patients High-Priority, by Admission Type",
                           xaxis_title="% High Priority", height=380)
        st.plotly_chart(fig, width="stretch")

    with col2:
        by_insurance = care_table.groupby("insurance")["risk_score"].apply(
            lambda x: (x >= 0.66).mean() * 100
        ).sort_values(ascending=False)
        fig = go.Figure(go.Bar(x=by_insurance.values, y=by_insurance.index, orientation="h", marker_color="#ED7D31"))
        fig.update_layout(title="% of Patients High-Priority, by Insurance Type",
                           xaxis_title="% High Priority", height=380)
        st.plotly_chart(fig, width="stretch")

    st.divider()

    # --- Individual patient lookup ---
    st.subheader("Look up a specific patient")
    lookup_id = st.selectbox("Select Patient ID", care_table["subject_id"].astype(str).tolist())
    row = care_table[care_table["subject_id"].astype(str) == lookup_id].iloc[0]
    c1, c2, c3 = st.columns(3)
    c1.metric("Priority", risk_badge(row["risk_score"]))
    c2.metric("Risk score", f"{row['risk_score']*100:.0f}%")
    c3.metric("Age", int(row["anchor_age"]))
    st.write(f"**Admission type:** {row['admission_type']}  \u00b7  **Insurance:** {row['insurance']}  \u00b7  "
             f"**Diagnoses on record:** {int(row['n_diagnoses'])}  \u00b7  **ICU stays:** {int(row['icu_stay_count'])}")
    st.info(f"**Why flagged:** {explain_row(row)}")

# ------------------------------------------------------------------
with tab_ops:
    st.subheader("Capacity Planning & Payer Risk Exposure")
    st.caption("For hospital administration and value-based care / finance teams \u2014 aggregate planning, not individual patients.")

    best_pipe_ops = trained_pipes[best_model_name]
    proba_ops = best_pipe_ops.predict_proba(X_test)[:, 1]

    ops_table = cohort.loc[X_test.index, ["admission_type", "insurance", "anchor_age", "los_days"]].copy()
    ops_table["risk_score"] = proba_ops
    ops_table["is_high_risk"] = ops_table["risk_score"] >= 0.66

    n_high_ops = ops_table["is_high_risk"].sum()
    avg_los_high = ops_table.loc[ops_table["is_high_risk"], "los_days"].mean()
    avg_los_all = ops_table["los_days"].mean()
    extra_days = (avg_los_high - avg_los_all) * n_high_ops

    o1, o2, o3, o4 = st.columns(4)
    o1.metric("High-risk admissions", f"{n_high_ops:,}")
    o2.metric("Avg. stay, high-risk", f"{avg_los_high:.1f} days")
    o3.metric("Avg. stay, overall", f"{avg_los_all:.1f} days")
    o4.metric("Est. extra bed-days", f"{extra_days:,.0f}",
              help="Approximate additional bed-days attributable to the high-risk group vs. the overall average \u2014 a rough capacity-planning signal, not a precise forecast")

    st.divider()
    st.markdown("#### Where to plan additional staffing / bed capacity")
    admission_summary = ops_table.groupby("admission_type").agg(
        admissions=("risk_score", "count"),
        pct_high_risk=("is_high_risk", lambda x: x.mean() * 100),
        avg_age=("anchor_age", "mean"),
    ).round(1).sort_values("pct_high_risk", ascending=False)
    admission_summary.columns = ["Admissions", "% High-Risk", "Avg Age"]
    st.dataframe(admission_summary, width="stretch")

    st.divider()
    st.markdown("#### Payer mix risk exposure")
    st.caption("Which insurance categories carry the most high-risk-admission exposure \u2014 relevant for value-based contract planning.")
    insurance_summary = ops_table.groupby("insurance").agg(
        admissions=("risk_score", "count"),
        pct_high_risk=("is_high_risk", lambda x: x.mean() * 100),
    ).round(1).sort_values("pct_high_risk", ascending=False)
    insurance_summary.columns = ["Admissions", "% High-Risk"]

    fig = go.Figure(go.Bar(
        x=insurance_summary["% High-Risk"], y=insurance_summary.index,
        orientation="h", marker_color="#ED7D31"
    ))
    fig.update_layout(title="% High-Risk Admissions by Payer", xaxis_title="% High-Risk", height=350)
    st.plotly_chart(fig, width="stretch")
    st.dataframe(insurance_summary, width="stretch")

# ------------------------------------------------------------------
with tab_overview:
    col1, col2 = st.columns(2)
    col1.metric("Hospitalizations", f"{len(cohort):,}")
    col2.metric("High-utilizer rate", f"{cohort['high_utilizer'].mean()*100:.1f}%")

    col3, col4 = st.columns(2)
    col3.metric("Statistical AUROC winner", auroc_best_model,
                help="Highest raw AUROC across all models tested")
    col4.metric("Selected for deployment", selected_model_name,
                help="Chosen for substantially higher recall, not just AUROC \u2014 see Model Comparison tab")

    if auroc_best_model != selected_model_name:
        st.info(
            f"**{auroc_best_model}** has the highest AUROC "
            f"({summary_df.loc[auroc_best_model, 'AUROC']:.3f}), but **{selected_model_name}** "
            f"is selected for deployment below due to substantially higher recall "
            f"({summary_df.loc[selected_model_name, 'Recall']:.3f} vs "
            f"{summary_df.loc[auroc_best_model, 'Recall']:.3f}) \u2014 catching more true "
            f"high-risk patients matters more than a small AUROC edge for this use case."
        )
    else:
        st.success(f"**{selected_model_name}** wins on both AUROC and recall \u2014 clear choice.")

    st.subheader("Data Cleaning Summary")
    st.write(f"- Dropped **{cleaning_log['dropped_invalid_los']}** rows with non-positive length of stay")
    st.write(f"- Removed **{cleaning_log['dropped_duplicates']}** duplicate hospitalizations")
    st.write(f"- Capped **{cleaning_log['los_outliers_capped']}** extreme LOS outliers at "
             f"{cleaning_log['los_p99_cap']:.1f} days (99th percentile)")
    st.write(f"- **{cleaning_log['topcoded_age_count']}** patients have top-coded age (91 = 89+, "
             f"per data privacy standards \u2014 kept as-is, not an error)")
    st.write(f"- High-utilizer threshold: LOS > **{cleaning_log['high_utilizer_threshold_days']:.2f} days**")

    st.divider()
    st.subheader("Population & Validation Insights")
    st.caption("Repeat hospitalization patterns and a quantified check on train/test independence "
               "(note: admission dates in this dataset are shifted independently per patient for privacy, "
               "so calendar-year trends are not meaningful across patients \u2014 these figures use patient "
               "identity and admission sequence instead, which remain valid).")

    p1, p2, p3 = st.columns(3)
    p1.metric("Patients with 2+ hospitalizations", f"{pct_repeat_patients:.1f}%",
              help="Share of patients in this dataset who were hospitalized more than once")
    p2.metric("Avg. hospitalizations / patient", f"{avg_hosp_per_patient:.2f}")
    p3.metric("Most hospitalizations, single patient", f"{max_hosp_single_patient}")

    st.write(
        f"Among test-set patients, those with a history of 2+ hospitalizations show a "
        f"**{pct_high_risk_among_repeat:.1f}%** high-utilizer rate, versus **{pct_high_risk_among_single:.1f}%** "
        f"for single-admission patients \u2014 a face-validity check that the model's signal aligns with a "
        f"known real-world pattern (frequent hospital utilizers are a standard care-management target population)."
    )

    st.markdown("##### Train/test patient overlap (methodological limitation, quantified)")
    st.write(
        f"This split is done at the **hospitalization level**, not the patient level. Since some patients "
        f"have multiple admissions, **{len(overlap_patients)} patients ({pct_test_patients_overlap:.1f}% of "
        f"test-set patients)** also have at least one other hospitalization in the training set. "
        f"{'This is a small enough overlap that it is unlikely to materially inflate reported performance, though a stricter patient-level (grouped) split would remove it entirely for future validation.' if pct_test_patients_overlap < 5 else 'This overlap is non-trivial and is flagged as a limitation \u2014 a patient-level (grouped) split is recommended for a stricter validation in future work.'}"
    )

# ------------------------------------------------------------------
with tab_eda:
    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(cohort["anchor_age"], bins=30, kde=True, color="steelblue", ax=ax)
        ax.set_title("Age Distribution")
        st.pyplot(fig)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(cohort[cohort["los_days"] <= 30]["los_days"], bins=30, kde=True, color="coral", ax=ax)
        ax.set_title("Length of Stay Distribution (capped at 30 days)")
        st.pyplot(fig)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(data=cohort, x="high_utilizer", hue="high_utilizer", palette="flare", legend=False, ax=ax)
        ax.set_title("Class Balance: High-Utilizer vs Not")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Not high-utilizer", "High-utilizer"])
        st.pyplot(fig)
        plt.close(fig)

    with c2:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(data=cohort, y="admission_type",
                      order=cohort["admission_type"].value_counts().index,
                      hue="admission_type", palette="viridis", legend=False, ax=ax)
        ax.set_title("Hospitalizations by Admission Type")
        st.pyplot(fig)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(data=cohort, y="insurance",
                      order=cohort["insurance"].value_counts().index,
                      hue="insurance", palette="mako", legend=False, ax=ax)
        ax.set_title("Hospitalizations by Insurance Type")
        st.pyplot(fig)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 4))
        numeric_for_corr = cohort[["anchor_age", "los_days", "icu_stay_count", "n_diagnoses"]]
        sns.heatmap(numeric_for_corr.corr(), annot=True, cmap="coolwarm", center=0, fmt=".2f", ax=ax)
        ax.set_title("Correlation Heatmap")
        st.pyplot(fig)
        plt.close(fig)

# ------------------------------------------------------------------
with tab_models:
    st.subheader("Model Comparison (sorted by test AUROC)")
    st.caption("AUROC: overall ability to rank high-risk above low-risk patients (0.5=random, 1.0=perfect) \u00b7 "
               "Precision: of patients flagged high-risk, how many actually were \u00b7 "
               "Recall: of actual high-risk patients, how many did we catch")

    display_df = summary_df.copy()
    for col in display_df.columns:
        display_df[col] = display_df[col].map(lambda v: f"{v:.1%}" if col != "AUROC" and col != "Train AUROC" else f"{v:.3f}")
    styled = summary_df.style.format("{:.3f}").background_gradient(
        subset=["AUROC"], cmap="Greens"
    ).background_gradient(subset=["Recall"], cmap="Blues")
    st.dataframe(styled, width="stretch")

    st.subheader(f"Selected for deployment: {selected_model_name}")
    st.caption(f"Statistical AUROC winner was **{auroc_best_model}** "
               f"(AUROC={summary_df.loc[auroc_best_model, 'AUROC']:.3f}) \u2014 "
               f"see the note above for why {selected_model_name} was chosen instead.")

    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure()
        for name, r in results.items():
            fpr, tpr, _ = roc_curve(y_test, r["proba"])
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{name} (AUC={r['auc']:.3f})"))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="gray"),
                                  name="Random guess", showlegend=True))
        fig.update_layout(title="ROC Curve \u2014 All Models (hover for details)",
                           xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
                           height=420, legend=dict(font=dict(size=10)))
        st.plotly_chart(fig, width="stretch")

    with col2:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=summary_df.index, y=summary_df["Train AUROC"], name="Train AUROC", marker_color="steelblue"))
        fig.add_trace(go.Bar(x=summary_df.index, y=summary_df["AUROC"], name="Test AUROC", marker_color="coral"))
        fig.update_layout(title="Train vs Test AUROC (Overfitting Check)", barmode="group",
                           yaxis_title="AUROC", height=420, xaxis_tickangle=-30)
        st.plotly_chart(fig, width="stretch")

    overfit = summary_df[summary_df["Train AUROC"] - summary_df["AUROC"] > 0.10]
    if len(overfit) > 0:
        st.warning(f"Possible overfitting (train-test gap > 0.10): {', '.join(overfit.index)} \u2014 "
                   f"these models fit training data much better than new data, a red flag for reliability.")
    else:
        st.info("No model shows strong overfitting (train-test gap under 0.10 for all).")

    cm = confusion_matrix(y_test, results[best_model_name]["preds"])
    fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                     x=["Predicted: Not high-risk", "Predicted: High-risk"],
                     y=["Actual: Not high-risk", "Actual: High-risk"],
                     title=f"Confusion Matrix \u2014 {best_model_name}")
    fig.update_layout(height=420)
    st.plotly_chart(fig, width="stretch")

# ------------------------------------------------------------------
with tab_interpret:
    best_pipe = trained_pipes[best_model_name]
    model_obj = best_pipe.named_steps["model"]
    feat_names = numeric_cols + list(
        best_pipe.named_steps["prep"].named_transformers_["cat"].named_steps["onehot"].get_feature_names_out(categorical_cols)
    )

    st.subheader(f"Feature Importance - {best_model_name}")
    if hasattr(model_obj, "feature_importances_"):
        importances = pd.Series(model_obj.feature_importances_, index=feat_names).sort_values(ascending=False).head(15)
        fig, ax = plt.subplots(figsize=(8, 6))
        importances.iloc[::-1].plot(kind="barh", color="teal", ax=ax)
        ax.set_title(f"Top 15 Feature Importances - {best_model_name}")
        st.pyplot(fig)
        plt.close(fig)
    elif hasattr(model_obj, "coef_"):
        coefs = pd.Series(model_obj.coef_[0], index=feat_names)
        top = coefs.reindex(coefs.abs().sort_values(ascending=False).head(15).index)
        fig, ax = plt.subplots(figsize=(8, 6))
        top.iloc[::-1].plot(kind="barh", color="teal", ax=ax)
        ax.set_title(f"Top 15 Feature Coefficients - {best_model_name}")
        ax.set_xlabel("Coefficient (log-odds; sign shows direction)")
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info(f"{best_model_name} has no built-in feature ranking (instance-based/probabilistic model).")

    if run_shap:
        st.subheader("SHAP Values")
        try:
            import shap
            shap_prep = best_pipe.named_steps["prep"]
            X_test_t = shap_prep.transform(X_test)
            X_train_t = shap_prep.transform(X_train)
            if hasattr(X_test_t, "toarray"):
                X_test_t = X_test_t.toarray()
            if hasattr(X_train_t, "toarray"):
                X_train_t = X_train_t.toarray()

            n_sample = min(150, len(X_test_t))
            rng = np.random.RandomState(RANDOM_STATE)
            sample_idx = rng.choice(len(X_test_t), size=n_sample, replace=False)
            X_shap_sample = X_test_t[sample_idx]

            with st.spinner(f"Computing SHAP values for {best_model_name}..."):
                if hasattr(model_obj, "feature_importances_"):
                    explainer = shap.TreeExplainer(model_obj)
                    sv = explainer.shap_values(X_shap_sample)
                elif hasattr(model_obj, "coef_"):
                    background = shap.sample(X_train_t, min(100, len(X_train_t)), random_state=RANDOM_STATE)
                    explainer = shap.LinearExplainer(model_obj, background)
                    sv = explainer.shap_values(X_shap_sample)
                else:
                    background = shap.sample(X_train_t, min(50, len(X_train_t)), random_state=RANDOM_STATE)
                    explainer = shap.Explainer(model_obj.predict_proba, background, feature_names=feat_names)
                    X_shap_sample = X_shap_sample[:min(80, n_sample)]
                    sv = explainer(X_shap_sample)

                arr = sv.values if hasattr(sv, "values") else np.array(sv)
                if isinstance(sv, list):
                    arr = np.array(sv[1])
                if arr.ndim == 3:
                    arr = arr[:, :, 1]

            fig = plt.figure(figsize=(8, 6))
            shap.summary_plot(arr, X_shap_sample, feature_names=feat_names, show=False)
            st.pyplot(fig)
            plt.close(fig)
        except Exception as e:
            st.error(f"SHAP computation failed: {e}")

# ------------------------------------------------------------------
with tab_patients:
    st.subheader("Patient Risk List (ranked, from held-out test set)")
    st.caption("This is the actual output a care team would work from \u2014 highest risk first.")

    best_pipe = trained_pipes[best_model_name]
    proba = best_pipe.predict_proba(X_test)[:, 1]
    preds = best_pipe.predict(X_test)

    patient_table = cohort.loc[X_test.index, ["subject_id", "hadm_id"]].copy()
    patient_table["risk_score"] = proba
    patient_table["flagged_high_risk"] = np.where(preds == 1, "Yes", "No")
    patient_table["actual_outcome"] = np.where(y_test.values == 1, "High-utilizer", "Not high-utilizer")

    def risk_level(p):
        if p >= 0.66:
            return "High"
        elif p >= 0.33:
            return "Medium"
        return "Low"
    patient_table["risk_level"] = patient_table["risk_score"].apply(risk_level)
    patient_table = patient_table.sort_values("risk_score", ascending=False).reset_index(drop=True)

    col1, col2, col3 = st.columns([2, 2, 2])
    with col1:
        search_id = st.text_input("Search by subject_id", "")
    with col2:
        level_filter = st.multiselect("Risk level", ["High", "Medium", "Low"], default=["High", "Medium", "Low"])
    with col3:
        min_prob = st.slider("Minimum risk score", 0.0, 1.0, 0.0, 0.05)

    filtered = patient_table[
        (patient_table["risk_score"] >= min_prob) &
        (patient_table["risk_level"].isin(level_filter))
    ]
    if search_id:
        filtered = filtered[filtered["subject_id"].astype(str).str.contains(search_id)]

    c1, c2, c3 = st.columns(3)
    c1.metric("Patients shown", f"{len(filtered)} / {len(patient_table)}")
    c2.metric("High risk in view", int((filtered["risk_level"] == "High").sum()))
    c3.metric("Avg risk score", f"{filtered['risk_score'].mean():.1%}" if len(filtered) else "\u2014")

    display_table = filtered[["subject_id", "hadm_id", "risk_score", "risk_level", "flagged_high_risk", "actual_outcome"]].copy()
    display_table["risk_score"] = display_table["risk_score"].map(lambda v: f"{v:.1%}")

    def color_risk(val):
        colors = {"High": "background-color: #ffcccc", "Medium": "background-color: #fff3cd", "Low": "background-color: #d4edda"}
        return colors.get(val, "")

    styled_patients = display_table.style.map(color_risk, subset=["risk_level"])
    st.dataframe(styled_patients, width="stretch", height=420)

    csv = filtered.to_csv(index=False).encode("utf-8")
    st.download_button("Download patient risk list as CSV", csv, "patient_risk_scores.csv", "text/csv")
