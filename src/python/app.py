import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import os
import io

# Set working directory to project root so all relative paths work
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    confusion_matrix
)
import seaborn as sns
import matplotlib.pyplot as plt

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="Airline Passenger Satisfaction",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -------------------------------
# Custom CSS — Modern Dashboard Style
# -------------------------------
st.markdown("""
<style>
/* --- Global --- */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

.main .block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}

/* --- Hero banner --- */
.hero-banner {
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1rem;
    color: white;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -50%;
    width: 100%;
    height: 200%;
    background: radial-gradient(circle, rgba(255,255,255,0.05) 0%, transparent 70%);
}
.hero-banner h1 {
    font-size: 1.5rem;
    font-weight: 700;
    margin-bottom: 0.15rem;
    color: white !important;
}
.hero-banner p {
    font-size: 0.85rem;
    opacity: 0.85;
    margin-bottom: 0;
    color: #e0e0e0;
}

/* --- Stat cards --- */
.stat-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 12px;
    padding: 1.2rem 1rem;
    text-align: center;
    color: white;
    margin-bottom: 0.5rem;
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    transition: transform 0.2s ease;
}
.stat-card:hover {
    transform: translateY(-2px);
}
.stat-card .stat-value {
    font-size: 1.6rem;
    font-weight: 700;
    margin-bottom: 0.1rem;
}
.stat-card .stat-label {
    font-size: 0.8rem;
    opacity: 0.85;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* --- Stat card variants --- */
.stat-card.green {
    background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
    box-shadow: 0 4px 15px rgba(17, 153, 142, 0.3);
}
.stat-card.orange {
    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    box-shadow: 0 4px 15px rgba(245, 87, 108, 0.3);
}
.stat-card.blue {
    background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    box-shadow: 0 4px 15px rgba(79, 172, 254, 0.3);
}
.stat-card.gold {
    background: linear-gradient(135deg, #f7971e 0%, #ffd200 100%);
    box-shadow: 0 4px 15px rgba(247, 151, 30, 0.3);
}
.stat-card.dark {
    background: linear-gradient(135deg, #2c3e50 0%, #4ca1af 100%);
    box-shadow: 0 4px 15px rgba(44, 62, 80, 0.3);
}

/* --- Metric tile --- */
.metric-tile {
    background: #f8f9fa;
    border-left: 4px solid #667eea;
    border-radius: 8px;
    padding: 0.8rem 1rem;
    margin-bottom: 0.5rem;
}
.metric-tile .metric-name {
    font-size: 0.75rem;
    color: #6c757d;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 0.2rem;
}
.metric-tile .metric-val {
    font-size: 1.4rem;
    font-weight: 700;
    color: #2c3e50;
}

/* --- Section header --- */
.section-header {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-size: 1.5rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
    padding-top: 0.5rem;
}

/* --- Prediction result card --- */
.prediction-card {
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
    margin: 1rem 0;
    font-size: 1.2rem;
    font-weight: 600;
}
.prediction-card.satisfied {
    background: linear-gradient(135deg, #d4efdf 0%, #a9dfbf 100%);
    border: 2px solid #27ae60;
    color: #1e8449;
}
.prediction-card.dissatisfied {
    background: linear-gradient(135deg, #fadbd8 0%, #f5b7b1 100%);
    border: 2px solid #e74c3c;
    color: #c0392b;
}

/* --- Sidebar --- */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
}
section[data-testid="stSidebar"] .stMarkdown,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] .stRadio label {
    color: #e0e0e0 !important;
}
section[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.1);
}
section[data-testid="stSidebar"] .streamlit-expanderHeader,
section[data-testid="stSidebar"] .streamlit-expanderHeader p,
section[data-testid="stSidebar"] details summary span,
section[data-testid="stSidebar"] details summary {
    color: #ffffff !important;
}

/* --- Tabs --- */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px 8px 0 0;
    padding: 8px 20px;
    font-weight: 600;
}

/* --- Progress bar --- */
.stProgress > div > div > div {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
}

/* --- Badge pill --- */
.badge {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 50px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.3px;
}
.badge-primary {
    background: #d6eaf8;
    color: #2471a3;
}
.badge-warning {
    background: #fdebd0;
    color: #b9770e;
}

/* --- Divider --- */
.styled-divider {
    height: 3px;
    background: linear-gradient(90deg, #667eea 0%, #764ba2 50%, transparent 100%);
    border: none;
    border-radius: 2px;
    margin: 1rem 0 1.5rem 0;
}

/* --- Footer --- */
.footer {
    text-align: center;
    padding: 1.5rem 0;
    margin-top: 2rem;
    border-top: 1px solid rgba(128,128,128,0.2);
    color: #888;
    font-size: 0.85rem;
}
</style>
""", unsafe_allow_html=True)


# -------------------------------
# Helper Functions
# -------------------------------
def stat_card(value, label, color_class=""):
    st.markdown(f"""
    <div class="stat-card {color_class}">
        <div class="stat-value">{value}</div>
        <div class="stat-label">{label}</div>
    </div>
    """, unsafe_allow_html=True)


def metric_tile(name, value):
    st.markdown(f"""
    <div class="metric-tile">
        <div class="metric-name">{name}</div>
        <div class="metric-val">{value}</div>
    </div>
    """, unsafe_allow_html=True)


def section_header(text):
    st.markdown(f'<div class="section-header">{text}</div>', unsafe_allow_html=True)
    st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)


# -------------------------------
# Load JSON
# -------------------------------
@st.cache_data
def load_json(path):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return None


# -------------------------------
# Load All JSON Reports
# -------------------------------
dataset_overview = load_json("reports/preprocessing/dataset_overview.json")
data_quality = load_json("reports/preprocessing/data_quality_report.json")

eda_metadata = load_json("reports/eda/configuration/eda_metadata.json")
eda_results = load_json("reports/eda/configuration/eda_results.json")
feature_config = load_json("reports/eda/configuration/feature_config.json")
model_insights = load_json("reports/eda/configuration/model_insights.json")

model_comparison = load_json("reports/model_training/model_comparison.json")
training_summary = load_json("reports/model_training/training_summary.json")
lr_report = load_json("reports/model_training/logistic_regression.json")
dt_report = load_json("reports/model_training/decision_tree.json")
knn_report = load_json("reports/model_training/knn.json")
nb_report = load_json("reports/model_training/naive_bayes.json")
rf_report = load_json("reports/model_training/random_forest.json")
xgb_report = load_json("reports/model_training/xgboost.json")
data_preparation = load_json("reports/model_training/data_preparation.json")

# -------------------------------
# Load Models & Scaler
# -------------------------------
@st.cache_resource
def load_models():
    mdls = {
        "Logistic Regression": pickle.load(open("models/logistic_regression.pkl", "rb")),
        "Decision Tree": pickle.load(open("models/decision_tree.pkl", "rb")),
        "KNN": pickle.load(open("models/knn.pkl", "rb")),
        "Naive Bayes": pickle.load(open("models/naive_bayes.pkl", "rb")),
        "Random Forest": pickle.load(open("models/random_forest.pkl", "rb")),
        "XGBoost": pickle.load(open("models/xgboost.pkl", "rb"))
    }
    sc = pickle.load(open("models/scaler.pkl", "rb"))
    fc = pickle.load(open("models/feature_columns.pkl", "rb"))
    return mdls, sc, fc

models, scaler, feature_columns = load_models()

@st.cache_data
def load_dataset():
    return pd.read_csv("dataset/test.csv")


# -------------------------------
# Sidebar
# -------------------------------
st.sidebar.markdown("""
<div style="text-align:center; padding: 1rem 0 0.5rem 0;">
    <span style="font-size: 3rem;">✈️</span>
    <h2 style="color: white; margin: 0.3rem 0 0 0; font-weight: 700;">SkyPredict</h2>
    <p style="color: #8899aa; font-size: 0.8rem; margin: 0;">Airline Satisfaction Dashboard</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")

page = st.sidebar.selectbox(
    "☰ Navigate",
    ["🏠 Home", "📊 Dataset Overview", "🏆 Model Performance", "📈 EDA Insights"],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")
st.sidebar.caption("ML Assignment 2 • 2026")


# ===============================
# PAGE: HOME
# ===============================
if page == "🏠 Home":

    # Session state for prediction results overlay & data persistence
    if 'prediction_results' not in st.session_state:
        st.session_state.prediction_results = None
    if 'uploaded_csv' not in st.session_state:
        st.session_state.uploaded_csv = None
    if 'manual_input_data' not in st.session_state:
        st.session_state.manual_input_data = None

    # Hero
    st.markdown("""
    <div class="hero-banner">
        <h1>✈️ Airline Passenger Satisfaction</h1>
        <p>Binary Classification Dashboard — Predict passenger satisfaction using 6 trained ML models.
        Upload a dataset for batch evaluation or enter passenger details manually for instant predictions.</p>
    </div>
    """, unsafe_allow_html=True)

    # Quick stats
    qs1, qs2, qs3, qs4 = st.columns(4)
    with qs1:
        stat_card("6", "Trained Models", "blue")
    with qs2:
        stat_card("98,904", "Training Records", "green")
    with qs3:
        best_name = training_summary['best_model'] if training_summary else "N/A"
        stat_card(best_name, "Best Model", "gold")
    with qs4:
        best_f1 = training_summary['best_f1_score'] if training_summary else "N/A"
        stat_card(str(best_f1), "Best F1 Score", "orange")

    st.markdown("")

    # Download Dataset (compact: selectbox + download side by side)
    section_header("📥 Download Dataset")
    dl1, dl2 = st.columns([2, 1])
    with dl1:
        dataset_choice = st.selectbox(
            "Select dataset",
            ["test.csv (Full — 98,904 rows)", "vis_dataset.csv (Sample — 5,000 rows)"],
            label_visibility="collapsed"
        )
    with dl2:
        if "test.csv" in dataset_choice:
            file_path, file_name = "dataset/test.csv", "test.csv"
        else:
            file_path, file_name = "dataset/vis_dataset.csv", "vis_dataset.csv"
        if os.path.exists(file_path):
            df_download = pd.read_csv(file_path)
            st.download_button(
                label=f"⬇️ Download {file_name}",
                data=df_download.to_csv(index=False).encode('utf-8'),
                file_name=file_name, mime="text/csv",
                use_container_width=True
            )

    st.markdown("")

    # =============================================
    # PREDICTION RESULTS VIEW (overlay — replaces input)
    # =============================================
    if st.session_state.prediction_results is not None:
        res = st.session_state.prediction_results

        section_header("🔮 Prediction Results")

        # --- Top bar: Back button + Model switcher (auto re-predict on change) ---
        back_col, model_col = st.columns([1, 3])
        with back_col:
            if st.button("← Back to Input", use_container_width=True):
                st.session_state.prediction_results = None
                st.rerun()
        with model_col:
            current_model = res.get('model_name', 'Random Forest')
            current_idx = list(models.keys()).index(current_model) if current_model in models else 4
            new_model = st.selectbox(
                "🧠 Switch Model", list(models.keys()),
                index=current_idx, key="results_model_switch"
            )

        # --- Auto re-prediction when model changes ---
        if new_model != current_model:
            scale_models = ["Logistic Regression", "KNN", "Naive Bayes"]

            if res['type'] == 'csv' and st.session_state.uploaded_csv is not None:
                df_re = pd.read_csv(io.BytesIO(st.session_state.uploaded_csv))
                df_re.drop(columns=['id', 'Unnamed: 0'], inplace=True, errors='ignore')
                if 'Arrival Delay in Minutes' in df_re.columns:
                    df_re['Arrival Delay in Minutes'] = df_re['Arrival Delay in Minutes'].fillna(
                        df_re['Arrival Delay in Minutes'].median())
                numeric_cols = df_re.select_dtypes(include=['float64', 'int64']).columns
                df_re[numeric_cols] = df_re[numeric_cols].fillna(df_re[numeric_cols].median())

                has_target = 'satisfaction' in df_re.columns
                if has_target:
                    y_true = df_re['satisfaction'].map({'neutral or dissatisfied': 0, 'satisfied': 1})
                    X_re = df_re.drop(columns=['satisfaction'])
                else:
                    X_re = df_re.copy()
                X_re = pd.get_dummies(X_re, columns=['Gender', 'Customer Type', 'Type of Travel', 'Class'], drop_first=True)
                X_re = X_re.reindex(columns=feature_columns, fill_value=0)

                X_input = scaler.transform(X_re) if new_model in scale_models else X_re.values
                mdl = models[new_model]
                y_pred = mdl.predict(X_input)
                y_prob = mdl.predict_proba(X_input)[:, 1]

                new_result = {'type': 'csv', 'model_name': new_model, 'has_target': has_target}
                if has_target:
                    new_result['metrics'] = {
                        'Accuracy': round(accuracy_score(y_true, y_pred), 4),
                        'Precision': round(precision_score(y_true, y_pred), 4),
                        'Recall': round(recall_score(y_true, y_pred), 4),
                        'F1 Score': round(f1_score(y_true, y_pred), 4),
                        'AUC Score': round(roc_auc_score(y_true, y_prob), 4),
                        'MCC Score': round(matthews_corrcoef(y_true, y_pred), 4),
                    }
                    new_result['confusion_matrix'] = confusion_matrix(y_true, y_pred).tolist()
                results_df = df_re.copy()
                results_df["Predicted"] = ["Satisfied" if p == 1 else "Dissatisfied" for p in y_pred]
                results_df["Confidence"] = [round(float(p), 4) for p in y_prob]
                new_result['predictions'] = results_df.head(50).to_dict('list')
                new_result['total_rows'] = len(df_re)
                st.session_state.prediction_results = new_result
                st.rerun()

            elif res['type'] == 'manual' and st.session_state.manual_input_data is not None:
                input_df = pd.DataFrame([st.session_state.manual_input_data])
                input_df = input_df.reindex(columns=feature_columns, fill_value=0)
                X_input = scaler.transform(input_df) if new_model in scale_models else input_df.values
                mdl = models[new_model]
                pred_val = int(mdl.predict(X_input)[0])
                prob_val = mdl.predict_proba(X_input)[0]
                st.session_state.prediction_results = {
                    'type': 'manual',
                    'pred': pred_val,
                    'satisfied_prob': float(prob_val[1]) * 100,
                    'dissatisfied_prob': float(prob_val[0]) * 100,
                    'model_name': new_model,
                }
                st.rerun()

        st.markdown("")

        # --- Show results ---
        if res['type'] == 'manual':
            pred = res['pred']
            satisfied_prob = res['satisfied_prob']
            dissatisfied_prob = res['dissatisfied_prob']

            if pred == 1:
                st.markdown(f"""
                <div class="prediction-card satisfied">
                    ✅ Satisfied — {satisfied_prob:.1f}% confidence
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="prediction-card dissatisfied">
                    ❌ Dissatisfied — {dissatisfied_prob:.1f}% confidence
                </div>
                """, unsafe_allow_html=True)

            st.markdown("")
            pb1, pb2 = st.columns(2)
            with pb1:
                stat_card(f"{satisfied_prob:.1f}%", "Satisfied Probability", "green")
            with pb2:
                stat_card(f"{dissatisfied_prob:.1f}%", "Dissatisfied Probability", "orange")

            st.progress(float(satisfied_prob / 100))

            prob_df = pd.DataFrame({
                "Class": ["Dissatisfied", "Satisfied"],
                "Probability (%)": [round(dissatisfied_prob, 2), round(satisfied_prob, 2)]
            })
            st.dataframe(prob_df, use_container_width=True, hide_index=True)
            st.caption(f"Model: {res['model_name']}")

        elif res['type'] == 'csv':
            st.caption(f"Model: {res['model_name']}  •  {res.get('total_rows', 0):,} rows")
            if res.get('has_target'):
                metrics_col, cm_col = st.columns([1, 1])
                with metrics_col:
                    st.markdown("#### 📊 Evaluation Metrics")
                    for m_name, m_val in res['metrics'].items():
                        metric_tile(m_name, m_val)
                with cm_col:
                    st.markdown("#### 📉 Confusion Matrix")
                    cm = np.array(res['confusion_matrix'])
                    fig, ax = plt.subplots(figsize=(5, 4.5))
                    sns.heatmap(cm, annot=True, fmt="d", cmap="PuBuGn", ax=ax,
                                annot_kws={"size": 14, "weight": "bold"},
                                xticklabels=["Dissatisfied", "Satisfied"],
                                yticklabels=["Dissatisfied", "Satisfied"],
                                linewidths=0.5, linecolor="white")
                    ax.set_xlabel("Predicted", fontsize=11, fontweight="bold")
                    ax.set_ylabel("Actual", fontsize=11, fontweight="bold")
                    ax.set_title(res['model_name'], fontsize=12, fontweight="bold", pad=10)
                    plt.tight_layout()
                    st.pyplot(fig, use_container_width=True)

            st.markdown("")
            with st.expander("🔮  Detailed Predictions", expanded=True):
                pred_data = res.get('predictions', {})
                if pred_data:
                    results_df = pd.DataFrame(pred_data)
                    st.dataframe(
                        results_df.head(50).style.applymap(
                            lambda v: "color: #27ae60; font-weight: bold" if v == "Satisfied"
                            else ("color: #e74c3c; font-weight: bold" if v == "Dissatisfied" else ""),
                            subset=["Predicted"]
                        ),
                        use_container_width=True, hide_index=True
                    )
                    st.caption(f"Showing top 50 of {res.get('total_rows', '?'):,} rows")

        st.stop()

    # =============================================
    # INPUT VIEW (default — pick a prediction method)
    # =============================================
    section_header("🤖 Model Prediction")

    # ──────────────────────────────
    # SECTION: Upload CSV
    # ──────────────────────────────
    with st.expander("📁  Upload CSV — Batch Prediction", expanded=False):
        uploaded_file = st.file_uploader(
            "Upload a CSV file (with satisfaction column for evaluation, or without for prediction only)",
            type=["csv"]
        )

        # Store uploaded CSV in session state so it persists across reruns
        if uploaded_file is not None:
            st.session_state.uploaded_csv = uploaded_file.read()
            uploaded_file.seek(0)  # reset for immediate use

        if st.session_state.uploaded_csv is not None:
            df = pd.read_csv(io.BytesIO(st.session_state.uploaded_csv))

            with st.expander("👀  Uploaded Dataset Preview", expanded=False):
                st.dataframe(df.head(10), use_container_width=True)
                st.caption(f"Showing first 10 of {len(df):,} rows  •  {df.shape[1]} columns")

            if st.button("🗑️ Clear Dataset", key="clear_csv"):
                st.session_state.uploaded_csv = None
                st.rerun()

            # Preprocess
            df.drop(columns=['id', 'Unnamed: 0'], inplace=True, errors='ignore')
            if 'Arrival Delay in Minutes' in df.columns:
                df['Arrival Delay in Minutes'] = df['Arrival Delay in Minutes'].fillna(
                    df['Arrival Delay in Minutes'].median()
                )
            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

            has_target = 'satisfaction' in df.columns
            if has_target:
                y_true = df['satisfaction'].map({'neutral or dissatisfied': 0, 'satisfied': 1})
                X = df.drop(columns=['satisfaction'])
            else:
                X = df.copy()

            X = pd.get_dummies(X, columns=['Gender', 'Customer Type', 'Type of Travel', 'Class'], drop_first=True)
            X = X.reindex(columns=feature_columns, fill_value=0)

            model_choice_upload = st.selectbox(
                "🧠  Select Model", list(models.keys()),
                index=list(models.keys()).index("Random Forest"), key="model_upload"
            )

            if st.button("🔮 Run Prediction", key="csv_predict_btn", type="primary", use_container_width=True):
                scale_models = ["Logistic Regression", "KNN", "Naive Bayes"]
                X_input = scaler.transform(X) if model_choice_upload in scale_models else X.values

                model = models[model_choice_upload]
                y_pred = model.predict(X_input)
                y_prob = model.predict_proba(X_input)[:, 1]

                result = {
                    'type': 'csv',
                    'model_name': model_choice_upload,
                    'has_target': has_target,
                }

                if has_target:
                    result['metrics'] = {
                        'Accuracy': round(accuracy_score(y_true, y_pred), 4),
                        'Precision': round(precision_score(y_true, y_pred), 4),
                        'Recall': round(recall_score(y_true, y_pred), 4),
                        'F1 Score': round(f1_score(y_true, y_pred), 4),
                        'AUC Score': round(roc_auc_score(y_true, y_prob), 4),
                        'MCC Score': round(matthews_corrcoef(y_true, y_pred), 4),
                    }
                    result['confusion_matrix'] = confusion_matrix(y_true, y_pred).tolist()

                results_df = df.copy()
                results_df["Predicted"] = ["Satisfied" if p == 1 else "Dissatisfied" for p in y_pred]
                results_df["Confidence"] = [round(float(p), 4) for p in y_prob]
                result['predictions'] = results_df.head(50).to_dict('list')
                result['total_rows'] = len(df)

                st.session_state.prediction_results = result
                st.rerun()

    # ──────────────────────────────
    # SECTION: Manual Input
    # ──────────────────────────────
    with st.expander("✍️  Manual Input — Single Prediction", expanded=False):
        st.markdown("#### ✏️ Enter Passenger Details")
        st.caption("Fill in the fields below and hit Predict for an instant satisfaction prediction.")

        mc1, mc2, mc3 = st.columns(3)

        with mc1:
            st.markdown("**🧑 Passenger Info**")
            gender = st.selectbox("Gender", ["Female", "Male"])
            customer_type = st.selectbox("Customer Type", ["Loyal Customer", "disloyal Customer"])
            age = st.slider("Age", 7, 85, 40)
            travel_type = st.selectbox("Type of Travel", ["Business travel", "Personal Travel"])
            flight_class = st.selectbox("Class", ["Business", "Eco", "Eco Plus"])
            flight_distance = st.number_input("Flight Distance (miles)", 30, 5000, 1000)

        with mc2:
            st.markdown("**⭐ Service Ratings (1-5)**")
            wifi = st.slider("Inflight wifi service", 1, 5, 3)
            time_conv = st.slider("Departure/Arrival time convenient", 1, 5, 3)
            online_book = st.slider("Ease of Online booking", 1, 5, 3)
            gate_loc = st.slider("Gate location", 1, 5, 3)
            food = st.slider("Food and drink", 1, 5, 3)
            online_board = st.slider("Online boarding", 1, 5, 3)

        with mc3:
            st.markdown("**🛫 Flight Experience**")
            seat = st.slider("Seat comfort", 1, 5, 3)
            entertainment = st.slider("Inflight entertainment", 1, 5, 3)
            onboard = st.slider("On-board service", 1, 5, 3)
            legroom = st.slider("Leg room service", 1, 5, 3)
            baggage = st.slider("Baggage handling", 1, 5, 3)
            checkin = st.slider("Checkin service", 1, 5, 3)
            inflight_svc = st.slider("Inflight service", 1, 5, 3)
            cleanliness = st.slider("Cleanliness", 1, 5, 3)
            dep_delay = st.number_input("Departure Delay (min)", 0, 1600, 0)
            arr_delay = st.number_input("Arrival Delay (min)", 0, 1600, 0)

        st.markdown("")
        sel1, sel2 = st.columns([1, 1])
        with sel1:
            model_choice_manual = st.selectbox(
                "🧠 Select Model", list(models.keys()),
                index=list(models.keys()).index("Random Forest"), key="model_manual"
            )
        with sel2:
            st.markdown("")
            st.markdown("")
            predict_btn = st.button("🔮 Predict Satisfaction", type="primary", use_container_width=True)

        if predict_btn:
            input_data = {
                "Age": age, "Flight Distance": flight_distance,
                "Inflight wifi service": wifi, "Departure/Arrival time convenient": time_conv,
                "Ease of Online booking": online_book, "Gate location": gate_loc,
                "Food and drink": food, "Online boarding": online_board,
                "Seat comfort": seat, "Inflight entertainment": entertainment,
                "On-board service": onboard, "Leg room service": legroom,
                "Baggage handling": baggage, "Checkin service": checkin,
                "Inflight service": inflight_svc, "Cleanliness": cleanliness,
                "Departure Delay in Minutes": dep_delay, "Arrival Delay in Minutes": arr_delay,
                "Gender_Male": 1 if gender == "Male" else 0,
                "Customer Type_disloyal Customer": 1 if customer_type == "disloyal Customer" else 0,
                "Type of Travel_Personal Travel": 1 if travel_type == "Personal Travel" else 0,
                "Class_Eco": 1 if flight_class == "Eco" else 0,
                "Class_Eco Plus": 1 if flight_class == "Eco Plus" else 0,
            }

            # Store input data for model switching
            st.session_state.manual_input_data = input_data

            input_df = pd.DataFrame([input_data])
            input_df = input_df.reindex(columns=feature_columns, fill_value=0)

            scale_models = ["Logistic Regression", "KNN", "Naive Bayes"]
            X_input = scaler.transform(input_df) if model_choice_manual in scale_models else input_df.values

            model = models[model_choice_manual]
            pred = int(model.predict(X_input)[0])
            prob = model.predict_proba(X_input)[0]
            satisfied_prob = float(prob[1]) * 100
            dissatisfied_prob = float(prob[0]) * 100

            st.session_state.prediction_results = {
                'type': 'manual',
                'pred': pred,
                'satisfied_prob': satisfied_prob,
                'dissatisfied_prob': dissatisfied_prob,
                'model_name': model_choice_manual,
            }
            st.rerun()


# ===============================
# PAGE: DATASET OVERVIEW
# ===============================
elif page == "📊 Dataset Overview":

    st.markdown("""
    <div class="hero-banner" style="background: linear-gradient(135deg, #0f3460 0%, #533483 50%, #e94560 100%);">
        <h1>📊 Dataset Overview</h1>
        <p>Explore the airline satisfaction dataset — structure, quality, and descriptive statistics at a glance.</p>
    </div>
    """, unsafe_allow_html=True)

    if dataset_overview:
        c1, c2, c3 = st.columns(3)
        with c1:
            stat_card(f"{dataset_overview['total_records']:,}", "Total Records", "blue")
        with c2:
            stat_card(str(dataset_overview['total_features']), "Total Features", "green")
        with c3:
            stat_card(dataset_overview['source_file'], "Source File", "dark")

    st.markdown("")

    with st.expander("📋  Column Information", expanded=False):
        if dataset_overview:
            col_df = pd.DataFrame({
                "Column": dataset_overview["column_names"],
                "Data Type": [dataset_overview["dtypes"][c] for c in dataset_overview["column_names"]]
            })
            col_df["Category"] = col_df["Data Type"].apply(
                lambda x: "📝 Categorical" if x == "object" else "🔢 Numeric"
            )

            num_count = len(col_df[col_df["Category"].str.contains("Numeric")])
            cat_count = len(col_df[col_df["Category"].str.contains("Categorical")])
            tc1, tc2 = st.columns(2)
            with tc1:
                st.markdown(f'<span class="badge badge-primary">🔢 {num_count} Numeric</span>', unsafe_allow_html=True)
            with tc2:
                st.markdown(f'<span class="badge badge-warning">📝 {cat_count} Categorical</span>', unsafe_allow_html=True)

            st.dataframe(col_df, use_container_width=True, hide_index=True)
        else:
            st.warning("Column information not available.")

    with st.expander("🔍  Data Quality Report", expanded=False):
        if data_quality:
            q1, q2, q3 = st.columns(3)
            with q1:
                stat_card(f"{data_quality['data_completeness_percent']}%", "Data Completeness", "green")
            with q2:
                stat_card(str(data_quality['duplicate_rows']), "Duplicate Rows", "blue")
            with q3:
                stat_card(str(len(data_quality['columns_with_missing'])), "Columns with Missing", "orange")

            if data_quality["missing_values"]:
                st.markdown("")
                st.markdown("**Missing Values Breakdown:**")
                miss_df = pd.DataFrame({
                    "Column": list(data_quality["missing_values"].keys()),
                    "Missing Count": list(data_quality["missing_values"].values()),
                    "Missing %": list(data_quality["missing_percentages"].values())
                })
                st.dataframe(miss_df, use_container_width=True, hide_index=True)
            else:
                st.success("✅ No missing values in the dataset!")
        else:
            st.warning("Data quality report not available.")

    with st.expander("👀  Dataset Preview", expanded=False):
        try:
            df_preview = load_dataset()
            st.dataframe(df_preview.head(20), use_container_width=True)
            st.markdown("")
            st.markdown("**📊 Descriptive Statistics**")
            st.dataframe(df_preview.describe().round(2), use_container_width=True)
        except Exception:
            st.warning("Dataset file not found.")


# ===============================
# PAGE: MODEL PERFORMANCE
# ===============================
elif page == "🏆 Model Performance":

    st.markdown("""
    <div class="hero-banner" style="background: linear-gradient(135deg, #1a2a6c 0%, #b21f1f 50%, #fdbb2d 100%);">
        <h1>🏆 Model Performance Comparison</h1>
        <p>Compare all 6 trained classifiers side by side — accuracy, F1, AUC, and more.</p>
    </div>
    """, unsafe_allow_html=True)

    if training_summary:
        ts1, ts2, ts3 = st.columns(3)
        with ts1:
            stat_card(training_summary['training_date'], "Training Date", "dark")
        with ts2:
            stat_card(str(training_summary['total_models']), "Models Trained", "blue")
        with ts3:
            stat_card(training_summary['best_model'],
                      f"Best Model (F1: {training_summary['best_f1_score']})", "gold")

    if model_comparison:
        results = model_comparison["results"]

        with st.expander("📊  All Models — Metrics Comparison", expanded=False):
            comp_df = pd.DataFrame(results).T.sort_values("F1 Score", ascending=False)
            st.dataframe(
                comp_df.style
                .highlight_max(axis=0, color="#90EE90")
                .highlight_min(axis=0, color="#FFCDD2")
                .format("{:.4f}"),
                use_container_width=True
            )

        with st.expander("📈  Visual Comparison", expanded=False):
            metrics_list = ["Accuracy", "AUC Score", "Precision", "Recall", "F1 Score", "MCC Score"]
            selected_metric = st.selectbox("Select metric to compare", metrics_list, index=4)

            chart_data = {m: vals[selected_metric] for m, vals in results.items()}
            sorted_models = sorted(chart_data.items(), key=lambda x: x[1], reverse=True)

            fig, ax = plt.subplots(figsize=(10, 5))
            palette = ["#667eea", "#764ba2", "#f093fb", "#f5576c", "#4facfe", "#38ef7d"]
            bars = ax.barh(
                [m[0] for m in sorted_models][::-1],
                [m[1] for m in sorted_models][::-1],
                color=palette[:len(sorted_models)], height=0.55,
                edgecolor="white", linewidth=0.5
            )
            for bar, (_, val) in zip(bars, sorted(sorted_models, key=lambda x: x[1])):
                ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                        f"{val:.4f}", va="center", fontsize=11, fontweight="bold", color="#333")
            ax.set_xlabel(selected_metric, fontsize=12, fontweight="bold")
            ax.set_title(f"Model Comparison — {selected_metric}", fontsize=14, fontweight="bold", pad=12)
            ax.set_xlim(0, 1.08)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)

    st.markdown("")

    model_reports = {
        "Logistic Regression": lr_report, "Decision Tree": dt_report,
        "KNN": knn_report, "Naive Bayes": nb_report,
        "Random Forest": rf_report, "XGBoost": xgb_report,
    }

    for model_name, report in model_reports.items():
        if report:
            with st.expander(f"📋  {model_name}", expanded=False):
                i1, i2, i3 = st.columns(3)
                with i1:
                    st.markdown(f"**Trained**: {report['timestamp']}")
                with i2:
                    st.markdown(f"**Scaled Data**: `{'Yes' if report['uses_scaled_data'] else 'No'}`")
                with i3:
                    if report.get("hyperparameters"):
                        st.markdown(f"**Params**: `{report['hyperparameters']}`")

                metrics = report["metrics"]
                mr1, mr2, mr3 = st.columns(3)
                with mr1:
                    metric_tile("Accuracy", f"{metrics['Accuracy']:.4f}")
                with mr2:
                    metric_tile("AUC Score", f"{metrics['AUC Score']:.4f}")
                with mr3:
                    metric_tile("Precision", f"{metrics['Precision']:.4f}")
                mr4, mr5, mr6 = st.columns(3)
                with mr4:
                    metric_tile("Recall", f"{metrics['Recall']:.4f}")
                with mr5:
                    metric_tile("F1 Score", f"{metrics['F1 Score']:.4f}")
                with mr6:
                    metric_tile("MCC Score", f"{metrics['MCC Score']:.4f}")


# ===============================
# PAGE: EDA INSIGHTS
# ===============================
elif page == "📈 EDA Insights":

    st.markdown("""
    <div class="hero-banner" style="background: linear-gradient(135deg, #134e5e 0%, #71b280 100%);">
        <h1>📈 Exploratory Data Analysis</h1>
        <p>Key insights from the data — feature importance, correlations, and visual analysis of satisfaction patterns.</p>
    </div>
    """, unsafe_allow_html=True)

    if eda_metadata:
        e1, e2 = st.columns(2)
        with e1:
            stat_card(eda_metadata['analysis_date'], "Analysis Date", "dark")
        with e2:
            stat_card(str(eda_metadata['total_files_generated']), "Visualizations Generated", "green")

        st.markdown("")
        insights = eda_metadata.get("key_insights", {})
        i1, i2 = st.columns(2)
        with i1:
            st.info(f"**🔍 Data Quality**: {insights.get('data_quality', 'N/A')}")
        with i2:
            st.success(f"**⭐ Strongest Predictor**: {insights.get('strongest_predictor', 'N/A')}")

    with st.expander("🏆  Top Predictive Features", expanded=False):
        if model_insights:
            top_features = model_insights.get("feature_selection", {}).get("top_predictive_features", {})
            if top_features:
                feat_df = pd.DataFrame({
                    "Feature": list(top_features.keys()),
                    "Correlation": list(top_features.values())
                }).sort_values("Correlation", ascending=False)

                fig, ax = plt.subplots(figsize=(10, 5))
                cmap = plt.cm.RdYlGn
                colors = [cmap(v) for v in np.linspace(0.3, 0.9, len(feat_df))]
                bars = ax.barh(feat_df["Feature"][::-1], feat_df["Correlation"][::-1],
                               color=colors, height=0.6, edgecolor="white", linewidth=0.5)
                for bar, val in zip(bars, feat_df["Correlation"][::-1]):
                    ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                            f"{val:.3f}", va="center", fontsize=10, fontweight="bold", color="#333")
                ax.set_xlabel("Correlation with Satisfaction", fontsize=11, fontweight="bold")
                ax.set_title("Feature Importance Ranking", fontsize=13, fontweight="bold", pad=10)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
        else:
            st.warning("Model insights not available.")

    with st.expander("🖼️  EDA Visualizations", expanded=False):
        eda_images = {
            "Satisfaction Relationships": "reports/eda/static_images/viz_01_satisfaction_relationships.png",
            "Correlation Heatmap": "reports/eda/static_images/viz_02_correlation_heatmap.png",
            "Service Quality Analysis": "reports/eda/static_images/viz_03_service_quality_analysis.png",
            "Statistical Summary & Outliers": "reports/eda/static_images/viz_04_statistical_summary_outliers.png",
            "Numerical vs Satisfaction": "reports/eda/static_images/viz_05_numerical_vs_satisfaction.png",
            "Feature Importance Summary": "reports/eda/static_images/viz_06_feature_importance_summary.png",
        }

        selected_viz = st.selectbox(
            "Select a visualization", list(eda_images.keys()), key="eda_viz_select"
        )
        viz_path = eda_images[selected_viz]
        if os.path.exists(viz_path):
            st.image(viz_path, caption=selected_viz, use_container_width=True)
        else:
            st.warning(f"Image not found: {viz_path}")


