import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import re
import io
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# --- STREAMLIT PAGE CONFIG ---
st.set_page_config(page_title="IE Energy Theft Detection Dashboard (ML)", layout="wide")

# --- CUSTOM CSS: GREEN & GOLD THEME ---
st.markdown("""
    <style>
    /* Main app background */
    .stApp {
        background-color: #002b16; /* Deep Emerald Green */
        color: #FFFFFF;
    }
    
    /* Global Text and Headers */
    h1, h2, h3, h4, h5, h6, p, span {
        font-family: 'Inter', sans-serif;
    }
    
    h1, h2, h3 {
        color: #D4AF37 !important; /* Metallic Gold */
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Sidebar Background */
    [data-testid="stSidebar"] {
        background-color: #001a0d;
        border-right: 2px solid #D4AF37;
    }

    /* Primary Buttons */
    .stButton>button {
        background-color: #D4AF37 !important;
        color: #002b16 !important;
        font-weight: bold;
        border-radius: 5px;
        border: 1px solid #B8860B;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background-color: #FFD700 !important;
        box-shadow: 0px 0px 15px rgba(212, 175, 55, 0.5);
        transform: scale(1.02);
    }

    /* Input Widgets */
    .stSelectbox, .stSlider, .stTextInput {
        color: #D4AF37;
    }
    
    /* Metric Cards and Dataframes */
    [data-testid="stMetricValue"] {
        color: #D4AF37;
    }
    
    .stDataFrame {
        border: 1px solid #D4AF37;
    }

    /* Alerts and Info Boxes */
    .stAlert {
        background-color: #003d1f;
        color: #f1f1f1;
        border-left: 5px solid #D4AF37;
    }

    /* Links */
    a {
        color: #D4AF37 !important;
        text-decoration: none;
    }
    </style>
    """, unsafe_allow_html=True)

# --- UTILITY FUNCTIONS ---
def preserve_exact_string(value):
    if pd.isna(value) or value is None:
        return ""
    return str(value)

def normalize_name(name):
    if not isinstance(name, str):
        return ""
    name = re.sub(r'\s+', ' ', name.strip().upper())
    name = re.sub(r'[^\w\s-]', '', name)
    name = re.sub(r'-+', '-', name)
    return name

def get_short_name(name, is_dt=False):
    if isinstance(name, str) and name and "-" in name:
        parts = name.split("-")
        return parts[-1].strip()
    return name if isinstance(name, str) else ""

def add_feeder_column_safe(df, name_of_dt_col="NAME_OF_DT"):
    if name_of_dt_col not in df.columns:
        st.error(f"Column '{name_of_dt_col}' missing. Cannot derive 'Feeder'.")
        df["Feeder"] = ""
        return df
    df = df.copy()
    df["Feeder"] = df[name_of_dt_col].apply(
        lambda x: "-".join(x.split("-")[:-1]) if isinstance(x, str) and "-" in x and len(x.split("-")) >= 3 else x
    )
    df["Feeder"] = df["Feeder"].apply(normalize_name)
    return df

# --- FEATURE CALCULATION FUNCTIONS ---
def calculate_pattern_deviation(df, id_col, value_cols):
    results = []
    valid_cols = [c for c in value_cols if c in df.columns]
    if not valid_cols:
        return pd.DataFrame({"id": [], "pattern_deviation_score": []})
    for id_val, group in df.groupby(id_col):
        values = group[valid_cols].iloc[0].values.astype(float)
        nonzero = values[values > 0]
        if len(nonzero) == 0:
            score = 1.0
        else:
            max_nonzero = nonzero.max()
            below = np.sum(values < 0.6 * max_nonzero)
            score = below / len(valid_cols)
        results.append({"id": id_val, "pattern_deviation_score": min(score, 1.0)})
    return pd.DataFrame(results).rename(columns={"id": id_col})

def calculate_zero_counter(df, id_col, value_cols):
    results = []
    valid_cols = [c for c in value_cols if c in df.columns]
    if not valid_cols:
        return pd.DataFrame({"id": [], "zero_counter_score": []})
    for id_val, group in df.groupby(id_col):
        values = group[valid_cols].iloc[0].values.astype(float)
        zeros = np.sum(values == 0)
        score = zeros / len(valid_cols) if len(valid_cols) > 0 else 0.0
        results.append({"id": id_val, "zero_counter_score": min(score, 1.0)})
    return pd.DataFrame(results).rename(columns={"id": id_col})

def calculate_dt_relative_usage(customer_monthly):
    cust_sum = customer_monthly.groupby(["ACCOUNT_NUMBER", "NAME_OF_DT"], as_index=False)["billed_kwh"].sum()
    dt_avg = cust_sum[cust_sum["billed_kwh"] > 0].groupby("NAME_OF_DT", as_index=False)["billed_kwh"].mean().rename(columns={"billed_kwh": "dt_avg_kwh"})
    cust_sum = cust_sum.merge(dt_avg, on="NAME_OF_DT", how="left")
    
    def _score(row):
        if pd.isna(row["dt_avg_kwh"]) or row["dt_avg_kwh"] == 0:
            return 0.5 if row["billed_kwh"] == 0 else 0.1
        if row["billed_kwh"] < 0.3 * row["dt_avg_kwh"]:
            return 0.9
        if row["billed_kwh"] > 0.7 * row["dt_avg_kwh"]:
            return 0.1
        ratio = row["billed_kwh"] / row["dt_avg_kwh"]
        return 0.1 + (0.9 - 0.1) * (0.7 - ratio) / (0.7 - 0.3)
    
    cust_sum["dt_relative_usage_score"] = cust_sum.apply(_score, axis=1)
    return cust_sum[["ACCOUNT_NUMBER", "dt_relative_usage_score"]]

def generate_escalations_report(ppm_df, ppd_df, escalations_df, customer_scores_df, months_list, final_score_col_name):
    escalations = escalations_df.copy()
    acct_col = None
    for col in escalations.columns:
        if col.strip().lower() in ["account no", "account_no", "accountnumber", "account number"]:
            acct_col = col
            break
    if acct_col is None:
        return pd.DataFrame()
    accounts = escalations[acct_col].astype(str).str.strip().unique().tolist()
    customers = pd.concat([ppm_df, ppd_df], ignore_index=True, sort=False)
    
    if "ACCOUNT_NUMBER" not in customers.columns:
        for c in customers.columns:
            if c.strip().lower() in ["account no", "account_no", "accountnumber", "account number", "acct"]:
                customers = customers.rename(columns={c: "ACCOUNT_NUMBER"})
                break
    
    reports = []
    for acc in accounts:
        matched = customers[customers["ACCOUNT_NUMBER"].astype(str).str.strip() == str(acc).strip()]
        if matched.empty:
            reports.append({
                "Account No": acc, "Found": "No", "CUSTOMER_NAME": "Not Found"
            })
        else:
            for _, r in matched.iterrows():
                row = {
                    "Account No": acc, "Found": "Yes", "Billing_Type": r.get("Billing_Type", ""),
                    "ACCOUNT_NUMBER": r.get("ACCOUNT_NUMBER", ""), "CUSTOMER_NAME": r.get("CUSTOMER_NAME", ""),
                    "Feeder": r.get("Feeder", ""), "NAME_OF_DT": r.get("NAME_OF_DT", ""),
                    "METER_NUMBER": r.get("METER_NUMBER", "")
                }
                for m in months_list:
                    colname = f"{m} (kWh)"
                    row[m] = r.get(colname, np.nan)
                
                tp_row = customer_scores_df[customer_scores_df["ACCOUNT_NUMBER"].astype(str) == str(acc)]
                row[final_score_col_name] = float(tp_row[final_score_col_name].mean()) if not tp_row.empty else np.nan
                reports.append(row)
    return pd.DataFrame(reports)

# --- ML FUNCTION ---
@st.cache_data
def run_isolation_forest(df, features, contamination_rate=0.01):
    X = df[features].copy().replace([np.inf, -np.inf], np.nan).fillna(X.mean())
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = IsolationForest(n_estimators=100, contamination=contamination_rate, random_state=42, n_jobs=-1)
    model.fit(X_scaled)
    
    anomaly_score = model.decision_function(X_scaled)
    normalized_score = (anomaly_score - anomaly_score.min()) / (anomaly_score.max() - anomaly_score.min())
    df['theft_probability_ml'] = 1 - normalized_score
    return df

# --- UI HEADER ---
col_logo, col_title = st.columns([1, 4])
with col_logo:
    try:
        st.image("Sniffit Logo.png", width=160)
    except Exception:
        st.markdown("### [SniffIt Logo]") # Placeholder if file not found

with col_title:
    st.title("SniffIt🐘")
    st.markdown("#### *AI-Powered Energy Theft Detection*")

# --- MAIN APP LOGIC ---
uploaded_file = st.file_uploader("Upload Network Data (Excel)", type=["xlsx"])
if uploaded_file is None:
    st.info("Waiting for Excel file upload to begin analysis.")
    st.stop()

# Load Data
try:
    sheets = pd.read_excel(uploaded_file, sheet_name=None)
    def _get_sheet(name):
        for k in sheets.keys():
            if k.strip().lower() == name.lower(): return sheets[k]
        return None

    feeder_df = _get_sheet("Feeder Data")
    dt_df = _get_sheet("Transformer Data")
    ppm_df = _get_sheet("Customer Data_PPM")
    ppd_df = _get_sheet("Customer Data_PPD")
    band_df = _get_sheet("Feeder Band")
    tariff_df = _get_sheet("Customer Tariffs")
    escalations_df = _get_sheet("Escalations")

    if any(x is None for x in [feeder_df, dt_df, ppm_df, ppd_df]):
        st.error("Essential sheets (Feeder, Transformer, or Customer data) are missing.")
        st.stop()
except Exception as e:
    st.error(f"Error loading file: {e}")
    st.stop()

# Data Preprocessing (Normalization & Merging)
months = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN"]
for df, unit in [(feeder_df, 1000), (ppm_df, 1), (ppd_df, 1)]:
    for m in months:
        col = f"{m} (kWh)"
        df[col] = pd.to_numeric(df[m], errors="coerce").fillna(0) * unit if m in df.columns else 0

ppm_df["Billing_Type"], ppd_df["Billing_Type"] = "PPM", "PPD"
customer_df = pd.concat([ppm_df, ppd_df], ignore_index=True, sort=False)
customer_df = add_feeder_column_safe(customer_df, "NAME_OF_DT")

dt_df["NAME_OF_DT"] = dt_df.get("New Unique DT Nomenclature", dt_df.get("NAME_OF_DT", ""))
dt_df["DT_Short_Name"] = dt_df["NAME_OF_DT"].apply(get_short_name)
customer_df["DT_Short_Name"] = customer_df["NAME_OF_DT"].apply(get_short_name)

# --- SIDEBAR & FILTERS ---
st.sidebar.header("Configuration")
selected_feeder = st.sidebar.selectbox("Select Feeder", sorted(feeder_df["Feeder"].unique()) if "Feeder" in feeder_df.columns else [])
model_choice = st.sidebar.radio("Analysis Model", ('Weighted Rule-Based', 'Isolation Forest ML'))

# Feature Weights / Contamination
if model_choice == 'Weighted Rule-Based':
    w_pattern = st.sidebar.slider("Consumption Pattern Weight", 0.0, 1.0, 0.7)
    w_relative = st.sidebar.slider("DT Relative Usage Weight", 0.0, 1.0, 0.7)
    w_zero = st.sidebar.slider("Zero Frequency Weight", 0.0, 1.0, 0.7)
    contamination_rate = 0.01
else:
    contamination_rate = st.sidebar.slider("ML Contamination (Theft %)", 0.005, 0.10, 0.02)
    w_pattern = w_relative = w_zero = 1.0

# --- CORE CALCULATIONS ---
pattern_df = calculate_pattern_deviation(customer_df, "ACCOUNT_NUMBER", [f"{m} (kWh)" for m in months])
zero_df = calculate_zero_counter(customer_df, "ACCOUNT_NUMBER", [f"{m} (kWh)" for m in months])

# Aggregating Customer Features
required_id_vars = ["NAME_OF_DT", "DT_Short_Name", "ACCOUNT_NUMBER", "METER_NUMBER", "CUSTOMER_NAME", "Billing_Type", "Feeder"]
customer_monthly = customer_df.melt(id_vars=required_id_vars, value_vars=[f"{m} (kWh)" for m in months], var_name="month", value_name="billed_kwh")
relative_usage_df = calculate_dt_relative_usage(customer_monthly)

customer_features = pattern_df.merge(zero_df, on="ACCOUNT_NUMBER").merge(relative_usage_df, on="ACCOUNT_NUMBER")
customer_features.columns = ["ACCOUNT_NUMBER", "F_Pattern", "F_Zero", "F_Relative"]

# Apply ML or Weighted Score
if model_choice == 'Isolation Forest ML':
    customer_features = run_isolation_forest(customer_features, ["F_Pattern", "F_Zero", "F_Relative"], contamination_rate)
    score_col = "theft_probability_ml"
    display_score_name = "ML Probability (Avg)"
else:
    customer_features["theft_probability_weighted"] = (
        (w_pattern * customer_features["F_Pattern"]) + 
        (w_relative * customer_features["F_Relative"]) + 
        (w_zero * customer_features["F_Zero"])
    ) / (w_pattern + w_relative + w_zero)
    score_col = "theft_probability_weighted"
    display_score_name = "Weighted Probability (Avg)"

# Final customer merge for display
final_display_df = customer_df[required_id_vars + ["DT_Short_Name"]].merge(customer_features, on="ACCOUNT_NUMBER")

# --- VISUALIZATION ---
st.subheader(f"High Risk Targets: {selected_feeder}")
feeder_mask = final_display_df["Feeder"] == normalize_name(selected_feeder)
feeder_data = final_display_df[feeder_mask].sort_values(score_col, ascending=False)

if feeder_data.empty:
    st.warning("No data found for the selected feeder.")
else:
    col_a, col_b = st.columns([2, 1])
    with col_a:
        st.dataframe(feeder_data[[
            "ACCOUNT_NUMBER", "CUSTOMER_NAME", "DT_Short_Name", score_col, "F_Pattern", "F_Zero"
        ]].style.background_gradient(subset=[score_col], cmap="YlOrRd"), use_container_width=True)
    
    with col_b:
        # Mini chart of risk distribution
        fig, ax = plt.subplots(facecolor="#002b16")
        sns.histplot(feeder_data[score_col], bins=20, color="#D4AF37", ax=ax)
        ax.set_title("Feeder Risk Distribution", color="#D4AF37")
        ax.set_facecolor("#002b16")
        ax.tick_params(colors='white')
        st.pyplot(fig)

# --- EXPORT SECTION ---
st.subheader("Data Export")
csv = feeder_data.to_csv(index=False).encode('utf-8')
st.download_button(label="📥 Download High Risk List (CSV)", data=csv, file_name="SniffIt_Theft_Report.csv", mime="text/csv")

# --- FOOTER ---
st.markdown("---")
st.markdown(f"<div style='text-align: center; color: #D4AF37;'>Built by Elvis Ebenuwah for Ikeja Electric. SniffIt🐘 2026.</div>", unsafe_allow_html=True)
