import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import base64
import seaborn as sns
import re
import io
import os
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from PIL import Image 

# --- LOGO PATHING LOGIC ---
script_dir = os.path.dirname(os.path.abspath(__file__))
logo_filename = "IMG_4445.jpeg"
logo_path = os.path.join(script_dir, logo_filename)

# --- STREAMLIT PAGE CONFIG (LOGO ON BROWSER TAB) ---
try:
    fav_icon = Image.open(logo_path)
    st.set_page_config(
        page_title="SniffIt | IE Energy Theft Detection AI", 
        page_icon=fav_icon, 
        layout="wide"
    )
except Exception:
    st.set_page_config(
        page_title="SniffIt | IE Energy Theft Detection AI", 
        page_icon="🐘", 
        layout="wide"
    )

# --- CUSTOM CSS: EMERALD GREEN & METALLIC GOLD THEME ---
st.markdown("""
    <style>
    .stApp {
        background-color: #002b16; 
        color: #FFFFFF;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #D4AF37 !important; 
        font-family: 'Inter', 'Segoe UI', sans-serif;
        font-weight: 700 !important;
    }
    [data-testid="stSidebar"] {
        background-color: #001a0d;
        border-right: 2px solid #D4AF37;
    }
    .stButton>button {
        background-color: #D4AF37 !important;
        color: #002b16 !important;
        font-weight: bold;
        border-radius: 8px;
        border: 1px solid #B8860B;
        transition: 0.3s ease-in-out;
    }
    .stButton>button:hover {
        background-color: #FFD700 !important;
        transform: scale(1.02);
        box-shadow: 0px 0px 15px rgba(212, 175, 55, 0.4);
    }
    .stSelectbox, .stSlider, .stTextInput, .stRadio {
        color: #D4AF37 !important;
    }
    .stDataFrame {
        border: 1px solid #D4AF37;
        background-color: #001a0d;
    }
    .stAlert {
        background-color: #004d26;
        color: #f1f1f1;
        border: 1px solid #D4AF37;
    }
    .stPlot {
        background-color: #ffffff;
        padding: 10px;
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

def get_base64_img(path):
    with open(path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

# --- Utility Functions (Preserved) ---
def preserve_exact_string(value):
    if pd.isna(value) or value is None: return ""
    return str(value)

def normalize_name(name):
    if not isinstance(name, str): return ""
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
        df["Feeder"] = ""
        return df
    df = df.copy()
    df["Feeder"] = df[name_of_dt_col].apply(
        lambda x: "-".join(x.split("-")[:-1]) if isinstance(x, str) and "-" in x and len(x.split("-")) >= 3 else x
    )
    df["Feeder"] = df["Feeder"].apply(normalize_name)
    return df

# --- Feature Calculation Functions (Preserved) ---
def calculate_pattern_deviation(df, id_col, value_cols):
    results = []
    valid_cols = [c for c in value_cols if c in df.columns]
    for id_val, group in df.groupby(id_col):
        values = group[valid_cols].iloc[0].values.astype(float)
        nonzero = values[values > 0]
        score = 1.0 if len(nonzero) == 0 else np.sum(values < 0.6 * np.max(nonzero)) / len(valid_cols)
        results.append({"id": id_val, "pattern_deviation_score": min(score, 1.0)})
    return pd.DataFrame(results).rename(columns={"id": id_col})

def calculate_zero_counter(df, id_col, value_cols):
    results = []
    valid_cols = [c for c in value_cols if c in df.columns]
    for id_val, group in df.groupby(id_col):
        values = group[valid_cols].iloc[0].values.astype(float)
        score = np.sum(values == 0) / len(valid_cols)
        results.append({"id": id_val, "zero_counter_score": min(score, 1.0)})
    return pd.DataFrame(results).rename(columns={"id": id_col})

def calculate_dt_relative_usage(customer_monthly):
    cust_sum = customer_monthly.groupby(["ACCOUNT_NUMBER", "NAME_OF_DT"], as_index=False)["billed_kwh"].sum()
    dt_avg = cust_sum[cust_sum["billed_kwh"] > 0].groupby("NAME_OF_DT", as_index=False)["billed_kwh"].mean().rename(columns={"billed_kwh": "dt_avg_kwh"})
    cust_sum = cust_sum.merge(dt_avg, on="NAME_OF_DT", how="left")
    def _score(row):
        if pd.isna(row["dt_avg_kwh"]) or row["dt_avg_kwh"] == 0: return 0.5
        ratio = row["billed_kwh"] / row["dt_avg_kwh"]
        return 0.9 if ratio < 0.3 else (0.1 if ratio > 0.7 else 0.1 + (0.8 * (0.7 - ratio) / 0.4))
    cust_sum["dt_relative_usage_score"] = cust_sum.apply(_score, axis=1)
    return cust_sum[["ACCOUNT_NUMBER", "dt_relative_usage_score"]]

def generate_escalations_report(ppm_df, ppd_df, escalations_df, customer_scores_df, months_list, final_score_col_name):
    # Standardizing columns for lookup
    escalations = escalations_df.copy()
    acct_col = next((c for c in escalations.columns if c.strip().lower() in ["account no", "account_no", "account number"]), None)
    if acct_col is None: return pd.DataFrame()
    
    customers = pd.concat([ppm_df, ppd_df], ignore_index=True, sort=False)
    if "ACCOUNT_NUMBER" not in customers.columns:
        acct_map = next((c for c in customers.columns if "account" in c.lower() or "acct" in c.lower()), None)
        if acct_map: customers = customers.rename(columns={acct_map: "ACCOUNT_NUMBER"})
    
    reports = []
    accounts = escalations[acct_col].astype(str).str.strip().unique()
    for acc in accounts:
        matched = customers[customers["ACCOUNT_NUMBER"].astype(str).str.strip() == str(acc)]
        if matched.empty:
            reports.append({"Account No": acc, "Found": "No", "CUSTOMER_NAME": "Not Found"})
        else:
            for _, r in matched.iterrows():
                row = {"Account No": acc, "Found": "Yes", "Billing_Type": r.get("Billing_Type", ""),
                       "ACCOUNT_NUMBER": r.get("ACCOUNT_NUMBER", ""), "CUSTOMER_NAME": r.get("CUSTOMER_NAME", ""),
                       "Feeder": r.get("Feeder", ""), "NAME_OF_DT": r.get("NAME_OF_DT", "")}
                tp_val = customer_scores_df[customer_scores_df["ACCOUNT_NUMBER"].astype(str) == str(acc)][final_score_col_name].mean()
                row[final_score_col_name] = tp_val
                reports.append(row)
    return pd.DataFrame(reports)

# --- ML Function ---
@st.cache_data
def run_isolation_forest(df, features, contamination_rate=0.01):
    X = df[features].copy().fillna(0)
    scaler = StandardScaler()
    try:
        X_scaled = scaler.fit_transform(X)
        model = IsolationForest(n_estimators=100, contamination=contamination_rate, random_state=42)
        model.fit(X_scaled)
        scores = model.decision_function(X_scaled)
        df['theft_probability_ml'] = 1 - (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
    except: df['theft_probability_ml'] = 0.0
    return df

# --- UI HEADER ---
col_logo, col_title = st.columns([1, 4])
with col_logo:
    if os.path.exists(logo_path):
        img_b64 = get_base64_img(logo_path)
        st.markdown(f'<img src="data:image/jpeg;base64,{img_b64}" style="border-radius: 50%; width: 160px; height: 160px; object-fit: cover; border: 4px solid #D4AF37; display: block; margin: auto;">', unsafe_allow_html=True)
    else: st.markdown("### 🐘 SniffIt")

with col_title:
    st.title("SniffIt🐘")
    st.subheader("Energy Theft Detector AI")

uploaded_file = st.file_uploader("Upload Network Data (Excel)", type=["xlsx"])
if uploaded_file is None: st.warning("Please upload a file to start."); st.stop()

# --- DATA LOADING (DataFrame Ambiguity Fix) ---
sheets = pd.read_excel(uploaded_file, sheet_name=None)
def load_sheet(name):
    for k in sheets.keys():
        if k.strip().lower() == name.lower(): return sheets[k]
    return None

feeder_df = load_sheet("Feeder Data")
dt_df = load_sheet("Transformer Data")
ppm_df = load_sheet("Customer Data_PPM")
ppd_df = load_sheet("Customer Data_PPD")
band_df = load_sheet("Feeder Band")
tariff_df = load_sheet("Customer Tariffs")
escalations_df = load_sheet("Escalations")

if any(x is None for x in [feeder_df, dt_df, ppm_df, ppd_df]): st.error("Essential sheets missing!"); st.stop()

# --- PREPROCESSING (KeyError Fix) ---
months = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN"]

# Ensure month columns are numeric before summing
for df_ref, unit in [(feeder_df, 1000), (ppm_df, 1), (ppd_df, 1), (dt_df, 1)]:
    for m in months:
        col_name = f"{m} (kWh)"
        if m in df_ref.columns:
            df_ref[col_name] = pd.to_numeric(df_ref[m], errors="coerce").fillna(0) * unit
        else:
            df_ref[col_name] = 0.0

# CALCULATE TOTAL BEFORE MELTING (Fixes Line 473 KeyError)
dt_df["total_energy_kwh"] = dt_df[[f"{m} (kWh)" for m in months]].sum(axis=1)

# Ensure essential columns exist in Transformer Data
for col in ["Ownership", "Connection Status"]:
    if col not in dt_df.columns: dt_df[col] = "Unknown"

ppm_df["Billing_Type"], ppd_df["Billing_Type"] = "PPM", "PPD"
customer_df = pd.concat([ppm_df, ppd_df], ignore_index=True, sort=False)
customer_df = add_feeder_column_safe(customer_df)

# Nomenclature updates
dt_df["NAME_OF_DT"] = dt_df.get("New Unique DT Nomenclature", dt_df.get("NAME_OF_DT", ""))
dt_df["Feeder"] = dt_df["NAME_OF_DT"].apply(lambda x: "-".join(x.split("-")[:-1]) if isinstance(x, str) and "-" in x else x).apply(normalize_name)
dt_df["DT_Short_Name"] = dt_df["NAME_OF_DT"].apply(lambda x: get_short_name(x, is_dt=True))
customer_df["DT_Short_Name"] = customer_df["NAME_OF_DT"].apply(lambda x: get_short_name(x, is_dt=True))

id_vars_cust = ["NAME_OF_DT", "DT_Short_Name", "ACCOUNT_NUMBER", "METER_NUMBER", "CUSTOMER_NAME", "ADDRESS", "Billing_Type", "Feeder", "BUSINESS_UNIT", "UNDERTAKING", "TARIFF"]
for col in id_vars_cust: 
    if col not in customer_df.columns: customer_df[col] = ""

# CATEGORICAL MONTHS (Fixes June-Only Heatmap)
customer_monthly = customer_df.melt(id_vars=id_vars_cust, value_vars=[f"{m} (kWh)" for m in months], var_name="month", value_name="billed_kwh")
customer_monthly["month"] = customer_monthly["month"].str.replace(" (kWh)", "").str.strip()
customer_monthly["month"] = pd.Categorical(customer_monthly["month"], categories=months, ordered=True)

# TRANSFORMATION WITH PRESERVED TOTALS
dt_agg_monthly = dt_df.melt(id_vars=["NAME_OF_DT", "DT_Short_Name", "Feeder", "Ownership", "Connection Status", "total_energy_kwh"], value_vars=[f"{m} (kWh)" for m in months], var_name="month", value_name="total_dt_kwh")
dt_agg_monthly["month"] = dt_agg_monthly["month"].str.replace(" (kWh)", "").str.strip()
dt_agg_monthly["month"] = pd.Categorical(dt_agg_monthly["month"], categories=months, ordered=True)

# --- HIERARCHICAL FILTERS ---
st.subheader("Navigation Filters 🐘")
f1, f2, f3, f4, f5, f6 = st.columns([2, 2, 3, 3, 1, 1])

with f1: sel_bu = st.selectbox("Business Unit", sorted(customer_df["BUSINESS_UNIT"].unique()))
with f2: sel_ut = st.selectbox("Undertaking", sorted(customer_df[customer_df["BUSINESS_UNIT"] == sel_bu]["UNDERTAKING"].unique()))
with f3: sel_feeder = st.selectbox("Feeder", sorted(customer_df[customer_df["UNDERTAKING"] == sel_ut]["Feeder"].unique()))
with f4: sel_dt = st.selectbox("DT (Transformer)", sorted(customer_df[customer_df["Feeder"] == sel_feeder]["DT_Short_Name"].unique()))
with f5: s_month = st.selectbox("Start", months, index=0)
with f6: e_month = st.selectbox("End", months, index=5)

# --- CALCULATIONS & MODELS ---
model_choice = st.radio("Choose Analysis Model", ('Weighted Rule-Based Model', 'Isolation Forest ML Model'))
sel_range = months[months.index(s_month):months.index(e_month)+1]

# Calculations
pattern_scores = calculate_pattern_deviation(customer_df, "ACCOUNT_NUMBER", [f"{m} (kWh)" for m in months])
zero_scores = calculate_zero_counter(customer_df, "ACCOUNT_NUMBER", [f"{m} (kWh)" for m in months])
cust_sel = customer_monthly[customer_monthly["month"].isin(sel_range)].copy()
rel_scores = calculate_dt_relative_usage(cust_sel)

cust_sel = cust_sel.merge(pattern_scores, on="ACCOUNT_NUMBER", how="left").merge(zero_scores, on="ACCOUNT_NUMBER", how="left").merge(rel_scores, on="ACCOUNT_NUMBER", how="left")
cust_sel.rename(columns={"pattern_deviation_score": "F_Pattern", "dt_relative_usage_score": "F_Relative", "zero_counter_score": "F_Zero"}, inplace=True)

# Model Decision
if model_choice == 'Weighted Rule-Based Model':
    cust_sel["final_score"] = (0.4*cust_sel["F_Pattern"] + 0.3*cust_sel["F_Zero"] + 0.3*cust_sel["F_Relative"]).clip(0,1)
    score_col_name = "Weighted Probability"
else:
    ml_input = cust_sel.groupby("ACCOUNT_NUMBER")[["F_Pattern", "F_Relative", "F_Zero"]].mean().reset_index()
    ml_res = run_isolation_forest(ml_input, ["F_Pattern", "F_Relative", "F_Zero"])
    cust_sel = cust_sel.merge(ml_res[["ACCOUNT_NUMBER", "theft_probability_ml"]], on="ACCOUNT_NUMBER", how="left")
    cust_sel["final_score"] = cust_sel["theft_probability_ml"]
    score_col_name = "ML Probability"

# --- VISUALIZATION (FULL RANGE HEATMAP) ---
st.subheader(f"Risk Heatmap: {s_month} to {e_month}")
filtered_c = cust_sel[cust_sel["DT_Short_Name"] == sel_dt]

if not filtered_c.empty:
    # Select Top 15 highest risks for clarity
    top_accounts = filtered_c.groupby("ACCOUNT_NUMBER")["final_score"].mean().sort_values(ascending=False).head(15).index
    piv = filtered_c[filtered_c["ACCOUNT_NUMBER"].isin(top_accounts)].pivot_table(index="ACCOUNT_NUMBER", columns="month", values="final_score")
    
    # REINDEX FIX: Forces the heatmap to show all months in the selected range
    piv = piv.reindex(columns=sel_range)
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(piv, vmin=0, vmax=1, cmap="Reds", annot=True, fmt=".2f", cbar_kws={'label': score_col_name})
    st.pyplot(plt.gcf())
    plt.close()

st.subheader("High Risk Priority List")
risk_summary = filtered_c.groupby(["ACCOUNT_NUMBER", "CUSTOMER_NAME", "METER_NUMBER"], as_index=False)["final_score"].mean().sort_values("final_score", ascending=False)
st.dataframe(risk_summary.style.format({"final_score": "{:.3f}"}), use_container_width=True)

# --- EXPORTS ---
st.subheader("Reports")
csv = risk_summary.to_csv(index=False).encode('utf-8')
st.download_button(label="📥 Download Priority List (CSV)🐘", data=csv, file_name=f"Theft_Risk_{sel_dt}.csv", mime="text/csv")

st.markdown("---")
st.markdown("Built by Elvis Ebenuwah. SniffIt🐘 2026.")
