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

# --- STREAMLIT PAGE CONFIG (Browser Tab Icon) ---
try:
    fav_icon = Image.open(logo_path)
    st.set_page_config(page_title="SniffIt | Energy Theft Detection AI", page_icon=fav_icon, layout="wide")
except Exception:
    st.set_page_config(page_title="SniffIt | Energy Theft Detection AI", page_icon="🐘", layout="wide")

# --- CUSTOM CSS: EMERALD GREEN & METALLIC GOLD THEME ---
st.markdown("""
    <style>
    .stApp { background-color: #002b16; color: #FFFFFF; }
    h1, h2, h3, h4, h5, h6 { color: #D4AF37 !important; font-family: 'Inter', 'Segoe UI', sans-serif; font-weight: 700 !important; }
    [data-testid="stSidebar"] { background-color: #001a0d; border-right: 2px solid #D4AF37; }
    .stButton>button { background-color: #D4AF37 !important; color: #002b16 !important; font-weight: bold; border-radius: 8px; border: 1px solid #B8860B; transition: 0.3s; }
    .stButton>button:hover { background-color: #FFD700 !important; transform: scale(1.02); box-shadow: 0px 0px 15px rgba(212, 175, 55, 0.4); }
    .stSelectbox, .stSlider, .stTextInput, .stRadio { color: #D4AF37 !important; }
    .stDataFrame { border: 1px solid #D4AF37; background-color: #001a0d; }
    .stAlert { background-color: #004d26; color: #f1f1f1; border: 1px solid #D4AF37; }
    .stPlot { background-color: #ffffff; padding: 10px; border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

def get_base64_img(path):
    with open(path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

# --- Utility Functions ---
def preserve_exact_string(value):
    return str(value) if pd.notna(value) else ""

def normalize_name(name):
    if not isinstance(name, str): return ""
    name = re.sub(r'\s+', ' ', name.strip().upper())
    name = re.sub(r'[^\w\s-]', '', name)
    return re.sub(r'-+', '-', name)

def get_short_name(name, is_dt=False):
    if isinstance(name, str) and "-" in name:
        return name.split("-")[-1].strip()
    return name if isinstance(name, str) else ""

def add_feeder_column_safe(df, name_of_dt_col="NAME_OF_DT"):
    if name_of_dt_col not in df.columns:
        df["Feeder"] = ""
        return df
    df = df.copy()
    df["Feeder"] = df[name_of_dt_col].apply(lambda x: "-".join(x.split("-")[:-1]) if isinstance(x, str) and len(x.split("-")) >= 3 else x)
    df["Feeder"] = df["Feeder"].apply(normalize_name)
    return df

# --- Calculation Functions ---
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
        score = np.sum(values == 0) / len(valid_cols) if len(valid_cols) > 0 else 0
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
                score_match = customer_scores_df[customer_scores_df["ACCOUNT_NUMBER"].astype(str) == str(acc)]
                row[final_score_col_name] = score_match[final_score_col_name].mean() if not score_match.empty else np.nan
                reports.append(row)
    return pd.DataFrame(reports)

@st.cache_data
def run_isolation_forest(df, features, contamination_rate=0.01):
    X = df[features].copy().fillna(0)
    scaler = StandardScaler()
    try:
        X_scaled = scaler.fit_transform(X)
        model = IsolationForest(n_estimators=100, contamination=contamination_rate, random_state=42)
        model.fit(X_scaled)
        anomaly_score = model.decision_function(X_scaled)
        df['theft_probability_ml'] = 1 - (anomaly_score - anomaly_score.min()) / (anomaly_score.max() - anomaly_score.min() + 1e-9)
    except: df['theft_probability_ml'] = 0.0
    return df

# --- UI HEADER ---
col_logo, col_title = st.columns([1, 4])
with col_logo:
    if os.path.exists(logo_path):
        img_b64 = get_base64_img(logo_path)
        st.markdown(f'<img src="data:image/jpeg;base64,{img_b64}" style="border-radius: 50%; width: 140px; height: 140px; object-fit: cover; border: 4px solid #D4AF37; display: block; margin: auto;">', unsafe_allow_html=True)
    else: st.markdown("### 🐘 SniffIt")

with col_title:
    st.title("SniffIt🐘")
    st.subheader("Energy Theft Detector (ML + Rules Engine)")

uploaded_file = st.file_uploader("Choose an Excel file (.xlsx)", type=["xlsx"])
if uploaded_file is None: st.warning("Please upload an Excel file to proceed."); st.stop()

# --- DATA LOADING ---
sheets = pd.read_excel(uploaded_file, sheet_name=None)
def load_sheet(name):
    for k in sheets.keys():
        if k.strip().lower() == name.lower(): return sheets[k]
    return None

feeder_df, dt_df = load_sheet("Feeder Data"), load_sheet("Transformer Data")
ppm_df, ppd_df = load_sheet("Customer Data_PPM"), load_sheet("Customer Data_PPD")
band_df, tariff_df = load_sheet("Feeder Band"), load_sheet("Customer Tariffs")
escalations_df = load_sheet("Escalations")

if any(x is None for x in [feeder_df, dt_df, ppm_df, ppd_df]): st.error("Essential sheets missing!"); st.stop()

# --- PREPROCESSING ---
months = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN"]
for df_ref, unit in [(feeder_df, 1000), (ppm_df, 1), (ppd_df, 1), (dt_df, 1)]:
    for m in months:
        col_name = f"{m} (kWh)"
        df_ref[col_name] = pd.to_numeric(df_ref[m], errors="coerce").fillna(0) * unit

dt_df["total_energy_kwh"] = dt_df[[f"{m} (kWh)" for m in months]].sum(axis=1)
for col in ["Ownership", "Connection Status"]:
    if col not in dt_df.columns: dt_df[col] = "Unknown"

ppm_df["Billing_Type"], ppd_df["Billing_Type"] = "PPM", "PPD"
customer_df = pd.concat([ppm_df, ppd_df], ignore_index=True, sort=False)
customer_df = add_feeder_column_safe(customer_df)

dt_df["NAME_OF_DT"] = dt_df.get("New Unique DT Nomenclature", dt_df.get("NAME_OF_DT", ""))
dt_df["Feeder"] = dt_df["NAME_OF_DT"].apply(lambda x: "-".join(x.split("-")[:-1]) if isinstance(x, str) and "-" in x else x).apply(normalize_name)
dt_df["DT_Short_Name"] = dt_df["NAME_OF_DT"].apply(lambda x: get_short_name(x, is_dt=True))
customer_df["DT_Short_Name"] = customer_df["NAME_OF_DT"].apply(lambda x: get_short_name(x, is_dt=True))

id_vars_cust = ["NAME_OF_DT", "DT_Short_Name", "ACCOUNT_NUMBER", "METER_NUMBER", "CUSTOMER_NAME", "ADDRESS", "Billing_Type", "Feeder", "BUSINESS_UNIT", "UNDERTAKING", "TARIFF"]
for col in id_vars_cust: 
    if col not in customer_df.columns: customer_df[col] = ""

customer_monthly = customer_df.melt(id_vars=id_vars_cust, value_vars=[f"{m} (kWh)" for m in months], var_name="month", value_name="billed_kwh")
customer_monthly["month"] = customer_monthly["month"].str.replace(" (kWh)", "").str.strip()
customer_monthly["month"] = pd.Categorical(customer_monthly["month"], categories=months, ordered=True)

dt_agg_monthly = dt_df.melt(id_vars=["NAME_OF_DT", "DT_Short_Name", "Feeder", "Ownership", "Connection Status", "total_energy_kwh"], value_vars=[f"{m} (kWh)" for m in months], var_name="month", value_name="total_dt_kwh")
dt_agg_monthly["month"] = dt_agg_monthly["month"].str.replace(" (kWh)", "").str.strip()
dt_agg_monthly["month"] = pd.Categorical(dt_agg_monthly["month"], categories=months, ordered=True)

# --- FILTERS ---
st.subheader("Filters 🐘")
f1, f2, f3, f4, f5, f6 = st.columns([2, 2, 3, 3, 1, 1])
with f1: sel_bu = st.selectbox("Business Unit", sorted(customer_df["BUSINESS_UNIT"].unique()))
with f2: sel_ut = st.selectbox("Undertaking", sorted(customer_df[customer_df["BUSINESS_UNIT"] == sel_bu]["UNDERTAKING"].unique()))
with f3: sel_feeder = st.selectbox("Feeder", sorted(customer_df[customer_df["UNDERTAKING"] == sel_ut]["Feeder"].unique()))
with f4: sel_dt = st.selectbox("DT", sorted(customer_df[customer_df["Feeder"] == sel_feeder]["DT_Short_Name"].unique()))
with f5: s_month = st.selectbox("Start Month", months, index=0)
with f6: e_month = st.selectbox("End Month", months, index=5)

model_choice = st.radio("Model Selection", ('Weighted Rule-Based Model', 'Isolation Forest ML Model'))
sel_range = months[months.index(s_month):months.index(e_month)+1]

# --- CALCULATION PIPELINE ---
pattern_scores = calculate_pattern_deviation(customer_df, "ACCOUNT_NUMBER", [f"{m} (kWh)" for m in months])
zero_scores = calculate_zero_counter(customer_df, "ACCOUNT_NUMBER", [f"{m} (kWh)" for m in months])
cust_sel = customer_monthly[customer_monthly["month"].isin(sel_range)].copy()
rel_scores = calculate_dt_relative_usage(cust_sel)

cust_sel = cust_sel.merge(pattern_scores, on="ACCOUNT_NUMBER", how="left").merge(zero_scores, on="ACCOUNT_NUMBER", how="left").merge(rel_scores, on="ACCOUNT_NUMBER", how="left")
cust_sel.rename(columns={"pattern_deviation_score": "F_Pattern", "dt_relative_usage_score": "F_Relative", "zero_counter_score": "F_Zero"}, inplace=True)

# Efficiency for Ranking
cust_billed = cust_sel.groupby(["NAME_OF_DT", "month"], as_index=False)["billed_kwh"].sum().rename(columns={"billed_kwh":"cust_billed_kwh"})
dt_merged_m = dt_agg_monthly.merge(cust_billed, on=["NAME_OF_DT", "month"], how="left")
dt_merged_m["dt_billing_efficiency"] = (dt_merged_m["cust_billed_kwh"].fillna(0) / dt_merged_m["total_dt_kwh"].replace(0,1)).clip(0,1)

if model_choice == 'Weighted Rule-Based Model':
    cust_sel["final_score"] = (0.4*cust_sel["F_Pattern"] + 0.3*cust_sel["F_Zero"] + 0.3*cust_sel["F_Relative"]).clip(0,1)
    score_col_name = "Weighted Probability"
else:
    ml_input = cust_sel.groupby("ACCOUNT_NUMBER")[["F_Pattern", "F_Relative", "F_Zero"]].mean().reset_index()
    ml_res = run_isolation_forest(ml_input, ["F_Pattern", "F_Relative", "F_Zero"])
    cust_sel = cust_sel.merge(ml_res[["ACCOUNT_NUMBER", "theft_probability_ml"]], on="ACCOUNT_NUMBER", how="left")
    cust_sel["final_score"] = cust_sel["theft_probability_ml"]
    score_col_name = "ML Probability"

# --- OUTPUT: DT HEATMAP & LIST ---
st.subheader("DT Efficiency Risk Heatmap")
dt_feeder_data = dt_merged_m[dt_merged_m["Feeder"] == sel_feeder]
if not dt_feeder_data.empty:
    dt_order = dt_feeder_data.groupby("DT_Short_Name")["dt_billing_efficiency"].mean().sort_values().index
    dt_piv = dt_feeder_data.pivot_table(index="DT_Short_Name", columns="month", values="dt_billing_efficiency").reindex(index=dt_order, columns=sel_range)
    plt.figure(figsize=(10, 6)); sns.heatmap(1 - dt_piv, cmap="Reds", vmin=0, vmax=1, annot=True); st.pyplot(plt.gcf()); plt.close()

# --- OUTPUT: CUSTOMER HEATMAP & LIST ---
st.subheader("Customer Theft Probability Heatmap")
filtered_c = cust_sel[cust_sel["DT_Short_Name"] == sel_dt]
if not filtered_c.empty:
    top_acc = filtered_c.groupby("ACCOUNT_NUMBER")["final_score"].mean().sort_values(ascending=False).head(15).index
    cust_piv = filtered_c[filtered_c["ACCOUNT_NUMBER"].isin(top_acc)].pivot_table(index="ACCOUNT_NUMBER", columns="month", values="final_score").reindex(index=top_acc, columns=sel_range)
    plt.figure(figsize=(10, 6)); sns.heatmap(cust_piv, cmap="Reds", vmin=0, vmax=1, annot=True); st.pyplot(plt.gcf()); plt.close()

st.subheader("High Risk Customer List")
display_list = filtered_c.groupby(["ACCOUNT_NUMBER", "CUSTOMER_NAME", "METER_NUMBER", "Billing_Type"], as_index=False)["final_score"].mean().sort_values("final_score", ascending=False)
st.dataframe(display_list.style.format({"final_score": "{:.3f}"}), use_container_width=True)

# --- REPORTS ---
st.subheader("Escalations & Exports")
cust_scores_avg = cust_sel.groupby("ACCOUNT_NUMBER", as_index=False).agg({"final_score": "mean"}).rename(columns={"final_score": score_col_name})
escal_report = generate_escalations_report(ppm_df, ppd_df, escalations_df, cust_scores_avg, months, score_col_name)
towrite = io.BytesIO()
with pd.ExcelWriter(towrite, engine='openpyxl') as writer:
    if not escal_report.empty: escal_report.to_excel(writer, index=False, sheet_name="Escalations")
    display_list.to_excel(writer, index=False, sheet_name="Top Risks")
st.download_button(label="📥 Download Full Analysis (Excel)🐘", data=towrite.getvalue(), file_name="SniffIt_Report.xlsx")

st.markdown("---")
st.markdown("Built by Elvis Ebenuwah for Ikeja Electric. SniffIt🐘 2026.")
