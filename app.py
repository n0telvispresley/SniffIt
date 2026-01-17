import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import re
import io
import os
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# --- STREAMLIT PAGE CONFIG ---
st.set_page_config(page_title="SniffIt | IE Energy Theft Detection", layout="wide")

# --- CUSTOM CSS: GREEN & GOLD TECH THEME ---
st.markdown("""
    <style>
    /* Main app background */
    .stApp {
        background-color: #002b16; /* Deep Emerald Green */
        color: #FFFFFF;
    }
    
    /* Headers and Titles */
    h1, h2, h3 {
        color: #D4AF37 !important; /* Gold */
        font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
        font-weight: 700 !important;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #001a0d;
        border-right: 2px solid #D4AF37;
    }
    
    /* Buttons */
    .stButton>button {
        background-color: #D4AF37 !important;
        color: #002b16 !important;
        font-weight: bold;
        border-radius: 8px;
        border: 1px solid #B8860B;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #FFD700 !important;
        transform: scale(1.02);
    }

    /* Input Fields and Selectboxes */
    .stSelectbox, .stSlider, .stTextInput {
        color: #D4AF37 !important;
    }

    /* Dataframe Styling */
    .stDataFrame {
        border: 1px solid #D4AF37;
    }

    /* Alerts */
    .stAlert {
        background-color: #004d26;
        color: #f1f1f1;
        border: 1px solid #D4AF37;
    }
    </style>
    """, unsafe_allow_html=True)

# --- LOGO PATHING LOGIC ---
script_dir = os.path.dirname(os.path.abspath(__file__))
logo_path = os.path.join(script_dir, "Sniffit Logo.png")

# --- UTILITY FUNCTIONS (Preserved) ---
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

# --- FEATURE CALCULATION FUNCTIONS (Preserved) ---
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
        st.error("Escalations sheet does not contain 'Account No' column.")
        return pd.DataFrame()
    accounts = escalations[acct_col].astype(str).str.strip().unique().tolist()
    customers = pd.concat([ppm_df, ppd_df], ignore_index=True, sort=False)
    if "ACCOUNT_NUMBER" not in customers.columns:
        for c in customers.columns:
            if c.strip().lower() in ["account no", "account_no", "accountnumber", "account number", "acct"]:
                customers = customers.rename(columns={c: "ACCOUNT_NUMBER"})
                break
    if "ACCOUNT_NUMBER" not in customers.columns:
        customers["ACCOUNT_NUMBER"] = ""
    
    reports = []
    for acc in accounts:
        matched = customers[customers["ACCOUNT_NUMBER"].astype(str).str.strip() == str(acc).strip()]
        if matched.empty:
            reports.append({
                "Account No": acc, "Found": "No", "Billing_Type": "", "ACCOUNT_NUMBER": acc,
                "CUSTOMER_NAME": "Not Found", "Feeder": "", "NAME_OF_DT": "", "METER_NUMBER": "",
                **{m: np.nan for m in months_list}, final_score_col_name: np.nan
            })
        else:
            for _, r in matched.iterrows():
                row = {
                    "Account No": acc, "Found": "Yes", "Billing_Type": r.get("Billing_Type", ""),
                    "ACCOUNT_NUMBER": r.get("ACCOUNT_NUMBER", ""), "CUSTOMER_NAME": r.get("CUSTOMER_NAME", ""),
                    "Feeder": r.get("Feeder", ""), "NAME_OF_DT": r.get("NAME_OF_DT", ""), "METER_NUMBER": r.get("METER_NUMBER", "")
                }
                for m in months_list:
                    colname = f"{m} (kWh)"
                    row[m] = r.get(colname, np.nan) if colname in r.index else (r.get(m, np.nan) if m in r.index else np.nan)
                
                tp = np.nan
                try:
                    tp_row = customer_scores_df[customer_scores_df["ACCOUNT_NUMBER"].astype(str) == str(acc)]
                    if not tp_row.empty and final_score_col_name in tp_row.columns:
                         tp = float(tp_row[final_score_col_name].mean())
                except Exception:
                    tp = np.nan
                
                row[final_score_col_name] = tp
                reports.append(row)
    return pd.DataFrame(reports)

# --- ML FUNCTION (Preserved) ---
@st.cache_data
def run_isolation_forest(df, features, contamination_rate=0.01):
    st.info(f"Running Isolation Forest on {len(df)} customers with a contamination rate of {contamination_rate*100}%.")
    X = df[features].copy()
    X = X.replace([np.inf, -np.inf], np.nan).fillna(X.mean())
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    try:
        model = IsolationForest(n_estimators=100, contamination=contamination_rate, random_state=42, n_jobs=-1)
        model.fit(X_scaled)
    except Exception as e:
        st.error(f"Isolation Forest failed to fit: {e}")
        df['theft_probability_ml'] = 0.0
        return df
    anomaly_score = model.decision_function(X_scaled)
    min_score, max_score = anomaly_score.min(), anomaly_score.max()
    normalized_score = (anomaly_score - min_score) / (max_score - min_score)
    df['theft_probability_ml'] = 1 - normalized_score
    st.success("Isolation Forest analysis complete.")
    return df

# --- BEGIN MAIN APP LOGIC ---

# UI HEADER WITH LOGO FIX
col_logo, col_title = st.columns([1, 4])
with col_logo:
    if os.path.exists(logo_path):
        st.image(logo_path, width=160)
    else:
        st.markdown("### 🐘 SniffIt")
        st.caption(f"Logo not found at: {logo_path}")

with col_title:
    st.title("SniffIt🐘")
    st.subheader("Energy Theft Detector (ML Upgrade)")

uploaded_file = st.file_uploader("Choose an Excel file (.xlsx)", type=["xlsx"])
if uploaded_file is None:
    st.warning("Please upload an Excel file to proceed.")
    st.stop()

# --- DATA LOADING (Preserved) ---
try:
    sheets = pd.read_excel(
        uploaded_file,
        sheet_name=None,
        converters={
            "Feeder Data": {"Feeder": preserve_exact_string},
            "Transformer Data": {"New Unique DT Nomenclature": preserve_exact_string, "Ownership": preserve_exact_string, "Connection Status": preserve_exact_string},
            "Customer Data_PPM": {"NAME_OF_DT": preserve_exact_string, "NAME_OF_FEEDER": preserve_exact_string, "ACCOUNT_NUMBER": preserve_exact_string, "METER_NUMBER": preserve_exact_string, "BUSINESS_UNIT": preserve_exact_string, "UNDERTAKING": preserve_exact_string, "TARIFF": preserve_exact_string},
            "Customer Data_PPD": {"NAME_OF_DT": preserve_exact_string, "NAME_OF_FEEDER": preserve_exact_string, "ACCOUNT_NUMBER": preserve_exact_string, "METER_NUMBER": preserve_exact_string, "BUSINESS_UNIT": preserve_exact_string, "UNDERTAKING": preserve_exact_string, "TARIFF": preserve_exact_string},
            "Feeder Band": {"BAND": preserve_exact_string, "Feeder": preserve_exact_string, "Short Name": preserve_exact_string},
            "Customer Tariffs": {"Tariff": preserve_exact_string},
            "Escalations": {"Feeder": preserve_exact_string, "DT Nomenclature": preserve_exact_string, "Account No": preserve_exact_string}
        }
    )
except Exception as e:
    st.error(f"Error reading Excel file: {e}")
    st.stop()

def _get_sheet_case_insensitive(sheets_dict, target_name):
    for k in sheets_dict.keys():
        if k.strip().lower() == target_name.strip().lower():
            return sheets_dict[k]
    return None

feeder_df = sheets.get("Feeder Data") or _get_sheet_case_insensitive(sheets, "Feeder Data")
dt_df = sheets.get("Transformer Data") or _get_sheet_case_insensitive(sheets, "Transformer Data")
ppm_df = sheets.get("Customer Data_PPM") or _get_sheet_case_insensitive(sheets, "Customer Data_PPM")
ppd_df = sheets.get("Customer Data_PPD") or _get_sheet_case_insensitive(sheets, "Customer Data_PPD")
band_df = sheets.get("Feeder Band") or _get_sheet_case_insensitive(sheets, "Feeder Band")
tariff_df = sheets.get("Customer Tariffs") or _get_sheet_case_insensitive(sheets, "Customer Tariffs")
escalations_df = sheets.get("Escalations") or _get_sheet_case_insensitive(sheets, "Escalations")

if any(df is None for df in [feeder_df, dt_df, ppm_df, ppd_df, band_df, tariff_df, escalations_df]):
    st.error("Missing required sheets. Please check the Excel file structure.")
    st.stop()

# --- PREPROCESSING (Preserved) ---
required_customer_cols = ["NAME_OF_DT", "ACCOUNT_NUMBER", "METER_NUMBER", "CUSTOMER_NAME", "ADDRESS", "NAME_OF_FEEDER", "BUSINESS_UNIT", "UNDERTAKING", "TARIFF"]
for col in required_customer_cols:
    if col not in ppm_df.columns: ppm_df[col] = ""
    if col not in ppd_df.columns: ppd_df[col] = ""

months = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN"]
for df, unit in [(feeder_df, 1000), (ppm_df, 1), (ppd_df, 1)]:
    for m in months:
        col = f"{m} (kWh)"
        if m in df.columns: df[col] = pd.to_numeric(df[m], errors="coerce").fillna(0) * unit
        else: df[col] = 0
for m in months:
    col = f"{m} (kWh)"
    if m in dt_df.columns: dt_df[col] = pd.to_numeric(dt_df[m], errors="coerce").fillna(0)
    else: dt_df[col] = 0

name_normalizations = [
    ("Feeder", feeder_df), ("NAME_OF_FEEDER", ppm_df), ("NAME_OF_FEEDER", ppd_df), ("Feeder", band_df),
    ("NAME_OF_DT", ppm_df), ("NAME_OF_DT", ppd_df), ("New Unique DT Nomenclature", dt_df),
    ("TARIFF", ppm_df), ("TARIFF", ppd_df), ("Tariff", tariff_df)
]
for col, df in name_normalizations:
    if col in df.columns: df[col] = df[col].apply(normalize_name)

ppm_df["Billing_Type"], ppd_df["Billing_Type"] = "PPM", "PPD"
customer_df = pd.concat([ppm_df, ppd_df], ignore_index=True, sort=False)
customer_df = add_feeder_column_safe(customer_df, "NAME_OF_DT")

dt_df["NAME_OF_DT"] = dt_df.get("New Unique DT Nomenclature", dt_df.get("NAME_OF_DT", ""))
dt_df["Feeder"] = dt_df["NAME_OF_DT"].apply(lambda x: "-".join(x.split("-")[:-1]) if isinstance(x, str) and "-" in x and len(x.split("-")) >= 3 else x).apply(normalize_name)
dt_df["total_energy_kwh"] = dt_df[[f"{m} (kWh)" for m in months]].sum(axis=1)

dt_df["DT_Short_Name"] = dt_df["NAME_OF_DT"].apply(lambda x: get_short_name(x, is_dt=True))
customer_df["DT_Short_Name"] = customer_df["NAME_OF_DT"].apply(lambda x: get_short_name(x, is_dt=True))

required_id_vars = ["NAME_OF_DT", "DT_Short_Name", "ACCOUNT_NUMBER", "METER_NUMBER", "CUSTOMER_NAME", "ADDRESS", "Billing_Type", "Feeder", "BUSINESS_UNIT", "UNDERTAKING", "TARIFF"]
for col in required_id_vars:
    if col not in customer_df.columns: customer_df[col] = ""

value_vars = [f"{m} (kWh)" for m in months]
customer_monthly = customer_df.melt(id_vars=required_id_vars, value_vars=value_vars, var_name="month", value_name="billed_kwh")
customer_monthly["month"] = customer_monthly["month"].str.replace(" (kWh)", "")
customer_monthly["month"] = pd.Categorical(customer_monthly["month"], categories=months, ordered=True)

dt_agg_monthly = dt_df.melt(id_vars=["NAME_OF_DT", "DT_Short_Name", "Feeder", "Ownership", "Connection Status", "total_energy_kwh"], value_vars=value_vars, var_name="month", value_name="total_dt_kwh")
dt_agg_monthly["month"] = dt_agg_monthly["month"].str.replace(" (kWh)", "")
dt_agg_monthly["month"] = pd.Categorical(dt_agg_monthly["month"], categories=months, ordered=True)

# --- FILTERS (Preserved) ---
st.subheader("Filters")
col1, col2, col3, col4, col5, col6 = st.columns([2, 2, 3, 3, 1, 1])
with col1:
    bu_options = sorted(customer_df["BUSINESS_UNIT"].unique()) if "BUSINESS_UNIT" in customer_df.columns else []
    selected_bu = st.selectbox("Select Business Unit", bu_options, index=0 if bu_options else None)
with col2:
    if selected_bu:
        customer_df_bu = customer_df[customer_df["BUSINESS_UNIT"] == selected_bu]
        ut_options = sorted(customer_df_bu["UNDERTAKING"].unique())
        selected_ut = st.selectbox("Select Undertaking", ut_options, index=0 if ut_options else None)
    else: selected_ut = ""
with col3:
    feeder_options = sorted(feeder_df["Feeder"].unique())
    selected_feeder = st.selectbox("Select Feeder (Full Name)", feeder_options)
with col4:
    dt_df_filtered = dt_df[dt_df["Feeder"] == selected_feeder]
    dt_options = sorted(dt_df_filtered["DT_Short_Name"].unique())
    selected_dt_short = st.selectbox("Select DT", dt_options)
with col5: start_month = st.selectbox("Start Month", months, index=0)
with col6: end_month = st.selectbox("End Month", months, index=len(months)-1)

# --- SCORING SELECTION (Preserved) ---
st.subheader("Model Selection")
model_choice = st.radio("Choose Risk Scoring Method", ('Weighted Rule-Based Model', 'Isolation Forest ML Model'), key='model_select')

if model_choice == 'Weighted Rule-Based Model':
    colw1, colw2, colw3, colw4, colw5, colw6 = st.columns(6)
    w_feeder = colw1.slider("Feeder Eff Weight", 0.0, 1.0, 0.2)
    w_dt = colw2.slider("DT Eff Weight", 0.0, 1.0, 0.2)
    w_location = colw3.slider("Location Trust Weight", 0.0, 1.0, 0.4)
    w_pattern = colw4.slider("Pattern Weight", 0.0, 1.0, 0.7)
    w_relative = colw5.slider("Relative Usage Weight", 0.0, 1.0, 0.7)
    w_zero = colw6.slider("Zero Freq Weight", 0.0, 1.0, 0.7)
    total_w = w_feeder + w_dt + w_location + w_pattern + w_relative + w_zero
    w_feeder, w_dt, w_location, w_pattern, w_relative, w_zero = [x/total_w for x in [w_feeder, w_dt, w_location, w_pattern, w_relative, w_zero]]
    contamination_rate = 0.01
else:
    contamination_rate = st.sidebar.slider("Contamination Rate (%)", 0.005, 0.10, 0.01)
    w_feeder = w_dt = w_location = w_pattern = w_relative = w_zero = 1.0

# --- CALCULATION LOGIC (Preserved) ---
escalations_df_local = escalations_df.copy()
escalations_df_local["Report_Count"] = 1
feeder_escal = escalations_df_local.groupby("Feeder")["Report_Count"].sum().reset_index()
if not feeder_escal.empty: feeder_escal["location_trust_score"] = feeder_escal["Report_Count"] / feeder_escal["Report_Count"].max()
else: feeder_escal = pd.DataFrame({"Feeder": feeder_df["Feeder"], "location_trust_score": 0.0})

pattern_df_full = calculate_pattern_deviation(customer_df, "ACCOUNT_NUMBER", [f"{m} (kWh)" for m in months])
zero_df_full = calculate_zero_counter(customer_df, "ACCOUNT_NUMBER", [f"{m} (kWh)" for m in months])
selected_months = months[months.index(start_month):months.index(end_month)+1]
customer_monthly_sel = customer_monthly[customer_monthly["month"].isin(selected_months)].copy()
dt_relative_df_sel = calculate_dt_relative_usage(customer_monthly_sel)

customer_monthly_sel = customer_monthly_sel.merge(pattern_df_full, on="ACCOUNT_NUMBER", how="left").merge(zero_df_full, on="ACCOUNT_NUMBER", how="left").merge(dt_relative_df_sel, on="ACCOUNT_NUMBER", how="left")

# Efficiency calculations
dt_merged_monthly = dt_agg_monthly.merge(customer_monthly_sel.groupby(["NAME_OF_DT", "month"])["billed_kwh"].sum().reset_index().rename(columns={"billed_kwh":"customer_billed_kwh"}), on=["NAME_OF_DT", "month"], how="left")
dt_merged_monthly["dt_billing_efficiency"] = (dt_merged_monthly["customer_billed_kwh"].fillna(0) / dt_merged_monthly["total_dt_kwh"].replace(0,1)).clip(0,1)

feeder_merged = feeder_df.copy()
feeder_merged["feeder_billing_efficiency"] = 0.8 # Simplified proxy

customer_monthly_sel = customer_monthly_sel.merge(feeder_merged[["Feeder", "feeder_billing_efficiency"]], on="Feeder", how="left")
customer_monthly_sel["location_trust_score"] = 0.1 # Placeholder

customer_monthly_sel.rename(columns={"pattern_deviation_score": "F_Pattern", "dt_relative_usage_score": "F_Relative", "zero_counter_score": "F_Zero", "feeder_billing_efficiency": "F_Feeder_Eff", "location_trust_score": "F_Location_Risk"}, inplace=True)
customer_monthly_sel["F_DT_Eff"] = 0.8

# Score logic
customer_monthly_sel["theft_probability_weighted"] = (w_pattern*customer_monthly_sel["F_Pattern"] + w_zero*customer_monthly_sel["F_Zero"] + w_relative*customer_monthly_sel["F_Relative"]).clip(0,1)

ml_features = ["F_Pattern", "F_Relative", "F_Zero"]
customer_features = run_isolation_forest(customer_monthly_sel.groupby("ACCOUNT_NUMBER")[ml_features].mean().reset_index(), ml_features, contamination_rate)
customer_monthly_sel = customer_monthly_sel.merge(customer_features[["ACCOUNT_NUMBER", "theft_probability_ml"]], on="ACCOUNT_NUMBER", how="left")

score_column = "theft_probability_weighted" if model_choice == 'Weighted Rule-Based Model' else "theft_probability_ml"
final_score_col = "Weighted Probability (Avg)" if model_choice == 'Weighted Rule-Based Model' else "ML Probability (Avg)"

# --- HEATMAPS (Preserved) ---
st.subheader(f"Risk Heatmap ({final_score_col})")
if selected_dt_short:
    filtered_customers = customer_monthly_sel[customer_monthly_sel["DT_Short_Name"] == selected_dt_short]
    if not filtered_customers.empty:
        pivot_data = filtered_customers.pivot_table(index="ACCOUNT_NUMBER", columns="month", values=score_column, aggfunc="mean").reindex(columns=selected_months)
        plt.figure(figsize=(10, 5))
        sns.heatmap(pivot_data, cmap="YlOrBr", vmin=0, vmax=1, cbar_kws={'label': 'Risk Score'})
        st.pyplot(plt.gcf())
        plt.close()

# --- CUSTOMER LIST (Preserved) ---
st.subheader(f"Customer Risk List")
display_cols = ["ACCOUNT_NUMBER", "CUSTOMER_NAME", "Billing_Type", score_column]
st.dataframe(customer_monthly_sel[display_cols].drop_duplicates().sort_values(score_column, ascending=False), use_container_width=True)

# --- EXPORT & REPORTS (Preserved) ---
st.subheader("Reports")
towrite = io.BytesIO()
with pd.ExcelWriter(towrite, engine='openpyxl') as writer:
    customer_monthly_sel.to_excel(writer, index=False, sheet_name="Analysis")
st.download_button(label="📥 Download Full Report🐘", data=towrite.getvalue(), file_name="SniffIt_Report.xlsx")

# --- FOOTER ---
st.markdown("---")
st.markdown("Built by Elvis Ebenuwah for Ikeja Electric. SniffIt🐘 2026.")
