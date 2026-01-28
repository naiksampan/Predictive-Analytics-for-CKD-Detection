import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import joblib


# ---------------------- Page Config ----------------------
st.set_page_config(
    page_title="CKD Analytics Dashboard",
    page_icon="🩺",
    layout="wide"
)

# ---------------------- Load Data ------------------------
@st.cache_data
def load_data():
    return pd.read_csv('/Users/nebula/Desktop/ME/interview/Clysys/data/ckd_preprocessed_data.csv')

df = load_data()

# ---------------------- Sidebar --------------------------
st.sidebar.title("CKD Dashboard")
st.sidebar.markdown("Publication-quality analytics & visualization")

# Target Column
TARGET = 'class' if 'class' in df.columns else df.columns[-1]

# Feature Selection
num_cols = df.select_dtypes(include=np.number).columns.tolist()
cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()

# ---------------------- Main Title ------------------------
st.title("🩺 Chronic Kidney Disease – Data Analytics & Visualization Dashboard")
st.markdown("""
This dashboard provides **publication-quality visualizations** and **interactive analytics**
for Chronic Kidney Disease (CKD) detection and clinical feature analysis.
""")

# ---------------------- Dataset Overview ------------------
st.subheader("📊 Dataset Overview")

c1, c2, c3 = st.columns(3)
c1.metric("Total Records", df.shape[0])
c2.metric("Total Features", df.shape[1])
c3.metric("Target Column", TARGET)

st.dataframe(df.head(), use_container_width=True)

# ---------------------- Target Distribution ----------------
st.subheader("🎯 Target Distribution")

fig_target = px.histogram(
    df, x=TARGET, color=TARGET,
    title="CKD vs Non-CKD Distribution",
    template="plotly_white"
)
st.plotly_chart(fig_target, use_container_width=True)

# ---------------------- Feature Distributions --------------
st.subheader("📈 Feature Distributions")

selected_feature = st.selectbox("Select Numerical Feature", num_cols)

fig_dist = px.histogram(
    df, x=selected_feature, color=TARGET, marginal="box",
    template="plotly_white",
    title=f"Distribution of {selected_feature}"
)
st.plotly_chart(fig_dist, use_container_width=True)

# ---------------------- Correlation Heatmap ----------------
st.subheader("🔗 Correlation Heatmap")

corr = df[num_cols].corr()

fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(
    corr, cmap="coolwarm", annot=False, linewidths=0.5,
    ax=ax, cbar_kws={'label': 'Correlation Coefficient'}
)
ax.set_title("Feature Correlation Matrix", fontsize=14)
st.pyplot(fig)

# ---------------------- Pairwise Scatter -------------------
st.subheader("📌 Feature Relationship Analysis")

x_axis = st.selectbox("X-axis Feature", num_cols, index=0)
y_axis = st.selectbox("Y-axis Feature", num_cols, index=1)

fig_scatter = px.scatter(
    df, x=x_axis, y=y_axis, color=TARGET,
    template="plotly_white",
    title=f"{x_axis} vs {y_axis}"
)
st.plotly_chart(fig_scatter, use_container_width=True)

# ---------------------- Boxplot Analysis -------------------
st.subheader("📦 Feature vs Target Boxplots")

box_feature = st.selectbox("Select Feature for Boxplot", num_cols, index=2)

fig_box = px.box(
    df, x=TARGET, y=box_feature, color=TARGET,
    template="plotly_white",
    title=f"{box_feature} Distribution by CKD Status"
)
st.plotly_chart(fig_box, use_container_width=True)

# ---------------------- Clinical Focus ---------------------
# Normal ranges mapped EXACTLY to dataset columns
st.subheader("🩺 Patient Clinical Profiles with Normal Range Overlays")

scaler_cols = [
    'age','blood_pressure','blood_glucose_random','blood_urea',
    'serum_creatinine','sodium','potassium','haemoglobin',
    'packed_cell_volume','white_blood_cell_count',
    'red_blood_cell_count','eGFR'
]

# Clinical columns used during Standard Scaling
clinical_cols = [
    'blood_pressure','blood_glucose_random','blood_urea',
    'serum_creatinine','sodium','potassium','haemoglobin',
    'packed_cell_volume','white_blood_cell_count',
    'red_blood_cell_count'
]

clinical_cols = [c for c in clinical_cols if c in df.columns]

# Normal ranges mapped EXACTLY to dataset columns
normal_ranges = {
    'blood_pressure': (80, 120),
    'blood_glucose_random': (70, 140),
    'blood_urea': (7, 20),
    'serum_creatinine': (0.6, 1.3),
    'sodium': (135, 145),
    'potassium': (3.5, 5.0),
    'haemoglobin': (12, 17),
    'packed_cell_volume': (36, 50),
    'white_blood_cell_count': (4000, 11000),
    'red_blood_cell_count': (4.5, 5.9)
}

# ---------------------- Reverse Standard Scaling ---------------------
st.markdown("### 🔄 Reverse Standard Scaling for Clinical Interpretation")

scaler = joblib.load("/Users/nebula/Desktop/ME/interview/Clysys/data/clinical_scaler.pkl")
#scaler = StandardScaler()
#scaler.fit(df[clinical_cols])

df_unscaled = df.copy()
df_unscaled[scaler_cols] = scaler.inverse_transform(df[scaler_cols])

# ---------------------- Patient Selection ---------------------

c1, c2 = st.columns([1, 3])

with c1:
    selected_patient = st.number_input(
        "Select Patient Index",
        min_value=0,
        max_value=len(df_unscaled)-1,
        value=0,
        step=1
    )

patient_data = df_unscaled.loc[selected_patient, clinical_cols]

plot_df = pd.DataFrame({
    'Feature': clinical_cols,
    'Patient Value': patient_data.values,
    'Low Normal': [normal_ranges[c][0] for c in clinical_cols],
    'High Normal': [normal_ranges[c][1] for c in clinical_cols]
})

# ---------------------- Clinical Profile Bar Chart ---------------------

fig_profile = px.bar(
    plot_df,
    x='Feature',
    y='Patient Value',
    title=f"Patient {selected_patient} – Clinical Profile vs Normal Ranges",
    template="plotly_white"
)

fig_profile.add_scatter(
    x=plot_df['Feature'],
    y=plot_df['Low Normal'],
    mode='lines+markers',
    name='Low Normal',
    line=dict(dash='dash')
)

fig_profile.add_scatter(
    x=plot_df['Feature'],
    y=plot_df['High Normal'],
    mode='lines+markers',
    name='High Normal',
    line=dict(dash='dash')
)

fig_profile.update_layout(
    xaxis_title="Clinical Parameters",
    yaxis_title="Measurement Value",
    legend_title="Reference Range"
)

st.plotly_chart(fig_profile, use_container_width=True)

# ---------------------- Abnormality Summary ---------------------

dev_df = plot_df.copy()

dev_df['Status'] = np.where(
    dev_df['Patient Value'] < dev_df['Low Normal'], 'Low',
    np.where(dev_df['Patient Value'] > dev_df['High Normal'], 'High', 'Normal')
)

st.markdown("### 🚦 Parameter Deviation Summary")
st.dataframe(
    dev_df[['Feature', 'Patient Value', 'Low Normal', 'High Normal', 'Status']],
    use_container_width=True
)

# ---------------------- Radar Profile ---------------------

radar_df = plot_df.copy()
radar_df['Normalized'] = (radar_df['Patient Value'] - radar_df['Low Normal']) / \
                         (radar_df['High Normal'] - radar_df['Low Normal'])

fig_radar = px.line_polar(
    radar_df,
    r='Normalized',
    theta='Feature',
    line_close=True,
    title="Normalized Clinical Radar Profile (0 = Low Normal, 1 = High Normal)",
    template="plotly_white"
)

fig_radar.update_traces(fill='toself')
st.plotly_chart(fig_radar, use_container_width=True)

# ---------------------- Disease Probability & Risk Scoring ---------------------
st.subheader("🧠 Disease Probability & Risk Stratification")

# Use weighted severity index if available, else fallback to abnormality score
if 'weighted_severity_index' in df_unscaled.columns:
    raw_score = df_unscaled.loc[selected_patient, 'weighted_severity_index']
    score_min = df_unscaled['weighted_severity_index'].min()
    score_max = df_unscaled['weighted_severity_index'].max()
    probability = (raw_score - score_min) / (score_max - score_min + 1e-9)

elif 'abnormality_count_score' in df_unscaled.columns:
    raw_score = df_unscaled.loc[selected_patient, 'abnormality_count_score']
    score_min = df_unscaled['abnormality_count_score'].min()
    score_max = df_unscaled['abnormality_count_score'].max()
    probability = (raw_score - score_min) / (score_max - score_min + 1e-9)

else:
    probability = 0.0

probability = float(np.clip(probability, 0, 1))

# Risk Category
if probability < 0.30:
    risk_category = "Low Risk"
    risk_color = "🟢"
elif probability < 0.60:
    risk_category = "Moderate Risk"
    risk_color = "🟡"
elif probability < 0.80:
    risk_category = "High Risk"
    risk_color = "🟠"
else:
    risk_category = "Very High Risk"
    risk_color = "🔴"

c1, c2, c3 = st.columns(3)
c1.metric("Disease Probability", f"{probability*100:.2f}%")
c2.metric("Risk Category", f"{risk_color} {risk_category}")
c3.metric("Raw Severity Score", f"{raw_score:.3f}" if 'raw_score' in locals() else "N/A")

# Probability Gauge
fig_prob = px.bar(
    x=["CKD Probability"],
    y=[probability*100],
    range_y=[0, 100],
    labels={'y': 'Probability (%)', 'x': ''},
    title="Estimated Disease Probability",
    template="plotly_white"
)
fig_prob.update_traces(marker_color='crimson')
st.plotly_chart(fig_prob, use_container_width=True)

# ---------------------- Key Biomarkers Driving Predictions ---------------------
st.subheader("🔬 Key Biomarkers Driving Predictions (Color-coded by Impact)")

# Encode target if categorical (e.g., 'ckd', 'notckd')
if df[TARGET].dtype == 'object':
    target_encoded = df[TARGET].map({
        df[TARGET].unique()[0]: 0,
        df[TARGET].unique()[1]: 1
    })
else:
    target_encoded = df[TARGET]

# Select biomarkers available
biomarker_cols = clinical_cols.copy()

# Compute correlation-based impact scores (robust to NaNs & non-numeric issues)
impact_scores = []

for b in biomarker_cols:
    if pd.api.types.is_numeric_dtype(df[b]):
        corr_val = df[b].corr(target_encoded)
        impact_scores.append(abs(corr_val) if not pd.isna(corr_val) else 0)
    else:
        impact_scores.append(0)

impact_df = pd.DataFrame({
    'Biomarker': biomarker_cols,
    'Impact': impact_scores
}).sort_values(by='Impact', ascending=False)

# Normalize for visualization
impact_df['Impact_norm'] = impact_df['Impact'] / (impact_df['Impact'].max() + 1e-9)

fig_imp = px.bar(
    impact_df,
    x='Impact_norm',
    y='Biomarker',
    orientation='h',
    color='Impact_norm',
    color_continuous_scale='RdYlGn_r',
    title="Relative Contribution of Key Biomarkers",
    labels={'Impact_norm': 'Normalized Impact Score'}
)

fig_imp.update_layout(yaxis=dict(autorange="reversed"))
st.plotly_chart(fig_imp, use_container_width=True)

st.caption("Impact computed using absolute Pearson correlation with encoded target variable (model-agnostic explainability proxy).")

# ---------------------- Biomarker Interaction Patterns ---------------------
st.subheader("🧩 Biomarker Interaction Patterns")

st.markdown("""
Understanding **interactions between biomarkers** helps reveal **non-linear disease mechanisms**
and synergistic effects that single-variable analysis may miss.
""")

# Select interaction method
interaction_mode = st.selectbox(
    "Select Interaction Visualization",
    ["Pairwise Scatter Matrix", "Top Interaction Heatmap", "Clinical Interaction Plots"]
)

# ------------------ Pairwise Scatter Matrix ------------------
if interaction_mode == "Pairwise Scatter Matrix":
    sel_feats = st.multiselect(
        "Select biomarkers (3–6 recommended)",
        clinical_cols,
        default=clinical_cols[:4]
    )

    if len(sel_feats) >= 2:
        fig = px.scatter_matrix(
            df_unscaled,
            dimensions=sel_feats,
            color=TARGET,
            title="Pairwise Biomarker Interaction Matrix",
            template="plotly_white"
        )
        fig.update_traces(diagonal_visible=False)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Select at least two biomarkers.")

# ------------------ Top Interaction Heatmap ------------------
elif interaction_mode == "Top Interaction Heatmap":

    # Compute absolute correlations among biomarkers
    corr_matrix = df_unscaled[clinical_cols].corr().abs()

    # Mask diagonal
    np.fill_diagonal(corr_matrix.values, 0)

    # Extract top interacting pairs
    top_pairs = (
        corr_matrix.unstack()
        .sort_values(ascending=False)
        .drop_duplicates()
        .head(15)
    )

    heat_df = corr_matrix.loc[top_pairs.index.get_level_values(0),
                              top_pairs.index.get_level_values(1)]

    fig = px.imshow(
        corr_matrix,
        text_auto=False,
        aspect="auto",
        title="Biomarker Interaction Strength Heatmap",
        template="plotly_white",
        color_continuous_scale="RdBu_r"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 🔝 Strongest Biomarker Interactions")
    st.dataframe(top_pairs.reset_index().rename(
        columns={'level_0':'Biomarker 1', 'level_1':'Biomarker 2', 0:'Interaction Strength'}
    ), use_container_width=True)

# ------------------ Clinical Interaction Plots ------------------
elif interaction_mode == "Clinical Interaction Plots":

    col1, col2 = st.columns(2)

    with col1:
        x_feat = st.selectbox("Select X Biomarker", clinical_cols)
    with col2:
        y_feat = st.selectbox("Select Y Biomarker", clinical_cols, index=1)

    fig = px.scatter(
        df_unscaled,
        x=x_feat,
        y=y_feat,
        color=TARGET,
        trendline="ols",
        title=f"{x_feat} vs {y_feat} — Interaction Pattern",
        template="plotly_white"
    )

    st.plotly_chart(fig, use_container_width=True)


# ---------------------- Download Clean Data ---------------
st.subheader("📥 Export Dataset")

csv = df.to_csv(index=False).encode('utf-8')
st.download_button(
    "Download Processed Dataset",
    csv,
    "ckd_cleaned_dataset.csv",
    "text/csv"
)

# ---------------------- Footer -----------------------------
st.markdown("""
---
📌 **Designed for academic publication, research reports, and ML explainability.**  
Developed with **Streamlit + Plotly + Seaborn** for interactive and static visualization.
""")
