import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis, normaltest
import seaborn as sns
import matplotlib.pyplot as plt
import io
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors


st.title("Dataset Real vs Synthetic Inspector (Side-by-Side Comparison)")

st.markdown("""
Upload two datasets (CSV): **Real** and **Synthetic**.  
The app analyzes **numerical columns**, displays statistics, visualizations, and concludes whether each dataset appears real or synthetic.
""")

# Two file uploads
real_file = st.file_uploader("Upload Real Dataset", type=["csv"], key="real")
synthetic_file = st.file_uploader("Upload Synthetic Dataset", type=["csv"], key="synthetic")


def analyze_numeric_columns(df):
    numeric_df = df.select_dtypes(include=[np.number])
    analysis = {}
    
    # Shape statistics: skewness and kurtosis
    analysis['skewness'] = numeric_df.apply(skew)
    analysis['kurtosis'] = numeric_df.apply(kurtosis)
    
    # Normality test (p-value)
    normality_pvals = {}
    for col in numeric_df.columns:
        stat, p = normaltest(numeric_df[col])
        normality_pvals[col] = p
    analysis['normality_pval'] = pd.Series(normality_pvals)
    
    # Combine skewness, kurtosis, normality
    analysis_table = pd.DataFrame({
        'Skewness': analysis['skewness'],
        'Kurtosis': analysis['kurtosis'],
        'Normality p-value': analysis['normality_pval']
    })
    
    # Summary statistics
    analysis['summary'] = numeric_df.describe().T
    
    # Correlation
    analysis['correlation'] = numeric_df.corr()
    
    # Covariance
    analysis['covariance'] = numeric_df.cov()
    
    return numeric_df, analysis, analysis_table


def infer_real_or_synthetic(analysis):
    reasoning = []
    score = 0  # positive score favors synthetic, negative favors real
    
    # Skewness
    skew_abs = analysis['skewness'].abs()
    if (skew_abs < 0.5).all():
        reasoning.append("All numeric features have very low skewness; may indicate synthetic data.")
        score += 1
    else:
        reasoning.append("Numeric features have natural skewness; likely real data.")
        score -= 1
    
    # Kurtosis
    kurt_abs = analysis['kurtosis'].abs()
    if (kurt_abs < 1).all():
        reasoning.append("All numeric features have low kurtosis; may indicate synthetic data.")
        score += 1
    else:
        reasoning.append("Numeric features have varied kurtosis; likely real data.")
        score -= 1
    
    # Normality
    normal_pvals = analysis['normality_pval']
    if (normal_pvals > 0.05).all():
        reasoning.append("All numeric features pass normality test perfectly; could be synthetic.")
        score += 1
    else:
        reasoning.append("Numeric features show varied normality; likely real data.")
        score -= 1
    
    # Correlation
    corr_std = analysis['correlation'].values.std()
    if corr_std < 0.1:
        reasoning.append("Correlations are extremely uniform; likely synthetic.")
        score += 1
    else:
        reasoning.append("Correlations vary naturally; likely real data.")
        score -= 1
    
    # Covariance
    cov_std = analysis['covariance'].values.std()
    if cov_std < 0.1:
        reasoning.append("Covariances are extremely uniform; likely synthetic.")
        score += 1
    else:
        reasoning.append("Covariances vary naturally; likely real data.")
        score -= 1
    
    # Variability
    std_mean = analysis['summary']['std'].mean()
    if std_mean < 1e-3:
        reasoning.append("Extremely low variability across numeric features; likely synthetic.")
        score += 1
    else:
        reasoning.append("Numeric feature variability seems realistic; likely real data.")
        score -= 1
    
    # Final conclusion
    if score >= 2:
        conclusion = "Overall Conclusion: Dataset is likely **synthetic**."
    elif score <= -2:
        conclusion = "Overall Conclusion: Dataset is likely **real**."
    else:
        conclusion = "Overall Conclusion: Dataset is **uncertain/mixed**; some indicators suggest synthetic, some real."
    
    return reasoning, conclusion


def save_plot_to_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    return buf


if real_file and synthetic_file:
    df_real = pd.read_csv(real_file)
    df_synth = pd.read_csv(synthetic_file)
    
    # Analyze both
    numeric_real, analysis_real, table_real = analyze_numeric_columns(df_real)
    numeric_synth, analysis_synth, table_synth = analyze_numeric_columns(df_synth)
    
    # Inference
    reasoning_real, conclusion_real = infer_real_or_synthetic(analysis_real)
    reasoning_synth, conclusion_synth = infer_real_or_synthetic(analysis_synth)
    
    # ---------------- PDF GENERATION ----------------
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=20, leftMargin=20)
    elements = []
    styles = getSampleStyleSheet()

    # Title
    elements.append(Paragraph("Dataset Real vs Synthetic Inspector Report", styles["Title"]))
    elements.append(Spacer(1, 20))

    # --- Skewness, Kurtosis & Normality Test (stacked)
    elements.append(Paragraph("Skewness, Kurtosis & Normality Test", styles["Heading2"]))
    elements.append(Spacer(1, 10))

    # Real
    elements.append(Paragraph("<b>Real Dataset</b>", styles["Heading3"]))
    table_data_real = [table_real.reset_index().columns.tolist()] + table_real.reset_index().values.tolist()
    t_real = Table(table_data_real,
                   style=[("GRID", (0,0), (-1,-1), 0.5, colors.black),
                          ("BACKGROUND", (0,0), (-1,0), colors.lightgrey)],
                   colWidths=80)
    elements.append(t_real)
    elements.append(Spacer(1, 15))

    # Synthetic
    elements.append(Paragraph("<b>Synthetic Dataset</b>", styles["Heading3"]))
    table_data_synth = [table_synth.reset_index().columns.tolist()] + table_synth.reset_index().values.tolist()
    t_synth = Table(table_data_synth,
                    style=[("GRID", (0,0), (-1,-1), 0.5, colors.black),
                           ("BACKGROUND", (0,0), (-1,0), colors.lightgrey)],
                    colWidths=80)
    elements.append(t_synth)
    elements.append(Spacer(1, 20))

    # --- Helper to add plots side by side
    def add_side_by_side(fig_real, title_real, fig_synth, title_synth):
        row = []
        for fig, title in [(fig_real, title_real), (fig_synth, title_synth)]:
            if fig:
                buf_img = save_plot_to_bytes(fig)
                cell = [Paragraph(title, styles["Heading3"]),
                        Image(buf_img, width=250, height=200)]
                row.append(cell)
                plt.close(fig)
        if row:
            t = Table([row], colWidths=[270, 270])
            elements.append(t)
            elements.append(Spacer(1, 20))

    # --- Histograms & Boxplots
    for col_name in numeric_real.columns:
        # Histogram
        fig_real, ax = plt.subplots()
        sns.histplot(numeric_real[col_name], kde=True, ax=ax, color='skyblue')
        ax.set_title(f"{col_name} Histogram (Real)")
        fig_synth = None
        if col_name in numeric_synth.columns:
            fig_synth, ax = plt.subplots()
            sns.histplot(numeric_synth[col_name], kde=True, ax=ax, color='skyblue')
            ax.set_title(f"{col_name} Histogram (Synthetic)")
        add_side_by_side(fig_real, f"{col_name} Histogram (Real)",
                         fig_synth, f"{col_name} Histogram (Synthetic)")

        # Boxplot
        fig_real, ax = plt.subplots()
        sns.boxplot(x=numeric_real[col_name], ax=ax, color='salmon')
        ax.set_title(f"{col_name} Boxplot (Real)")
        fig_synth = None
        if col_name in numeric_synth.columns:
            fig_synth, ax = plt.subplots()
            sns.boxplot(x=numeric_synth[col_name], ax=ax, color='salmon')
            ax.set_title(f"{col_name} Boxplot (Synthetic)")
        add_side_by_side(fig_real, f"{col_name} Boxplot (Real)",
                         fig_synth, f"{col_name} Boxplot (Synthetic)")

    # --- Skewness charts
    fig_real, ax = plt.subplots(figsize=(6,4))
    analysis_real['skewness'].plot(kind='bar', color='skyblue', ax=ax)
    ax.set_title("Skewness per Feature (Real)")
    fig_synth, ax = plt.subplots(figsize=(6,4))
    analysis_synth['skewness'].plot(kind='bar', color='skyblue', ax=ax)
    ax.set_title("Skewness per Feature (Synthetic)")
    add_side_by_side(fig_real, "Skewness per Feature (Real)",
                     fig_synth, "Skewness per Feature (Synthetic)")

    # --- Kurtosis charts
    fig_real, ax = plt.subplots(figsize=(6,4))
    analysis_real['kurtosis'].plot(kind='bar', color='salmon', ax=ax)
    ax.set_title("Kurtosis per Feature (Real)")
    fig_synth, ax = plt.subplots(figsize=(6,4))
    analysis_synth['kurtosis'].plot(kind='bar', color='salmon', ax=ax)
    ax.set_title("Kurtosis per Feature (Synthetic)")
    add_side_by_side(fig_real, "Kurtosis per Feature (Real)",
                     fig_synth, "Kurtosis per Feature (Synthetic)")

    # --- Correlation heatmaps
    fig_real, ax = plt.subplots(figsize=(6,4))
    sns.heatmap(analysis_real['correlation'], annot=True, cmap='coolwarm', ax=ax)
    ax.set_title("Correlation Heatmap (Real)")
    fig_synth, ax = plt.subplots(figsize=(6,4))
    sns.heatmap(analysis_synth['correlation'], annot=True, cmap='coolwarm', ax=ax)
    ax.s
