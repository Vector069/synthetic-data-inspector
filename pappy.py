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
    # ---------------- PDF GENERATION ----------------
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=20, leftMargin=20)
    elements = []
    styles = getSampleStyleSheet()
    normal = styles["Normal"]

    # Helper: dataframe -> nice table
    from reportlab.platypus import TableStyle
    def df_to_table(df, max_rows=10):
        df = df.head(max_rows).reset_index()
        data = [df.columns.tolist()] + df.values.tolist()
        data = [[Paragraph(str(cell), normal) for cell in row] for row in data]
        col_width = doc.width / len(df.columns)
        t = Table(data, colWidths=[col_width] * len(df.columns))
        t.setStyle(TableStyle([
            ("GRID", (0,0), (-1,-1), 0.25, colors.black),
            ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
            ("FONTSIZE", (0,0), (-1,-1), 7),
            ("VALIGN", (0,0), (-1,-1), "TOP")
        ]))
        return t

    # Helper: save plot as image
    def save_plot_to_bytes(fig):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        return buf

    # Helper: plots side by side
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

    # --- Title
    elements.append(Paragraph("Dataset Real vs Synthetic Inspector Report", styles["Title"]))
    elements.append(Spacer(1, 20))

    # --- Skewness, Kurtosis & Normality
    elements.append(Paragraph("Skewness, Kurtosis & Normality Test", styles["Heading2"]))
    elements.append(Spacer(1, 8))
    elements.append(Paragraph("<b>Real Dataset</b>", styles["Heading3"]))
    elements.append(df_to_table(table_real))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph("<b>Synthetic Dataset</b>", styles["Heading3"]))
    elements.append(df_to_table(table_synth))
    elements.append(Spacer(1, 20))

    # --- Summary Statistics
    elements.append(Paragraph("Summary Statistics", styles["Heading2"]))
    elements.append(Spacer(1, 8))
    elements.append(Paragraph("<b>Real Dataset</b>", styles["Heading3"]))
    elements.append(df_to_table(analysis_real['summary']))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph("<b>Synthetic Dataset</b>", styles["Heading3"]))
    elements.append(df_to_table(analysis_synth['summary']))
    elements.append(Spacer(1, 20))

    # --- Visualizations
    elements.append(Paragraph("Visualizations", styles["Heading2"]))
    elements.append(Spacer(1, 12))

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

    # Skewness
    fig_real, ax = plt.subplots(figsize=(6,4))
    analysis_real['skewness'].plot(kind='bar', color='skyblue', ax=ax)
    ax.set_title("Skewness per Feature (Real)")
    fig_synth, ax = plt.subplots(figsize=(6,4))
    analysis_synth['skewness'].plot(kind='bar', color='skyblue', ax=ax)
    ax.set_title("Skewness per Feature (Synthetic)")
    add_side_by_side(fig_real, "Skewness per Feature (Real)",
                     fig_synth, "Skewness per Feature (Synthetic)")

    # Kurtosis
    fig_real, ax = plt.subplots(figsize=(6,4))
    analysis_real['kurtosis'].plot(kind='bar', color='salmon', ax=ax)
    ax.set_title("Kurtosis per Feature (Real)")
    fig_synth, ax = plt.subplots(figsize=(6,4))
    analysis_synth['kurtosis'].plot(kind='bar', color='salmon', ax=ax)
    ax.set_title("Kurtosis per Feature (Synthetic)")
    add_side_by_side(fig_real, "Kurtosis per Feature (Real)",
                     fig_synth, "Kurtosis per Feature (Synthetic)")

    # Correlation
    fig_real, ax = plt.subplots(figsize=(6,4))
    sns.heatmap(analysis_real['correlation'], annot=True, cmap='coolwarm', ax=ax)
    ax.set_title("Correlation Heatmap (Real)")
    fig_synth, ax = plt.subplots(figsize=(6,4))
    sns.heatmap(analysis_synth['correlation'], annot=True, cmap='coolwarm', ax=ax)
    ax.set_title("Correlation Heatmap (Synthetic)")
    add_side_by_side(fig_real, "Correlation Heatmap (Real)",
                     fig_synth, "Correlation Heatmap (Synthetic)")

    # Covariance
    fig_real, ax = plt.subplots(figsize=(6,4))
    sns.heatmap(analysis_real['covariance'], annot=True, cmap='viridis', ax=ax)
    ax.set_title("Covariance Heatmap (Real)")
    fig_synth, ax = plt.subplots(figsize=(6,4))
    sns.heatmap(analysis_synth['covariance'], annot=True, cmap='viridis', ax=ax)
    ax.set_title("Covariance Heatmap (Synthetic)")
    add_side_by_side(fig_real, "Covariance Heatmap (Real)",
                     fig_synth, "Covariance Heatmap (Synthetic)")

    # --- Inference & Conclusion
    elements.append(Paragraph("Inference and Conclusion", styles["Heading2"]))
    elements.append(Spacer(1, 8))

    reasoning_real_text = "".join([f"• {r}<br/>" for r in reasoning_real]) + f"<br/><b>{conclusion_real}</b>"
    elements.append(Paragraph("<b>Real Dataset</b>", styles["Heading3"]))
    elements.append(Paragraph(reasoning_real_text, normal))
    elements.append(Spacer(1, 12))

    reasoning_synth_text = "".join([f"• {r}<br/>" for r in reasoning_synth]) + f"<br/><b>{conclusion_synth}</b>"
    elements.append(Paragraph("<b>Synthetic Dataset</b>", styles["Heading3"]))
    elements.append(Paragraph(reasoning_synth_text, normal))

    # Build PDF
    doc.build(elements)
    buffer.seek(0)

    # --- Floating Download Button ---
    st.markdown("""
        <style>
        .fixed-download {
            position: fixed;
            top: 20px;
            right: 20px;
            z-index: 100;
            background-color: white;
            padding: 10px;
            border-radius: 10px;
            box-shadow: 0px 2px 8px rgba(0,0,0,0.2);
        }
        </style>
    """, unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="fixed-download">', unsafe_allow_html=True)
        st.download_button(
            label="Download Full Report (PDF)",
            data=buffer,
            file_name="dataset_report.pdf",
            mime="application/pdf"
        )
        st.markdown('</div>', unsafe_allow_html=True)
