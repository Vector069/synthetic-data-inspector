import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis, normaltest
import seaborn as sns
import matplotlib.pyplot as plt
import io
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, Image, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

st.title("Dataset Real vs Synthetic Inspector (Side-by-Side Comparison)")

st.markdown("""
Upload two datasets (CSV): **Real** and **Synthetic**.  
The app analyzes **numerical columns**, displays statistics, visualizations, and concludes whether each dataset appears real or synthetic.
""")

# File uploads
real_file = st.file_uploader("Upload Real Dataset", type=["csv"], key="real")
synthetic_file = st.file_uploader("Upload Synthetic Dataset", type=["csv"], key="synthetic")


# --- Analysis functions ---
def analyze_numeric_columns(df):
    numeric_df = df.select_dtypes(include=[np.number])
    analysis = {}

    analysis['skewness'] = numeric_df.apply(skew)
    analysis['kurtosis'] = numeric_df.apply(kurtosis)

    normality_pvals = {}
    for col in numeric_df.columns:
        stat, p = normaltest(numeric_df[col])
        normality_pvals[col] = p
    analysis['normality_pval'] = pd.Series(normality_pvals)

    analysis_table = pd.DataFrame({
        'Skewness': analysis['skewness'],
        'Kurtosis': analysis['kurtosis'],
        'Normality p-value': analysis['normality_pval']
    })

    analysis['summary'] = numeric_df.describe().T
    analysis['correlation'] = numeric_df.corr()
    analysis['covariance'] = numeric_df.cov()

    return numeric_df, analysis, analysis_table


def infer_real_or_synthetic(analysis):
    reasoning = []
    score = 0

    skew_abs = analysis['skewness'].abs()
    if (skew_abs < 0.5).all():
        reasoning.append("All numeric features have very low skewness; may indicate synthetic data.")
        score += 1
    else:
        reasoning.append("Numeric features have natural skewness; likely real data.")
        score -= 1

    kurt_abs = analysis['kurtosis'].abs()
    if (kurt_abs < 1).all():
        reasoning.append("All numeric features have low kurtosis; may indicate synthetic data.")
        score += 1
    else:
        reasoning.append("Numeric features have varied kurtosis; likely real data.")
        score -= 1

    normal_pvals = analysis['normality_pval']
    if (normal_pvals > 0.05).all():
        reasoning.append("All numeric f
