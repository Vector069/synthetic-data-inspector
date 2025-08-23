import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis, normaltest
import seaborn as sns
import matplotlib.pyplot as plt

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


# Ensure both datasets are uploaded
if real_file and synthetic_file:
    df_real = pd.read_csv(real_file)
    df_synth = pd.read_csv(synthetic_file)
    
    # Analyze both
    numeric_real, analysis_real, table_real = analyze_numeric_columns(df_real)
    numeric_synth, analysis_synth, table_synth = analyze_numeric_columns(df_synth)
    
    # Side-by-side display for preview and statistics
    st.subheader("Numeric Columns Preview")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Real Dataset Preview**")
        st.dataframe(numeric_real.head())
    with col2:
        st.markdown("**Synthetic Dataset Preview**")
        st.dataframe(numeric_synth.head())
    
    st.subheader("Skewness, Kurtosis & Normality Test")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Real Dataset**")
        st.dataframe(table_real.style.format({
            'Skewness': "{:.3f}",
            'Kurtosis': "{:.3f}",
            'Normality p-value': "{:.4f}"
        }))
    with col2:
        st.markdown("**Synthetic Dataset**")
        st.dataframe(table_synth.style.format({
            'Skewness': "{:.3f}",
            'Kurtosis': "{:.3f}",
            'Normality p-value': "{:.4f}"
        }))
    
    st.subheader("Summary Statistics")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Real Dataset**")
        st.dataframe(analysis_real['summary'])
    with col2:
        st.markdown("**Synthetic Dataset**")
        st.dataframe(analysis_synth['summary'])
    
    st.subheader("Visualizations")
    
    # Histograms & Boxplots side by side
    for col_name in numeric_real.columns:
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots()
            sns.histplot(numeric_real[col_name], kde=True, ax=ax, color='skyblue')
            ax.set_title(f"{col_name} Histogram (Real)")
            st.pyplot(fig)
            
            fig, ax = plt.subplots()
            sns.boxplot(x=numeric_real[col_name], ax=ax, color='salmon')
            ax.set_title(f"{col_name} Boxplot (Real)")
            st.pyplot(fig)
        with col2:
            if col_name in numeric_synth.columns:
                fig, ax = plt.subplots()
                sns.histplot(numeric_synth[col_name], kde=True, ax=ax, color='skyblue')
                ax.set_title(f"{col_name} Histogram (Synthetic)")
                st.pyplot(fig)
                
                fig, ax = plt.subplots()
                sns.boxplot(x=numeric_synth[col_name], ax=ax, color='salmon')
                ax.set_title(f"{col_name} Boxplot (Synthetic)")
                st.pyplot(fig)
    
    # Skewness bar charts side by side
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(8,4))
        analysis_real['skewness'].plot(kind='bar', color='skyblue', ax=ax)
        ax.set_title("Skewness per Feature (Real)")
        st.pyplot(fig)
    with col2:
        fig, ax = plt.subplots(figsize=(8,4))
        analysis_synth['skewness'].plot(kind='bar', color='skyblue', ax=ax)
        ax.set_title("Skewness per Feature (Synthetic)")
        st.pyplot(fig)
    
    # Kurtosis bar charts side by side
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(8,4))
        analysis_real['kurtosis'].plot(kind='bar', color='salmon', ax=ax)
        ax.set_title("Kurtosis per Feature (Real)")
        st.pyplot(fig)
    with col2:
        fig, ax = plt.subplots(figsize=(8,4))
        analysis_synth['kurtosis'].plot(kind='bar', color='salmon', ax=ax)
        ax.set_title("Kurtosis per Feature (Synthetic)")
        st.pyplot(fig)
    
    # Correlation heatmaps side by side
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(8,6))
        sns.heatmap(analysis_real['correlation'], annot=True, cmap='coolwarm', ax=ax)
        ax.set_title("Correlation Heatmap (Real)")
        st.pyplot(fig)
    with col2:
        fig, ax = plt.subplots(figsize=(8,6))
        sns.heatmap(analysis_synth['correlation'], annot=True, cmap='coolwarm', ax=ax)
        ax.set_title("Correlation Heatmap (Synthetic)")
        st.pyplot(fig)
    
    # Covariance heatmaps side by side
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(8,6))
        sns.heatmap(analysis_real['covariance'], annot=True, cmap='viridis', ax=ax)
        ax.set_title("Covariance Heatmap (Real)")
        st.pyplot(fig)
    with col2:
        fig, ax = plt.subplots(figsize=(8,6))
        sns.heatmap(analysis_synth['covariance'], annot=True, cmap='viridis', ax=ax)
        ax.set_title("Covariance Heatmap (Synthetic)")
        st.pyplot(fig)
    
    # Inference side by side
    reasoning_real, conclusion_real = infer_real_or_synthetic(analysis_real)
    reasoning_synth, conclusion_synth = infer_real_or_synthetic(analysis_synth)
    
    st.subheader("Inference and Conclusion")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Real Dataset**")
        for r in reasoning_real:
            st.write(f"- {r}")
        st.markdown(f"### {conclusion_real}")
    with col2:
        st.markdown("**Synthetic Dataset**")
        for r in reasoning_synth:
            st.write(f"- {r}")
        st.markdown(f"### {conclusion_synth}")
