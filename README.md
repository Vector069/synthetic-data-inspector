# Dataset Real vs Synthetic Inspector

A **Streamlit** application to compare and analyze **real** vs **synthetic** datasets side by side.
The tool evaluates **statistical shape**, **correlation**, **covariance**, and **distribution patterns** of numerical features, then provides an automated **inference** on whether the dataset is more likely to be *real* or *synthetic*.

---

## Features

* **Upload & Preview**

  * Upload two CSV files: one real dataset and one synthetic dataset.
  * Side-by-side preview of numeric features.

* **Statistical Analysis**

  * Skewness, Kurtosis, and Normality test (D’Agostino and Pearson).
  * Summary statistics (mean, std, min, max, etc.).

* **Visualizations**

  * Histograms & Boxplots (distribution comparison).
  * Skewness and Kurtosis bar charts.
  * Correlation heatmaps.
  * Covariance heatmaps.

* **Automated Inference**

  * Rule-based scoring system evaluates skewness, kurtosis, normality, correlation, covariance, and variability.
  * Provides reasoning for each indicator.
  * Concludes whether the dataset appears **Real**, **Synthetic**, or **Uncertain**.

---

1. Upload your **Real Dataset** CSV.
2. Upload your **Synthetic Dataset** CSV.
3. Inspect:

   * Statistical tables side by side.
   * Distribution plots and heatmaps.
   * Automated inference and conclusion.

---

## Example Output

* **Real Dataset Inference**

  * Numeric features show varied skewness and kurtosis.
  * Correlations and covariances vary naturally.
  * Conclusion: Dataset is likely **Real**.

* **Synthetic Dataset Inference**

  * Features have uniform skewness and kurtosis.
  * Correlations and covariances are too consistent.
  * Conclusion: Dataset is likely **Synthetic**.

---

## Notes

* Focuses only on **numeric columns** to avoid errors.
* Works best when real and synthetic datasets have the same schema.
* Designed for quick **EDA + inference** of dataset authenticity.
