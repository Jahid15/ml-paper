# Electricity Demand Forecasting Using Temporal Fusion Transformer and Prophet Models

## A Comparative Study on Bangladesh Power Grid Data (2016-2024)

> **Term Paper** — Machine Learning Course (20 Marks)

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [What is TFT?](#2-what-is-tft-temporal-fusion-transformer)
3. [What is Facebook Prophet?](#3-what-is-facebook-prophet)
4. [Why This Problem?](#4-why-this-problem)
5. [Why Compare TFT vs Prophet?](#5-why-compare-tft-vs-prophet)
6. [Dataset Description](#6-dataset-description)
7. [Methodology](#7-methodology)
8. [How to Reproduce](#8-how-to-reproduce)
9. [Code Structure](#9-code-structure)
10. [Results Summary](#10-results-summary)
11. [Literature Review](#11-literature-review)
12. [Statistical Methods](#12-statistical-methods)
13. [FAQ / Potential Viva Questions](#13-faq--potential-viva-questions)
14. [References](#14-references)

---

## 1. Project Overview

### What?
We compare two forecasting models for predicting Bangladesh's hourly electricity demand:
- **Temporal Fusion Transformer (TFT)** — A state-of-the-art deep learning architecture
- **Facebook Prophet** — A robust statistical decomposition model

### Why?
Accurate electricity demand forecasting is critical for:
- Power generation scheduling
- Grid stability maintenance
- Infrastructure investment planning
- Reducing energy waste and costs

### How?
- **Dataset**: 78,912 hourly records from Bangladesh (2016-2024) with weather covariates
- **Platform**: Kaggle (free GPU)
- **Libraries**: `darts` (TFT), `prophet` (Prophet), `pandas`, `numpy`, `matplotlib`, `scikit-learn`
- **Evaluation**: MAPE, MSE, RMSE, MAE, R² Score

---

## 2. What is TFT (Temporal Fusion Transformer)?

### Overview
The Temporal Fusion Transformer (TFT) was proposed by **Lim et al. (2021)** in the paper *"Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting"* published in the International Journal of Forecasting.

### Architecture Components

```
Input → Variable Selection → LSTM Encoder → Multi-Head Attention → Quantile Output
         Networks (VSN)      Decoder         Mechanism
```

#### 1. Variable Selection Networks (VSN)
- **Purpose**: Automatically select the most relevant input features
- **How**: Uses Softmax-weighted GRN (Gated Residual Network) blocks
- **Why important**: Not all features contribute equally; VSN learns which weather variables matter most for demand prediction

#### 2. Gated Residual Networks (GRN)
- **Purpose**: Flexible nonlinear processing with skip connections
- **Formula**: `GRN(a, c) = LayerNorm(a + GLU(W₁·η₁ + b₁))`
  where `η₁ = W₂·ELU(W₃·a + W₄·c + b₃) + b₂`
- **Why important**: Allows the network to learn complex relationships while maintaining training stability

#### 3. LSTM Encoder-Decoder
- **Purpose**: Capture local temporal patterns and dependencies
- **How**: LSTM processes sequences in both directions (encoder) and generates future predictions (decoder)
- **Why important**: LSTMs handle sequential dependencies better than feed-forward networks

#### 4. Multi-Head Attention
- **Purpose**: Capture long-range temporal dependencies
- **How**: Adapted from the Transformer architecture (Vaswani et al., 2017)
- **Formula**: `Attention(Q,K,V) = softmax(QKᵀ/√dₖ)V`
- **Why important**: Can relate demand at hour `t` to patterns weeks or months ago (e.g., similar weather conditions)

#### 5. Quantile Regression Output
- **Purpose**: Generate prediction intervals (not just point estimates)
- **How**: Outputs multiple quantiles (e.g., 10th, 50th, 90th percentiles)
- **Why important**: Gives confidence intervals for risk-aware decision making

### Key Hyperparameters Used
| Parameter | Value | Explanation |
|-----------|-------|-------------|
| `input_chunk_length` | 48 | Model looks at past 48 hours (2 days) |
| `output_chunk_length` | 24 | Model predicts next 24 hours (1 day) |
| `hidden_size` | 64 | Size of hidden layers |
| `lstm_layers` | 2 | Number of LSTM layers |
| `dropout` | 0.1 | 10% dropout for regularization |
| `batch_size` | 32 | Training batch size |
| `n_epochs` | 5 | Number of training passes |

---

## 3. What is Facebook Prophet?

### Overview
Prophet was developed by **Taylor and Letham (2018)** at Facebook (Meta) and published in *"Forecasting at Scale"* in The American Statistician.

### Mathematical Model
Prophet decomposes a time series into three additive components:

```
y(t) = g(t) + s(t) + h(t) + ε(t)
```

Where:
- **g(t)** = Trend function (linear or logistic growth)
- **s(t)** = Seasonality (Fourier series)
- **h(t)** = Holiday/regressor effects
- **ε(t)** = Error term (normally distributed)

### Components Explained

#### 1. Trend Component g(t)
- **Linear growth**: `g(t) = k·t + m` (used in our study)
- **Logistic growth**: `g(t) = C / (1 + exp(-k·(t-m)))` (for saturating growth)
- **Changepoints**: Automatic detection of points where the trend changes slope

#### 2. Seasonality Component s(t)
- Uses **Fourier series** to model periodic patterns:
  ```
  s(t) = Σ (aₙ·cos(2πnt/P) + bₙ·sin(2πnt/P))
  ```
- **Daily seasonality**: P = 1 day (captures hourly patterns)
- **Weekly seasonality**: P = 7 days (captures weekday/weekend patterns)
- **Yearly seasonality**: P = 365.25 days (captures seasonal patterns)

#### 3. Regressor Effects h(t)
- In our study, we add **weather regressors**:
  - Temperature (°C) — strongest predictor of cooling demand
  - Humidity (%) — affects perceived temperature
  - Surface Pressure (kPa) — atmospheric conditions

### Key Parameters Used
| Parameter | Value | Explanation |
|-----------|-------|-------------|
| `growth` | 'linear' | Linear trend model |
| `daily_seasonality` | True | Capture hour-level patterns |
| `weekly_seasonality` | True | Capture weekday/weekend |
| `yearly_seasonality` | True | Capture seasonal patterns |
| `changepoint_prior_scale` | 0.05 | Flexibility of trend changes |
| `seasonality_prior_scale` | 10.0 | Flexibility of seasonality |
| `interval_width` | 0.95 | 95% prediction interval |

---

## 4. Why This Problem?

### Bangladesh Energy Context
- Bangladesh has a population of ~170 million with rapidly growing electricity demand
- Peak demand has grown from ~5,000 MW (2009) to ~15,000+ MW (2024)
- **Load shedding** (planned power outages) remains a challenge
- Accurate forecasting helps:
  - **Grid operators**: Plan generation schedules
  - **Policy makers**: Plan infrastructure investments
  - **Consumers**: Manage electricity costs

### Weather-Demand Relationship
- Bangladesh has a **tropical climate** with hot, humid summers
- **Air conditioning** and cooling loads drive demand peaks
- Temperature is the strongest weather predictor of demand
- This creates strong nonlinear relationships that deep learning models can capture

---

## 5. Why Compare TFT vs Prophet?

| Aspect | TFT | Prophet |
|--------|-----|---------|
| **Type** | Deep Learning | Statistical |
| **Approach** | Learns patterns from data | Decomposes into components |
| **Interpretability** | Attention weights | Explicit trend/seasonality |
| **Complexity** | High (millions of parameters) | Low (few hundred parameters) |
| **Training** | Requires GPU | CPU sufficient |
| **Covariates** | Future covariates natively | Regressors added manually |
| **Uncertainty** | Quantile regression | Bayesian intervals |
| **Flexibility** | Very flexible | More structured |

**The comparison answers**: *When should a power utility use a deep learning model (expensive, complex, powerful) vs. a statistical model (simple, fast, interpretable)?*

---

## 6. Dataset Description

### Source
- **Kaggle**: `electricity-demand-bangladesh`
- **Period**: January 1, 2016 — December 31, 2024
- **Frequency**: Hourly (8,760-8,784 records per year)
- **Total**: ~78,912 records

### Raw Features
| Feature | Type | Range | Description |
|---------|------|-------|-------------|
| `Datetime` | Timestamp | 2016-01-01 to 2024-12-31 | Hourly timestamp |
| `Temperature(2m)` | Float | ~10-40°C | Air temperature at 2m |
| `Rhumadity(2m)` | Float | ~20-100% | Relative humidity at 2m |
| `SPressure(kPa)` | Float | ~99-103 kPa | Surface pressure |
| `Demand(MW)` | Float | ~3,000-16,000 MW | Electricity demand (TARGET) |

### Engineered Features
| Feature | Type | Description |
|---------|------|-------------|
| `hour` | Int (0-23) | Hour of day |
| `day_of_week` | Int (0-6) | 0=Monday |
| `day_of_month` | Int (1-31) | Day in month |
| `day_of_year` | Int (1-366) | Day in year |
| `month` | Int (1-12) | Month |
| `quarter` | Int (1-4) | Quarter |
| `year` | Int | Year |
| `week_of_year` | Int (1-53) | ISO week |
| `Lagged_col` | Float | Demand 4 hours ago |
| `Rolling_mean` | Float | 24-hour rolling average |

---

## 7. Methodology

### Pipeline Overview
```
Raw Data → Outlier Removal (IQR) → Feature Engineering → Normalization (StandardScaler)
    → Train/Test Split → TFT Training → TFT Prediction → Metrics
                       → Prophet Training → Prophet Prediction → Metrics
    → Comparative Analysis → Paper Figures & Tables
```

### Step 1: Outlier Removal (IQR Method)
- Compute Q1 (25th percentile) and Q3 (75th percentile)
- IQR = Q3 - Q1
- Lower bound = Q1 - 1.5 × IQR
- Upper bound = Q3 + 1.5 × IQR
- Replace outliers with column mean

### Step 2: Feature Engineering
- Extract 8 temporal features from datetime index
- Create 4-hour lag feature (captures recent demand patterns)
- Create 24-hour rolling mean (captures daily trends)
- Drop NaN rows from lag/rolling calculations

### Step 3: Normalization
- StandardScaler: z = (x - μ) / σ
- Fit on training data only
- Transform both train and test data
- Inverse transform predictions for evaluation

### Step 4: Train/Test Split
- **Train**: 2016-01-01 to 2024-11-30 (~8 years 11 months)
- **Test**: 2024-12-01 to 2024-12-31 (~744 hours)
- Ratio: ~99% train, ~1% test (typical for long time series)

### Step 5: Model Training & Prediction
- TFT: Uses `darts.models.TFTModel` with future covariates
- Prophet: Uses `prophet.Prophet` with weather regressors

### Step 6: Evaluation
- 5 metrics: MAPE, MSE, RMSE, MAE, R²
- Error distribution analysis
- Comparative visualization

---

## 8. How to Reproduce

### Step-by-Step on Google Colab (Recommended)

1. **Upload dataset** to Google Drive:
   - Create folder: `My Drive > machine-learning-paper`
   - Upload `BangladeshData_2016_2024.csv` into it
2. **Open** [colab.research.google.com](https://colab.research.google.com)
3. **Enable GPU**: Runtime → Change runtime type → **T4 GPU**
4. **Create 8 code cells** using scripts from the `colab/` folder:
   - Cell 1: `colab/cell1_setup.py` → installs libraries
   - **Restart session** when prompted, then skip Cell 1
   - Cell 2: `colab/cell2_data_loading.py` → mounts Drive, loads data
   - Cell 3: `colab/cell3_preprocessing.py` → preprocessing & features
   - Cell 4: `colab/cell4_train_test_split.py` → split & normalize
   - Cell 5: `colab/cell5_tft_model.py` → TFT training (~5-10 min)
   - Cell 6: `colab/cell6_prophet_model.py` → Prophet training (~1-2 min)
   - Cell 7: `colab/cell7_comparison.py` → comparison & analysis
   - Cell 8: `colab/cell8_results_export.py` → export results
5. **All outputs** are auto-saved to `Google Drive > machine-learning-paper/`

### Alternative: Kaggle

1. Go to [kaggle.com](https://www.kaggle.com) → search dataset `electricity-demand-bangladesh`
2. Create a Notebook with GPU enabled, use scripts from the root `cell*.py` files
3. Download outputs from `/kaggle/working/`

### Important Notes
- Cell 5 (TFT) requires **GPU** and takes ~5-10 minutes
- Cell 6 (Prophet) runs on **CPU** and takes ~1-2 minutes
- All cells must be run **in order** (each depends on previous variables)
- If dataset is not found, an upload widget will appear automatically

---

## 9. Code Structure

```
ML-Paper/
├── BangladeshData_2016_2024.csv          # Dataset
├── cell1_setup.py                         # Kaggle: Environment setup
├── cell2_data_loading.py                  # Kaggle: Data loading & EDA
├── cell3_preprocessing.py                 # Kaggle: Preprocessing
├── cell4_train_test_split.py              # Kaggle: Split & normalize
├── cell5_tft_model.py                     # Kaggle: TFT model
├── cell6_prophet_model.py                 # Kaggle: Prophet model
├── cell7_comparison.py                    # Kaggle: Comparison
├── cell8_results_export.py                # Kaggle: Export results
├── colab/                                 # ★ Google Colab versions
│   ├── cell1_setup.py                     # Install + restart
│   ├── cell2_data_loading.py              # Mount Drive + load data
│   ├── cell3_preprocessing.py             # Preprocessing
│   ├── cell4_train_test_split.py          # Split & normalize
│   ├── cell5_tft_model.py                # TFT model
│   ├── cell6_prophet_model.py             # Prophet model
│   ├── cell7_comparison.py                # Comparison
│   └── cell8_results_export.py            # Export to Drive
├── paper.tex                              # IEEE LaTeX paper
├── README.md                              # This documentation
└── electricity-demand-forecasting-based-on-tft.ipynb  # Original reference
```

### Cell Dependencies
```
cell1 → cell2 → cell3 → cell4 → cell5 ─┐
                                  └→ cell6 ─┤
                                            └→ cell7 → cell8
```

---

## 10. Results Summary

> **Note**: Replace these with actual results after running on Colab/Kaggle.

| Metric | TFT | Prophet | Winner |
|--------|-----|---------|--------|
| MAPE (%) | [Run experiment] | [Run experiment] | [TBD] |
| MSE | [Run experiment] | [Run experiment] | [TBD] |
| RMSE | [Run experiment] | [Run experiment] | [TBD] |
| MAE | [Run experiment] | [Run experiment] | [TBD] |
| R² Score | [Run experiment] | [Run experiment] | [TBD] |
| Training Time | [Run experiment] | [Run experiment] | [TBD] |

---

## 11. Literature Review

### Key Papers

1. **Lim et al. (2021)** — *"Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting"*
   - Introduced TFT architecture
   - Combines attention, LSTM, and variable selection
   - IJOF, Vol. 37, No. 4

2. **Taylor & Letham (2018)** — *"Forecasting at Scale"*
   - Introduced Facebook Prophet
   - Additive decomposition with Fourier seasonality
   - The American Statistician, Vol. 72, No. 1

3. **Vaswani et al. (2017)** — *"Attention Is All You Need"*
   - Original Transformer architecture
   - Self-attention mechanism foundation
   - NeurIPS 2017

4. **Hochreiter & Schmidhuber (1997)** — *"Long Short-Term Memory"*
   - LSTM architecture used in TFT's encoder-decoder
   - Neural Computation, Vol. 9, No. 8

5. **Herzen et al. (2022)** — *"Darts: User-Friendly Modern Machine Learning for Time Series"*
   - The library used for TFT implementation
   - JMLR, Vol. 23, No. 124

6. **Box et al. (2015)** — *"Time Series Analysis: Forecasting and Control"*
   - Foundation of time series analysis (ARIMA)
   - Wiley, 5th edition

7. **Hong & Fan (2016)** — *"Probabilistic Electric Load Forecasting"*
   - Tutorial review of load forecasting methods
   - IJF, Vol. 32, No. 3

---

## 12. Statistical Methods

### Normality Testing
- **StandardScaler** transforms features to zero mean and unit variance: `z = (x - μ) / σ`
- This is important because:
  - Neural networks train better with normalized inputs
  - Prevents features with large magnitudes from dominating
  - Ensures gradient descent converges efficiently

### IQR Outlier Method
- Non-parametric method (doesn't assume normal distribution)
- Robust to extreme values
- Standard threshold: 1.5 × IQR
- Outliers replaced with mean (preserves sample size)

### Evaluation Metrics Explained
1. **MAPE** (Mean Absolute Percentage Error) — Percentage-based, intuitive, scale-independent
2. **MSE** (Mean Squared Error) — Penalizes large errors heavily (squared)
3. **RMSE** (Root Mean Squared Error) — Same units as target, most commonly reported
4. **MAE** (Mean Absolute Error) — Average absolute deviation, robust to outliers
5. **R²** (Coefficient of Determination) — Proportion of variance explained (0 to 1, higher is better)

### Residual Analysis
- Residuals = Actual - Predicted
- Good model: residuals should be:
  - Centered around zero (no systematic bias)
  - Normally distributed (no patterns)
  - Homoscedastic (constant variance)

---

## 13. FAQ / Potential Viva Questions

### Q1: What is TFT and how does it work?
**A**: The Temporal Fusion Transformer is a deep learning architecture for multi-horizon time series forecasting. It combines Variable Selection Networks (to identify important features), LSTM encoder-decoder (to capture local temporal patterns), and Multi-Head Attention (to capture long-range dependencies). It was proposed by Lim et al. (2021).

### Q2: What is Facebook Prophet and how does it handle seasonality?
**A**: Prophet is a statistical forecasting model that decomposes time series into trend + seasonality + regressors + error. It handles seasonality using **Fourier series** — representing periodic patterns as sums of sine and cosine functions. It supports daily, weekly, and yearly seasonalities simultaneously.

### Q3: Why did you choose TFT and Prophet for comparison?
**A**: They represent two fundamentally different approaches: TFT is a deep learning method that learns patterns from data (data-driven), while Prophet is a statistical method that models explicit components (model-driven). Comparing them shows when to use complex deep learning vs. simpler statistical models for practical forecasting.

### Q4: What is the IQR method for outlier detection?
**A**: The Interquartile Range (IQR) method identifies outliers as values below Q1 - 1.5×IQR or above Q3 + 1.5×IQR, where Q1 is the 25th percentile, Q3 is the 75th percentile, and IQR = Q3 - Q1. It's a non-parametric method that doesn't assume any specific distribution.

### Q5: Why do you use StandardScaler normalization?
**A**: StandardScaler transforms features to have zero mean and unit variance. This is essential because: (1) neural networks train faster with normalized inputs, (2) it prevents features with larger magnitudes from dominating, and (3) it ensures fair comparison between different features.

### Q6: What are covariates in time series forecasting?
**A**: Covariates are additional variables that help predict the target. In our study, weather variables (temperature, humidity, pressure) are **covariates** that influence electricity demand. TFT uses them as "future covariates" (known in advance from weather forecasts), and Prophet uses them as "regressors."

### Q7: What is MAPE and why is it important?
**A**: MAPE (Mean Absolute Percentage Error) = average of |actual - predicted| / |actual| × 100%. It's important because it's scale-independent (expressed as a percentage), making it easy to interpret and compare across different datasets. A MAPE < 5% is generally considered excellent for load forecasting.

### Q8: What is the attention mechanism in TFT?
**A**: The attention mechanism allows the model to weigh different time steps differently when making predictions. For example, when predicting demand at 6 PM, the model might pay more "attention" to the same hour on previous days, or to similar weather conditions weeks ago. The formula is: Attention(Q,K,V) = softmax(QK^T / √d_k) × V

### Q9: Why does temperature strongly affect electricity demand?
**A**: In Bangladesh's tropical climate, high temperatures drive air conditioning usage. There's a strong positive correlation between temperature and demand. During summer months (May-September), temperatures exceed 35°C, causing demand peaks of 15,000+ MW. Winter months show lower demand due to reduced cooling needs.

### Q10: What are the limitations of your study?
**A**: (1) Limited test period (only December 2024), (2) Low number of TFT training epochs due to computational constraints, (3) No holiday/special event modeling, (4) Aggregate national demand (no regional breakdown), (5) Limited hyperparameter tuning.

### Q11: What is feature engineering and why is it important?
**A**: Feature engineering is the process of creating new informative features from raw data. We created: temporal features (hour, day, month), lag features (demand 4 hours ago), and rolling statistics (24-hour average). These help models capture time-dependent patterns that raw datetime values cannot directly convey.

### Q12: How does Prophet's changepoint detection work?
**A**: Prophet automatically identifies points where the trend changes direction. It uses a regularized approach where potential changepoints are placed at uniform intervals, and a Laplacian prior encourages sparsity (most changepoints have zero magnitude). The `changepoint_prior_scale` parameter controls how flexible the trend is.

### Q13: What is the R² score and how to interpret it?
**A**: R² (coefficient of determination) measures the proportion of variance in the actual values that is explained by the predictions. R² = 1 means perfect prediction, R² = 0 means no better than predicting the mean. Values above 0.85 indicate good model fit for time series forecasting.

### Q14: Could you improve the results?
**A**: Yes, by: (1) increasing TFT training epochs, (2) hyperparameter tuning (grid search or Bayesian optimization), (3) adding more features (economic data, holidays), (4) using ensemble methods (combining TFT and Prophet), (5) using more advanced attention variants.

### Q15: What is the darts library?
**A**: Darts is an open-source Python library by Unit8 for time series forecasting. It provides a unified interface for multiple models including ARIMA, Prophet, TFT, N-BEATS, and more. It supports multivariate time series, covariates, and automatic training/prediction pipelines. Published in JMLR (2022).

---

## 14. References

1. Lim, B., Arik, S.O., Loeff, N., & Pfister, T. (2021). Temporal Fusion Transformers for interpretable multi-horizon time series forecasting. *International Journal of Forecasting*, 37(4), 1748-1764.
2. Taylor, S.J. & Letham, B. (2018). Forecasting at scale. *The American Statistician*, 72(1), 37-45.
3. Vaswani, A., et al. (2017). Attention is all you need. *NeurIPS*, 5998-6008.
4. Hochreiter, S. & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation*, 9(8), 1735-1780.
5. Herzen, J., et al. (2022). Darts: User-friendly modern machine learning for time series. *JMLR*, 23(124), 1-6.
6. Box, G.E.P., et al. (2015). *Time Series Analysis: Forecasting and Control*, 5th ed. Wiley.
7. Hong, T. & Fan, S. (2016). Probabilistic electric load forecasting. *International Journal of Forecasting*, 32(3), 914-938.
8. Chen, T. & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *ACM SIGKDD*, 785-794.
9. Salinas, D., et al. (2020). DeepAR: Probabilistic forecasting with autoregressive recurrent networks. *IJF*, 36(3), 1181-1191.
10. Hyndman, R.J. & Athanasopoulos, G. (2018). *Forecasting: Principles and Practice*, 2nd ed. OTexts.

---

*This documentation was prepared as part of the Machine Learning term paper project.*
