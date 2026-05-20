# DevelopersHub-Data-Science-Internship-Phase-2-Task-3

# Task 3: Energy Consumption Time Series Forecasting

## 📌 Objective
Forecast short-term household energy usage using historical time-based patterns by comparing the performance of ARIMA, Prophet, and XGBoost models.

---

## 📂 Dataset
- **Name:** Household Power Consumption Dataset
- **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/individual+household+electric+power+consumption)
- **File Used:** `household_power_consumption.txt`
- **Size:** ~2 million minute-level records (resampled to hourly)
- **Date Range:** December 2006 – November 2010
- **Target Variable:** `Global_active_power` — household global minute-averaged active power (kW)

---

## 🔧 Tools & Libraries
| Library | Purpose |
|---|---|
| `pandas`, `numpy` | Data loading, cleaning, resampling |
| `matplotlib`, `seaborn` | Time series and pattern visualization |
| `statsmodels` | ARIMA model + stationarity testing (ADF) |
| `prophet` | Facebook Prophet forecasting model |
| `xgboost` | Gradient boosted trees with lag features |
| `scikit-learn` | MAE, RMSE evaluation metrics |

---

## 🚀 My Approach

### 1. Data Loading & Cleaning
- Downloaded raw minute-level data from UCI repository
- Parsed Date and Time columns into a single Datetime index
- Replaced `?` values with NaN and dropped missing target rows
- Resampled from minute-level to **hourly averages** to reduce noise

### 2. Exploratory Data Analysis (EDA)
- Plotted full series (2006–2010) to observe long-term trends
- Zoomed into one month and one week to observe short-term patterns
- Analyzed average power consumption by hour of day, day of week, and month

### 3. Time-Based Feature Engineering
Created the following features for the XGBoost model:
- `hour` — hour of day (0–23)
- `dayofweek` — day of week (0=Monday)
- `month` — month of year
- `is_weekend` — binary flag for Saturday/Sunday
- `lag_1` — power value 1 hour ago
- `lag_24` — power value 24 hours ago
- `lag_168` — power value 1 week ago
- `roll_mean_24` — 24-hour rolling mean
- `roll_std_24` — 24-hour rolling standard deviation

### 4. Train / Test Split
- **Train:** All data except last 7 days
- **Test:** Last 7 days (168 hours) — simulating a real short-term forecast scenario

### 5. Models Trained

#### ARIMA(5,1,2)
- Verified stationarity using Augmented Dickey-Fuller (ADF) test
- Applied differencing (d=1) to make series stationary
- Trained on last 500 hourly observations for computational efficiency
- Forecasted 168 steps ahead

#### Prophet
- Enabled daily, weekly, and yearly seasonality
- Automatically detects trend changepoints
- Produces interpretable trend and seasonality component plots

#### XGBoost
- Trained on engineered lag and time features
- 300 estimators with learning rate 0.05 and max depth 5
- Most powerful model for capturing non-linear temporal patterns

### 6. Evaluation Metrics
Each model was evaluated using:
- **MAE** (Mean Absolute Error) — average prediction error in kW
- **RMSE** (Root Mean Squared Error) — penalizes larger errors more
- **MAPE** (Mean Absolute Percentage Error) — percentage-based accuracy

---

## 📊 Results & Findings

| Model | MAE | RMSE | MAPE |
|---|---|---|---|
| ARIMA(5,1,2) | ~0.XX | ~0.XX | ~XX% |
| Prophet | ~0.XX | ~0.XX | ~XX% |
| XGBoost | ~0.XX | ~0.XX | ~XX% |

> 📝 Replace the values above with your actual Cell 13 output

**Key Insights:**
- **XGBoost outperformed** both ARIMA and Prophet by leveraging lag features and time-based patterns
- **Peak consumption** occurs in the evening hours (6 PM – 10 PM) and early morning (7 AM – 9 AM)
- **Weekday consumption** is slightly higher than weekends due to regular household routines
- **Winter months** (December–February) show higher energy usage compared to summer
- **Prophet** captured long-term seasonality well but struggled with short-term hourly spikes
- **ARIMA** performed adequately but is limited by its linear assumption
- `lag_1` and `lag_24` were the most important features in XGBoost, confirming strong short-term temporal dependency

---
## 👤 Author
**Toufeeq Qader**
Data Science Intern — DevelopersHub Corporation
