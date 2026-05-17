# ☕ Coffee Vending Machine Sales — EDA & Prediction

A data science project analyzing sales data from a coffee vending machine (March–August 2024), covering time series exploratory data analysis, feature engineering, and daily sales prediction using a Random Forest Regressor.

---

## 📊 Dataset

- **Source:** Coffee vending machine transaction logs
- **Size:** 1,313 transactions × 6 columns
- **Period:** March 2024 – August 2024
- **Features:** Date, Datetime, Cash Type, Card ID, Amount Paid (money), Coffee Name

**Coffee types sold:**

| Coffee | Transactions |
|---|---|
| Americano with Milk | 319 |
| Latte | 282 |
| Cappuccino | 222 |
| Americano | 185 |
| Cortado | 123 |
| Hot Chocolate | 79 |
| Espresso | 59 |
| Cocoa | 44 |

**Payment method:** 93.2% card / 6.8% cash

---

## 🔍 Project Pipeline

### 1. Exploratory Data Analysis (EDA)
- Inspected nulls, duplicates, data types, and descriptive statistics
- Identified repeat card usage — 793 duplicate card entries, indicating a loyal local customer base near the machine
- Distribution and time series plots for all numerical features
- Sales breakdown by weekday and month
- Coffee type sales distribution with bar chart
- Monthly sales by coffee type — identified top 3 sellers: Americano with Milk, Latte, Cappuccino
- Business insight: Cocoa and Espresso are the lowest sellers — stock reduction recommended to minimize expiry risk

### 2. Feature Engineering
- Parsed `date` into year, month, day, and weekday
- Parsed `datetime` into hour, minute, and second (for time-of-day analysis)
- Computed `total_sale_daily` — total revenue per day (groupby aggregation)
- Computed `total_sale_monthly` — total revenue per month (groupby aggregation)
- Renamed `money` column to `sales`

### 3. Data Transformation
- Label encoded `cash_type` (card=0, cash=1)
- One-Hot Encoded `coffee_name` (drop_first=True)
- Applied MinMax Scaling across all features

### 4. Machine Learning — Daily Sales Prediction
- **Target:** `total_sale_daily`
- **Model:** Random Forest Regressor
- **Tuning:** GridSearchCV with 5-fold cross-validation
  - Parameters tuned: `n_estimators` [100, 200, 300], `max_depth` [10, 20, None], `min_samples_split` [2, 5, 10]
- **Metrics reported:** MSE, MAE, R² Score (before and after tuning)
- **Feature importance** bar chart to identify most predictive variables

---

## 📁 Project Structure

```
coffee-sales-prediction/
│
├── coffee_sales_EDA_and_prediction_.ipynb   # Main notebook
├── coffee_sales.csv                          # Dataset
└── README.md
```

---

## ▶️ How to Run

1. Clone the repository
```bash
git clone https://github.com/your-username/coffee-sales-prediction.git
```

2. Install the required libraries
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

3. Open the notebook in Jupyter
```bash
jupyter notebook coffee_sales_EDA_and_prediction_.ipynb
```

4. Make sure `coffee_sales.csv` is in the same folder as the notebook, then run all cells

---

## 📦 Libraries Used

| Library | Purpose |
|---|---|
| `pandas` / `numpy` | Data loading and manipulation |
| `matplotlib` / `seaborn` | Data visualization |
| `scikit-learn` | Preprocessing, Random Forest, GridSearchCV, metrics |

---

## 💡 Key Findings

- **Americano with Milk, Latte, and Cappuccino** account for over 62% of all transactions — these are the core products to always keep stocked
- **Cocoa and Espresso** are the slowest movers — reducing their stock would minimize waste and expiry risk
- The vast majority of purchases (93%) are made by card, and repeat card usage suggests a loyal customer base likely living or working close to the machine
- Random Forest with GridSearchCV hyperparameter tuning achieved strong R² performance on daily sales prediction
- `total_sale_daily`, month, and hour features were the strongest predictors of sales volume

---

## 👤 Author

**Khaled Abdulaziz**
