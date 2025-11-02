# 📈 S&P 500 Stock Market Prediction using Machine Learning

This project uses historical S&P 500 data and applies machine learning techniques to predict whether the index will go up or down the next day. It involves data collection, feature engineering, model training, backtesting, and evaluation — all within a Jupyter Notebook/Colab environment.

## 🚀 Project Overview

The goal of this project is to:
- Collect historical S&P 500 data using the **yfinance** library.
- Build a **Random Forest Classifier** to predict stock market direction.
- Create custom indicators using rolling averages and trends.
- Implement a **backtesting** strategy to evaluate performance over time.
- Measure model performance using accuracy and precision.

## 📂 Technologies Used

| Library/Tool            | Purpose                              |
|-------------------------|----------------------------------------|
| `yfinance`              | Fetch S&P 500 historical data         |
| `pandas`, `numpy`       | Data manipulation and analysis        |
| `matplotlib`            | Data visualization                    |
| `scikit-learn`          | Machine learning (Random Forest)      |
| `Jupyter Notebook/Colab`| Development environment               |

## 📊 Dataset

- Ticker used: `^GSPC` (S&P 500 Index)
- Columns used: `Open`, `High`, `Low`, `Close`, `Volume`
- Columns removed: `"Dividends"`, `"Stock Splits"`
- Additional columns:
  - `"Tomorrow"`: Next day's closing price
  - `"Target"`: 1 if price goes up tomorrow, else 0

## ⚙️ Model Workflow

### 1. Data Preparation

```python
sp500 = yf.Ticker("^GSPC").history(period="max")
sp500 = sp500.loc["1990-01-01":]
sp500["Tomorrow"] = sp500["Close"].shift(-1)
sp500["Target"] = (sp500["Tomorrow"] > sp500["Close"]).astype(int)
sp500.drop(columns=["Dividends", "Stock Splits"], inplace=True)
```

### 2. Train/Test Split

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=150, min_samples_split=150, random_state=1)
train = sp500.iloc[:-100]
test = sp500.iloc[-100:]
predictors = ["Close", "Volume", "Open", "High", "Low"]
model.fit(train[predictors], train["Target"])
```

## 🔁 Backtesting Function

```python
def backtest(data, model, predictors, start=2500, step=250):
    all_predictions = []
    for i in range(start, data.shape[0], step):
        train = data.iloc[0:i].copy()
        test = data.iloc[i:(i+step)].copy()
        predictions = prediction(train, test, predictors, model)
        all_predictions.append(predictions)
    return pd.concat(all_predictions)
```

## 📈 Performance Metrics

| Metric          | Result |
|-----------------|--------|
| Precision Score | ~0.53  |
| Baseline (up days %) | ~53.7% |

## 🧠 Feature Engineering (Advanced Model)

```python
horizons = [2, 5, 60, 250, 1000]
for horizon in horizons:
    rolling_averages = sp500.rolling(horizon).mean()
    sp500[f"Close_Ratio_{horizon}"] = sp500["Close"] / rolling_averages["Close"]
    sp500[f"Trend_{horizon}"] = sp500.shift(1).rolling(horizon).sum()["Target"]
```

A confidence threshold of **0.75** was applied to predictions.

## ✅ How to Run

```bash
pip install yfinance pandas scikit-learn matplotlib
```

Then open the notebook in **Colab** or **Jupyter Notebook**, and run each cell sequentially.

## 📌 Future Improvements

- Add more technical indicators (RSI, MACD, EMA, etc.)
- Test deep learning models (LSTMs, Transformers)
- Optimize model using Grid Search or Bayesian Optimization
- Deploy using Flask or Streamlit

## 📝 License

This project is open-source and free to use for educational purposes.
