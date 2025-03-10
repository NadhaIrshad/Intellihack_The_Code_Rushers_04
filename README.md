# Stock Price Prediction

## Overview
This project focuses on stock price prediction using multiple machine learning models, including Linear Regression, Support Vector Regression (SVR), XGBoost Regressor, and LSTM. The best-performing model is selected based on evaluation metrics such as R² score and RMSE.

## Dataset
The dataset contains historical stock prices with the following columns:
- Date
- Open
- High
- Low
- Close
- Volume

## Data Preprocessing
- **Missing Values:**
  - Forward fill method was used for price-related columns.
  - Linear interpolation was used for volume data.
- **Feature Engineering:**
  - `open-close`: Difference between open and close prices.
  - `low-high`: Difference between low and high prices.
  - `volatility`: 10-day rolling standard deviation of closing price.
  - `volatility_adjusted_high_low`: High-low price range adjusted by volatility.
  - `target`: Future closing price shifted by 5 days.

## Models and Evaluation
### 1. **Linear Regression**
   - Training R² Score: **0.9965**
   - Validation R² Score: **0.9962**

### 2. **SVR (Polynomial Kernel)**
   - Training R² Score: **0.7485**
   - Validation R² Score: **0.7124**

### 3. **XGBoost Regressor** (Selected Model)
   - Training R² Score: **0.9992**
   - Validation R² Score: **0.9968**

### 4. **LSTM Model**
   - Performed worse than XGBoost due to higher RMSE.

### **Why XGBoost?**
- Highest validation R² score (**0.9968**), showing strong predictive capability.
- Handles missing data well and captures non-linear relationships better than Linear Regression.
- Outperformed SVR and LSTM in both training and validation scores.

## Reproducing Results
### **Installation**
```bash
pip install -r requirements.txt
```

### **Running the Model**
```bash
python Q4.py
```

### **Saving Predictions**
The predictions are saved as a CSV file:
```python
predictions.to_csv('predictions.csv', index=False)
```

## Conclusion
XGBoost was chosen as the best model due to its superior performance. Future improvements could include hyperparameter tuning and additional feature engineering.

