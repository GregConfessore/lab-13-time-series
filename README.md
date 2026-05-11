# Lab — Time Series Analysis with the UCI Bike Sharing Dataset

**Module:** Time Series Analysis and Forecasting  
**Estimated Time:** 90–120 minutes  
**Dataset:** UCI Bike Sharing Dataset, `day.csv`  
**Primary Skills:** datetime handling, time series data quality, resampling, rolling averages, seasonality analysis, baseline forecasting, forecast evaluation

---

## Lab Scenario

You are a data analyst for a bike sharing company.

Your manager wants to understand how bike rental demand changes over time and whether simple baseline forecasts can provide a useful starting point for short-term demand planning.

Before building any advanced forecasting model, your job is to complete a practical time series workflow:

1. Load and inspect the dataset.
2. Convert the date column into a proper datetime format.
3. Sort and index the data chronologically.
4. Check whether the time series has missing dates or duplicate timestamps.
5. Explore trend, seasonality, and rolling averages.
6. Build simple baseline forecasts.
7. Evaluate the forecasts with MAE and RMSE.
8. Communicate your findings clearly.

This lab is intentionally focused on workflow and interpretation, not advanced forecasting models.

---

## Dataset Overview

You will use the **UCI Bike Sharing Dataset**.

For this lab, use the file:

```text
day.csv
```

Each row represents one day of bike rental activity.

### Important Columns

| Column | Description |
|---|---|
| `instant` | Record index |
| `dteday` | Date |
| `season` | Season encoded as a number |
| `yr` | Year encoded as `0` or `1` |
| `mnth` | Month number |
| `holiday` | Whether the day was a holiday |
| `weekday` | Day of the week |
| `workingday` | Whether the day was a working day |
| `weathersit` | Weather situation category |
| `temp` | Normalized temperature |
| `atemp` | Normalized feeling temperature |
| `hum` | Normalized humidity |
| `windspeed` | Normalized wind speed |
| `casual` | Count of casual users |
| `registered` | Count of registered users |
| `cnt` | Total rental bike count |

For this lab, your main target column is:

```python
cnt
```

This represents total bike rentals for the day.

---

## Lab Learning Objectives

By the end of this lab, you will be able to:

- Load a real time series dataset into pandas.
- Convert a date column using `pd.to_datetime()`.
- Sort a dataset chronologically.
- Set a datetime column as the DataFrame index.
- Check for duplicate timestamps.
- Check for missing dates in a daily time series.
- Resample daily data into weekly and monthly summaries.
- Use rolling averages to smooth noisy daily data.
- Analyze trend and seasonality.
- Create simple baseline forecasts.
- Evaluate forecasts using MAE and RMSE.
- Explain the limitations of baseline forecasting.

---

## Part 0 — Setup

### Task

Import the libraries needed for the lab. 

### Code

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import mean_absolute_error, mean_squared_error
```

```python
sns.set_theme(style="whitegrid")
pd.set_option("display.max_columns", 30)
```

### Checkpoint

Run the cell without errors before moving on.

---

## Part 1 — Load the Dataset

### Option A: Load from a Local CSV File

Use this option if `day.csv` is already saved in your project folder.

```python
bike = pd.read_csv("day.csv")
```

### Option B: Load Directly from a URL

Use this option if your instructor gives you a direct raw CSV URL.

```python
url = "YOUR_RAW_CSV_URL_HERE"
bike = pd.read_csv(url)
```

### Inspect the Dataset

```python
bike.head()
```

```python
bike.shape
```

```python
bike.info()
```

```python
bike.describe()
```

### Questions

Answer these in a markdown cell:

1. How many rows are in the dataset?
2. What does one row represent?
3. Which column contains the date?
4. Which column contains the total number of rentals?
5. Is the date column already stored as a datetime type?

---

## Part 2 — Create a Proper Datetime Index

The column `dteday` contains the date, but pandas may initially read it as text.

### Tasks:

1. Convert `dteday` to Datetime

2. Sort by Date

3. Set the Date as the Index

### Questions

Answer these in a markdown cell:

1. Why do we convert `dteday` with `pd.to_datetime()`?
2. Why should time series data be sorted before analysis?
3. What changed after setting `dteday` as the index?

---

## Part 3 — Time Series Data Quality Checks

Before analyzing a time series, check whether the time index is complete, unique, sorted, and regular.

### Tasks 

1. Check for Duplicate Timestamps
2. Check Whether the Index Is Sorted
3. Infer the Frequency
4. Check for Missing Dates

### Questions

Answer these in a markdown cell:

1. Are there duplicate dates?
2. Is the index sorted chronologically?
3. Does pandas detect a daily frequency?
4. Are any dates missing?
5. Why are missing timestamps different from missing values?

---

## Part 4 — Initial Time Series Visualization

### Tasks 

1. Plot Daily Bike Rentals

### Questions

Answer these in a markdown cell:

1. Does bike rental demand appear to change over time?
2. Do you notice any repeating patterns?
3. Does the data look noisy at the daily level?
4. Are there any sudden drops or spikes worth investigating?

---

## Part 5 — Resampling Daily Data

Daily data can be noisy. Resampling helps summarize the data at a different time level.

### Tasks 

1. Weekly Total Rentals
2. Monthly Total Rentals
3. Monthly Average Daily Rentals

### Questions

Answer these in a markdown cell:

1. What question does weekly total rentals answer?
2. What question does monthly total rentals answer?
3. What question does monthly average daily rentals answer?
4. How does resampling change what you can see in the data?
5. Which version is easier to interpret: daily, weekly, or monthly? Why?

---

## Part 6 — Rolling Averages

Rolling averages help smooth short-term fluctuations.

### Tasks 

1. Create Rolling Average Columns for 7 and 30 days
2. Plot Daily Rentals with Rolling Averages

### Questions

Answer these in a markdown cell:

1. What does the 7-day rolling average smooth out?
2. What does the 30-day rolling average reveal?
3. Which line is most useful for seeing short-term changes?
4. Which line is most useful for seeing the larger trend?
5. Why do the first few rolling average values appear as missing?

---

## Part 7 — Date Part Features

Extracting date parts helps us analyze calendar-based patterns.

### Tasks 

1. Create Date Part Columns (Year, Month, Day, etc.)

---

## Part 8 — Analyze Seasonality

### Tasks 

1. Average Rentals by Month
2. Average Rentals by Day of Week
3. Compare Rentals by Season

### Questions

Answer these in a markdown cell:

1. Which months have the highest average rentals?
2. Which months have the lowest average rentals?
3. Are rentals higher on weekdays or weekends?
4. Which season has the highest rental counts?
5. What seasonal patterns do you notice?

---

## Part 9 — Chronological Train/Test Split

Forecasting requires training on the past and testing on the future.

Do **not** use a random train/test split.

### Tasks 

1. Create the your forecasting Series
2. Use the Last 60 Days as the Test Set
3. Plot the Split

### Questions

Answer these in a markdown cell:

1. What date range is used for training?
2. What date range is used for testing?
3. Why is a random split inappropriate for time series forecasting?

---

## Part 11 — Baseline Forecast 1: Naive Forecast

A naive forecast predicts that every future value will equal the last observed training value.

### Task

```python
last_train_value = train.iloc[-1]

naive_forecast = pd.Series(
    data=last_train_value,
    index=test.index
)
```

```python
naive_forecast.head()
```

```python
plt.figure(figsize=(14, 5))

sns.lineplot(x=train.index, y=train.values, label="Train")
sns.lineplot(x=test.index, y=test.values, label="Actual Test")
sns.lineplot(x=naive_forecast.index, y=naive_forecast.values, label="Naive Forecast")

plt.title("Naive Forecast")
plt.xlabel("Date")
plt.ylabel("Total Rentals")
plt.legend()
plt.show()
```

### Question

Answer in a markdown cell:

1. What assumption does the naive forecast make?
2. Does that assumption seem reasonable for this dataset? Why or why not?

---

## Part 12 — Baseline Forecast 2: 7-Day Average Forecast

A 7-day average forecast uses the average of the final week of training data and repeats it across the test period.

### Task

```python
last_7_day_average = train.tail(7).mean()

average_7_day_forecast = pd.Series(
    data=last_7_day_average,
    index=test.index
)
```

```python
average_7_day_forecast.head()
```

```python
plt.figure(figsize=(14, 5))

sns.lineplot(x=test.index, y=test.values, label="Actual Test")
sns.lineplot(x=average_7_day_forecast.index, y=average_7_day_forecast.values, label="7-Day Average Forecast")

plt.title("7-Day Average Forecast")
plt.xlabel("Date")
plt.ylabel("Total Rentals")
plt.legend()
plt.show()
```

### Question

Answer in a markdown cell:

1. How is this forecast different from the naive forecast?
2. What information does it use?
3. What information does it ignore?

---

## Part 13 — Baseline Forecast 3: Seasonal Weekly Naive Forecast

Because this is daily data, demand may follow a weekly pattern.

A seasonal weekly naive forecast predicts each test day using the value from 7 days earlier.

### Task

```python
seasonal_weekly_values = []

for date in test.index:
    previous_week_date = date - pd.DateOffset(days=7)
    seasonal_weekly_values.append(series.loc[previous_week_date])

seasonal_weekly_forecast = pd.Series(
    data=seasonal_weekly_values,
    index=test.index
)
```

```python
seasonal_weekly_forecast.head(10)
```

```python
plt.figure(figsize=(14, 5))

sns.lineplot(x=test.index, y=test.values, label="Actual Test")
sns.lineplot(x=seasonal_weekly_forecast.index, y=seasonal_weekly_forecast.values, label="Seasonal Weekly Naive Forecast")

plt.title("Seasonal Weekly Naive Forecast")
plt.xlabel("Date")
plt.ylabel("Total Rentals")
plt.legend()
plt.show()
```

### Question

Answer in a markdown cell:

1. Why might a 7-day seasonal forecast make sense for daily bike rental data?
2. What pattern is this forecast trying to capture?
3. When might this forecast fail?

---

## Part 14 — Compare Baseline Forecasts

### Task

```python
forecast_comparison = pd.DataFrame({
    "actual": test,
    "naive": naive_forecast,
    "average_7_day": average_7_day_forecast,
    "seasonal_weekly_naive": seasonal_weekly_forecast
})

forecast_comparison.head()
```

```python
plt.figure(figsize=(14, 6))

sns.lineplot(data=forecast_comparison, x=forecast_comparison.index, y="actual", label="Actual")
sns.lineplot(data=forecast_comparison, x=forecast_comparison.index, y="naive", label="Naive")
sns.lineplot(data=forecast_comparison, x=forecast_comparison.index, y="average_7_day", label="7-Day Average")
sns.lineplot(data=forecast_comparison, x=forecast_comparison.index, y="seasonal_weekly_naive", label="Seasonal Weekly Naive")

plt.title("Baseline Forecast Comparison")
plt.xlabel("Date")
plt.ylabel("Total Rentals")
plt.legend()
plt.show()
```

### Questions

Answer these in a markdown cell:

1. Which forecast looks most realistic visually?
2. Which forecast appears too flat?
3. Which forecast best follows short-term movement?
4. Do any forecasts appear to consistently overpredict or underpredict?

---

## Part 15 — Evaluate Forecast Accuracy

Use MAE and RMSE to compare the baseline forecasts.

### Task 1: Define an Evaluation Function

```python
def evaluate_forecast(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    
    return {
        "MAE": mae,
        "RMSE": rmse
    }
```

### Task 2: Evaluate Each Forecast

```python
naive_metrics = evaluate_forecast(test, naive_forecast)
average_7_day_metrics = evaluate_forecast(test, average_7_day_forecast)
seasonal_weekly_metrics = evaluate_forecast(test, seasonal_weekly_forecast)
```

```python
results = pd.DataFrame({
    "Naive": naive_metrics,
    "7-Day Average": average_7_day_metrics,
    "Seasonal Weekly Naive": seasonal_weekly_metrics
}).T

results
```

```python
results.sort_values("MAE")
```

### Questions

Answer these in a markdown cell:

1. Which forecast had the lowest MAE?
2. Which forecast had the lowest RMSE?
3. Did the same forecast perform best on both metrics?
4. Did the metric results match what you saw visually?
5. Which baseline would you recommend as the best simple forecast?

---

## Part 16 — Forecast Error Analysis

Forecast errors can show whether a model is consistently too high or too low.

### Task 1: Calculate Errors

```python
forecast_comparison["naive_error"] = forecast_comparison["actual"] - forecast_comparison["naive"]
forecast_comparison["average_7_day_error"] = forecast_comparison["actual"] - forecast_comparison["average_7_day"]
forecast_comparison["seasonal_weekly_error"] = forecast_comparison["actual"] - forecast_comparison["seasonal_weekly_naive"]
```

```python
forecast_comparison.head()
```

### Task 2: Plot Seasonal Weekly Forecast Errors

```python
plt.figure(figsize=(14, 5))

sns.lineplot(
    data=forecast_comparison,
    x=forecast_comparison.index,
    y="seasonal_weekly_error",
    marker="o"
)

plt.axhline(0, linestyle="--")
plt.title("Seasonal Weekly Naive Forecast Errors")
plt.xlabel("Date")
plt.ylabel("Actual - Forecast")
plt.show()
```

### Questions

Answer these in a markdown cell:

1. What does a positive error mean?
2. What does a negative error mean?
3. Was the seasonal weekly forecast usually too high, too low, or mixed?
4. Are there any days where the forecast made a large mistake?
5. What might explain large forecast errors?

---

## Part 17 — Final Written Interpretation

Write a short conclusion that answers the following questions.

### Required Questions

1. What is the overall trend in bike rentals?
2. What seasonal or calendar patterns did you find?
3. How did weather appear to relate to rental demand?
4. Which baseline forecast performed best?
5. Why do you think that forecast performed best?
6. What are the limitations of these baseline forecasts?
7. What additional information or modeling approach might improve future forecasts?

### Example Format

```text
The bike sharing data shows that rental demand changes over time and appears to have seasonal patterns. Rentals tend to be higher during ______ and lower during ______. The rolling averages helped reveal ______. Among the baseline forecasts, ______ performed best based on ______. This likely happened because ______. However, these baseline forecasts are limited because ______. A stronger future model might include ______.
```

---

## Optional Extension 1 — Registered vs. Casual Riders

Analyze casual and registered users separately.

```python
plt.figure(figsize=(14, 5))

sns.lineplot(data=bike_ts, x=bike_ts.index, y="casual", label="Casual")
sns.lineplot(data=bike_ts, x=bike_ts.index, y="registered", label="Registered")

plt.title("Casual vs. Registered Bike Rentals Over Time")
plt.xlabel("Date")
plt.ylabel("Rentals")
plt.legend()
plt.show()
```

Questions:

1. Which group has more rentals overall?
2. Which group appears more seasonal?
3. How might these user groups behave differently?

---


## Optional Extension 2 — Try a Monthly Forecast

Instead of forecasting daily rentals, resample the data to monthly totals and forecast monthly demand.

```python
monthly_series = bike_ts["cnt"].resample("MS").sum()
```

Then repeat the train/test split and baseline forecast workflow.

Questions:

1. Is monthly demand easier or harder to forecast visually?
2. How do the forecast errors compare with the daily forecast errors?
3. What information is lost when moving from daily to monthly data?

---