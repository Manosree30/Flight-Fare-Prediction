# Flight Price Prediction Project

## Project Overview

This project aims to **predict flight prices** based on various features such as airline, source and destination cities, departure and arrival times, flight class, duration, and days left for travel. Multiple machine learning regression models are applied to identify the most accurate predictor of flight prices.

---

## Dataset

- **Source**: CSV file (`Clean_Dataset.csv`) containing 300,153 flight records.  
- **Columns**:

| Column Name         | Description |
|--------------------|-------------|
| airline            | Airline name |
| source_city        | City of departure |
| departure_time     | Departure time category |
| stops              | Number of stops (zero, one, two) |
| arrival_time       | Arrival time category |
| destination_city   | City of arrival |
| class              | Flight class (Economy=1, Business=0) |
| duration           | Flight duration in hours |
| days_left          | Days left until flight |
| price              | Flight price (target variable) |

- Unnecessary columns like `Unnamed: 0` and `flight` were dropped.

---

## Data Preprocessing

1. **Encoding**:
   - `class` column: Economy → 1, Business → 0  
   - Categorical features (airline, source_city, departure_time, stops, arrival_time, destination_city) → One-hot encoded  

2. **Outlier Handling**:
   - Numeric columns were checked for outliers using **boxplots**.  
   - Outliers were capped using the **IQR method**.

3. **Skewness Reduction**:
   - `price` column skewness reduced using **log transformation** (`np.log1p`).

4. **Feature Scaling**:
   - Numeric columns scaled using **StandardScaler** in the pipeline.

---

## Exploratory Data Analysis (EDA)

- Distribution of numeric features was visualized using **histograms and KDE plots**.
- Boxplots were used to identify and cap outliers.
- **Correlation matrix** was plotted to visualize relationships between numeric features.

---

## Model Building

The following regression models were tested:

| Model                  | Parameters |
|------------------------|------------|
| Linear Regression      | Default |
| Ridge Regression       | alpha=1.0 |
| Decision Tree Regressor | max_depth=10, random_state=42 |
| Random Forest Regressor | n_estimators=20, random_state=42 |
| Gradient Boosting Regressor | n_estimators=20, random_state=42 |
| AdaBoost Regressor     | n_estimators=20, random_state=42 |

- Models were trained using a **pipeline** with preprocessing (one-hot encoding + scaling).
- **Train/Test Split**: 80/20 split with random_state=42.
- **Evaluation Metrics**: MAE, RMSE, R² Score.

---

## Model Performance

| Model                  | MAE       | RMSE       | R² Score |
|------------------------|-----------|------------|----------|
| Random Forest          | 0.196     | 2.206      | 1.000    |
| Decision Tree          | 7.548     | 14.531     | 1.000    |
| AdaBoost               | 1217.287  | 1483.326   | 0.996    |
| Gradient Boosting      | 2425.575  | 2885.956   | 0.984    |
| Linear Regression      | 3740.436  | 5669.276   | 0.938    |
| Ridge Regression       | 3740.228  | 5669.384   | 0.938    |

- **Best Model**: Random Forest Regressor  
- Final pipeline combines **preprocessing** + **best model** for predictions.

---

## Visualization

- Distribution plots (histograms + KDE) for numeric features.
- Boxplots for outlier detection.
- Correlation heatmap for feature relationships.

---

## Usage

1. Load dataset:

```python
import pandas as pd
df = pd.read_csv('Clean_Dataset.csv')
