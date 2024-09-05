# Housing Price Prediction

This project demonstrates various machine learning techniques for predicting house prices using regression models. We use datasets with housing-related features such as total rooms, total bedrooms, population, households, and median income to predict house values.

## Project Structure

The project includes several steps:
1. Data Preprocessing
2. Feature Engineering
3. Model Training
4. Model Evaluation
5. Hyperparameter Tuning with GridSearchCV (for RandomForest model)

## Prerequisites

Make sure you have the following Python packages installed:

```bash
pip install pandas numpy scikit-learn
```

## Dataset

The dataset used is the `california_housing_test.csv`. Make sure to update the file path to point to your local file if you are running this script in your environment.

## Code Overview

1. **Linear Regression on California Housing Dataset**
   - Preprocesses the data by selecting relevant features.
   - Splits the data into training and testing sets.
   - Builds a simple linear regression model.
   - Evaluates the model using Mean Squared Error (MSE) and R² score.

2. **Random Forest Regressor with Hyperparameter Tuning**
   - Adds additional feature engineering, such as `TotalRooms` and `TotalBath`.
   - Encodes categorical features using one-hot encoding.
   - Applies StandardScaler to normalize the features.
   - Uses `RandomForestRegressor` for building the model.
   - Uses `GridSearchCV` for hyperparameter tuning to find the best model.
   - Evaluates the tuned model using MSE and R² score.

## Usage

### 1. Linear Regression Example

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
url = "/content/sample_data/california_housing_test.csv"
data = pd.read_csv(url)

# Select features and target
X = data[['total_rooms', 'total_bedrooms', 'population', 'households', 'median_income']]
y = data['median_house_value']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")
```

### 2. Random Forest with Hyperparameter Tuning

```python
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Load the dataset
url = "/content/sample_data/california_housing_test.csv"
data = pd.read_csv(url)

# Feature Engineering
data['TotalRooms'] = data['LivingArea'] + data['BasementArea']
data['TotalBath'] = data['FullBathrooms'] + data['HalfBathrooms']

# Feature and target selection
X = data.drop('SalePrice', axis=1)  # Adjust column names as per your dataset
y = data['SalePrice']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardizing the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model Building with RandomForest and Hyperparameter Tuning
model = RandomForestRegressor(random_state=42)
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Best model
best_model = grid_search.best_estimator_

# Predictions
y_pred = best_model.predict(X_test)

# Evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Best Parameters: {grid_search.best_params_}")
print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")
```

## Results

The models are evaluated using Mean Squared Error (MSE) and R² score:
- **Linear Regression** provides a simple baseline model.
- **Random Forest Regressor** improves the model using feature engineering and hyperparameter tuning.
