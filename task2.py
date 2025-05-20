import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

df = pd.read_csv("house price data.csv")

df = df.dropna()
df = df.drop_duplicates()

df['house_age'] = 2025 - df['yr_built']
df['is_renovated'] = (df['yr_renovated'] > 0).astype(int)
df['total_sqft'] = df['sqft_above'] + df['sqft_basement']

df = df.drop(columns=['date', 'street', 'country', 'yr_renovated', 'yr_built', 'sqft_above', 'sqft_basement'])

df = pd.get_dummies(df, columns=['city', 'statezip'], drop_first=True)

X = df.drop(columns=['price'])
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print("R² Score:", round(r2, 4))
print("Mean Squared Error:", round(mse, 2))

# 9. Test on Sample Inputs
print("\nSample Predictions:")
sample_input = X_test.iloc[:5]
sample_prediction = model.predict(sample_input)
print(sample_prediction)
