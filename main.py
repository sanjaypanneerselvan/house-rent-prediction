import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Load data
df = pd.read_csv('data/house_rent_data.csv')
df = df[['BHK', 'Size', 'City', 'Furnishing Status', 'Bathroom', 'Rent']]

# Extend cities to include new ones
cities = ['Bangalore', 'Chennai', 'Delhi', 'Coimbatore', 'Madurai', 'Trichy', 'Dindigul']
furnishings = ['Furnished', 'Semi-Furnished', 'Unfurnished']

# Inject dummy rows for all city-furnishing combinations
for city in cities:
    for furnish in furnishings:
        dummy = {
            'BHK': 0,
            'Size': 0,
            'City': city,
            'Furnishing Status': furnish,
            'Bathroom': 0,
            'Rent': 0
        }
        df = pd.concat([df, pd.DataFrame([dummy])], ignore_index=True)

# One-hot encode
df = pd.get_dummies(df, drop_first=True)

# Remove dummy entries (with Rent = 0)
df = df[df['Rent'] > 0]

# Prepare data
X = df.drop('Rent', axis=1)
y = df['Rent']

# Split and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# Save model and feature columns
os.makedirs('model', exist_ok=True)
joblib.dump(model, 'model/rent_model.pkl')
joblib.dump(X.columns.tolist(), 'model/features.pkl')

# Optional: print scores
print("✅ Model trained and saved.")
print("R² Score:", r2_score(y_test, model.predict(X_test)))
print("MSE:", mean_squared_error(y_test, model.predict(X_test)))
