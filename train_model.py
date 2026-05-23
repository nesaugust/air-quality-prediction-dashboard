import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Load data
df = pd.read_csv("global_urban_smog_pm25_hourly.csv")

# Convert timestamp
df["Timestamp"] = pd.to_datetime(df["Timestamp"])

# Time features
df["Hour"] = df["Timestamp"].dt.hour
df["Day"] = df["Timestamp"].dt.day
df["Month"] = df["Timestamp"].dt.month
df["DayOfWeek"] = df["Timestamp"].dt.dayofweek

# Drop missing values
df = df.dropna()

# Features and target
features = [
    "PM10_ug_m3",
    "Carbon_Monoxide_ug_m3",
    "Nitrogen_Dioxide_ug_m3",
    "Ozone_ug_m3",
    "Dust_ug_m3",
    "UV_Index",
    "European_AQI",
    "Hour",
    "Day",
    "Month",
    "DayOfWeek"
]

X = df[features]
y = df["PM2_5_ug_m3"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("MAE:", mae)
print("R2 Score:", r2)

# Save model
joblib.dump(model, "pm25_model.pkl")
print("Model saved as pm25_model.pkl")