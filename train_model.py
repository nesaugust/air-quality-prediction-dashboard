import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("global_urban_smog_pm25_hourly.csv")

# Convert timestamp
df["Timestamp"] = pd.to_datetime(df["Timestamp"])

# =========================
# FEATURE ENGINEERING
# =========================
df["Hour"] = df["Timestamp"].dt.hour
df["Day"] = df["Timestamp"].dt.day
df["Month"] = df["Timestamp"].dt.month
df["DayOfWeek"] = df["Timestamp"].dt.dayofweek

# Drop missing values
df = df.dropna()

# =========================
# USE SMALLER SAMPLE
# =========================
# Reduce dataset size for lighter model
sample_size = min(50000, len(df))
df = df.sample(n=sample_size, random_state=42)

# =========================
# FEATURES & TARGET
# =========================
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

# =========================
# SPLIT DATA
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =========================
# LIGHTWEIGHT MODEL
# =========================
model = RandomForestRegressor(
    n_estimators=30,
    max_depth=12,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)

# Train model
model.fit(X_train, y_train)

# =========================
# EVALUATION
# =========================
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("MAE:", round(mae, 2))
print("R2 Score:", round(r2, 2))

# =========================
# SAVE MODEL
# =========================
joblib.dump(model, "pm25_model_small.pkl", compress=3)

print("Model saved as pm25_model_small.pkl")