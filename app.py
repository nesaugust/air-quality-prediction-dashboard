import streamlit as st
import pandas as pd
import plotly.express as px
import joblib

st.set_page_config(
    page_title="Air Quality Prediction Dashboard",
    layout="wide"
)

st.title("Air Quality Prediction Dashboard")
st.write("Predict PM2.5 levels and analyze air pollution trends across global cities.")

# Load data
df = pd.read_csv("global_urban_smog_pm25_hourly.csv")
df["Timestamp"] = pd.to_datetime(df["Timestamp"])

# Load model
import os
import joblib
from sklearn.ensemble import RandomForestRegressor

MODEL_PATH = "pm25_model.pkl"

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, MODEL_PATH)

# Sidebar
st.sidebar.header("Filter Options")

city_list = sorted(df["City"].unique())
selected_city = st.sidebar.selectbox("Select City", city_list)

city_df = df[df["City"] == selected_city].copy()

# Time features
city_df["Hour"] = city_df["Timestamp"].dt.hour
city_df["Day"] = city_df["Timestamp"].dt.day
city_df["Month"] = city_df["Timestamp"].dt.month
city_df["DayOfWeek"] = city_df["Timestamp"].dt.dayofweek

# Main metrics
avg_pm25 = city_df["PM2_5_ug_m3"].mean()
max_pm25 = city_df["PM2_5_ug_m3"].max()
avg_aqi = city_df["European_AQI"].mean()
hazard_count = city_df["Hazardous_Event"].sum()

col1, col2, col3, col4 = st.columns(4)

col1.metric("Average PM2.5", round(avg_pm25, 2))
col2.metric("Maximum PM2.5", round(max_pm25, 2))
col3.metric("Average AQI", round(avg_aqi, 2))
col4.metric("Hazardous Events", int(hazard_count))

st.divider()

# PM2.5 trend
st.subheader("PM2.5 Trend Over Time")

fig_trend = px.line(
    city_df,
    x="Timestamp",
    y="PM2_5_ug_m3",
    title=f"PM2.5 Trend in {selected_city}"
)

st.plotly_chart(fig_trend, use_container_width=True)

# Pollution map
st.subheader("City Pollution Map")

latest_df = df.sort_values("Timestamp").groupby("City").tail(1)

fig_map = px.scatter_mapbox(
    latest_df,
    lat="Latitude",
    lon="Longitude",
    size="PM2_5_ug_m3",
    color="PM2_5_ug_m3",
    hover_name="City",
    hover_data=["European_AQI", "Hazardous_Event"],
    zoom=1,
    height=500,
    title="Latest PM2.5 Level by City"
)

fig_map.update_layout(mapbox_style="open-street-map")
st.plotly_chart(fig_map, use_container_width=True)

# Top polluted cities
st.subheader("Top 10 Most Polluted Cities")

top_cities = (
    df.groupby("City")["PM2_5_ug_m3"]
    .mean()
    .sort_values(ascending=False)
    .head(10)
    .reset_index()
)

fig_bar = px.bar(
    top_cities,
    x="City",
    y="PM2_5_ug_m3",
    title="Top 10 Cities by Average PM2.5"
)

st.plotly_chart(fig_bar, use_container_width=True)

st.divider()

# Prediction section
st.subheader("Predict PM2.5 Level")

col1, col2, col3 = st.columns(3)

with col1:
    pm10 = st.number_input("PM10", value=50.0)
    carbon = st.number_input("Carbon Monoxide", value=200.0)
    nitrogen = st.number_input("Nitrogen Dioxide", value=30.0)
    ozone = st.number_input("Ozone", value=40.0)

with col2:
    dust = st.number_input("Dust", value=20.0)
    uv = st.number_input("UV Index", value=3.0)
    aqi = st.number_input("European AQI", value=80.0)

with col3:
    hour = st.slider("Hour", 0, 23, 12)
    day = st.slider("Day", 1, 31, 15)
    month = st.slider("Month", 1, 12, 5)
    day_of_week = st.slider("Day of Week", 0, 6, 2)

input_data = pd.DataFrame({
    "PM10_ug_m3": [pm10],
    "Carbon_Monoxide_ug_m3": [carbon],
    "Nitrogen_Dioxide_ug_m3": [nitrogen],
    "Ozone_ug_m3": [ozone],
    "Dust_ug_m3": [dust],
    "UV_Index": [uv],
    "European_AQI": [aqi],
    "Hour": [hour],
    "Day": [day],
    "Month": [month],
    "DayOfWeek": [day_of_week]
})

if st.button("Predict PM2.5"):
    prediction = model.predict(input_data)[0]

    st.success(f"Predicted PM2.5 Level: {prediction:.2f} µg/m³")

    if prediction <= 15:
        st.info("Air quality condition: Good")
    elif prediction <= 35:
        st.warning("Air quality condition: Moderate")
    elif prediction <= 55:
        st.error("Air quality condition: Unhealthy for Sensitive Groups")
    else:
        st.error("Air quality condition: Unhealthy / Dangerous")