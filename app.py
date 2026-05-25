import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import os

st.set_page_config(
    page_title="Air Quality Prediction Dashboard",
    page_icon="🌍",
    layout="wide"
)

# =========================
# CUSTOM CSS
# =========================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #071B33 0%, #0A2342 45%, #0F3D68 100%);
    color: white;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #06213F 0%, #0B3D91 100%);
}

.main-title {
    font-size: 38px;
    font-weight: 800;
    color: #EAF6FF;
}

.subtitle {
    font-size: 16px;
    color: #BFDFFF;
    margin-bottom: 25px;
}

.card {
    background: rgba(255, 255, 255, 0.10);
    padding: 22px;
    border-radius: 20px;
    border: 1px solid rgba(255, 255, 255, 0.18);
    box-shadow: 0 8px 30px rgba(0,0,0,0.25);
}

[data-testid="stMetric"] {
    background: linear-gradient(135deg, #0B77E3, #00A6FB);
    padding: 20px;
    border-radius: 18px;
    color: white;
    box-shadow: 0px 8px 25px rgba(0, 166, 251, 0.25);
}

[data-testid="stMetricLabel"] {
    color: #EAF6FF !important;
}

[data-testid="stMetricValue"] {
    color: white !important;
    font-size: 28px !important;
    font-weight: 800 !important;
}

h1, h2, h3 {
    color: #EAF6FF;
}

.stButton > button {
    background: linear-gradient(90deg, #00A6FB, #0077FF);
    color: white;
    border: none;
    border-radius: 14px;
    padding: 12px 28px;
    font-weight: 700;
    width: 100%;
}

.stButton > button:hover {
    background: linear-gradient(90deg, #0077FF, #005BBB);
    color: white;
}

hr {
    border: 1px solid rgba(255,255,255,0.15);
}
</style>
""", unsafe_allow_html=True)

# =========================
# HEADER
# =========================
st.markdown('<div class="main-title">Air Quality Prediction Dashboard</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Monitor PM2.5 levels, explore pollution trends, and predict air quality across global cities.</div>',
    unsafe_allow_html=True
)

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("global_urban_smog_pm25_hourly.csv")
df["Timestamp"] = pd.to_datetime(df["Timestamp"])

MODEL_PATH = "pm25_model_small.pkl"
model = joblib.load(MODEL_PATH)

# =========================
# SIDEBAR
# =========================
st.sidebar.title("Dashboard Filters")
st.sidebar.write("Choose a city to explore pollution trends.")

city_list = sorted(df["City"].unique())
selected_city = st.sidebar.selectbox("Select City", city_list)

city_df = df[df["City"] == selected_city].copy()

city_df["Hour"] = city_df["Timestamp"].dt.hour
city_df["Day"] = city_df["Timestamp"].dt.day
city_df["Month"] = city_df["Timestamp"].dt.month
city_df["DayOfWeek"] = city_df["Timestamp"].dt.dayofweek

# =========================
# METRICS
# =========================
avg_pm25 = city_df["PM2_5_ug_m3"].mean()
max_pm25 = city_df["PM2_5_ug_m3"].max()
avg_aqi = city_df["European_AQI"].mean()
hazard_count = city_df["Hazardous_Event"].sum()

col1, col2, col3, col4 = st.columns(4)

col1.metric("Average PM2.5", f"{avg_pm25:.2f}")
col2.metric("Maximum PM2.5", f"{max_pm25:.2f}")
col3.metric("Average AQI", f"{avg_aqi:.2f}")
col4.metric("Hazardous Events", int(hazard_count))

st.divider()

# =========================
# PM2.5 TREND
# =========================
st.subheader("PM2.5 Trend Over Time")

fig_trend = px.line(
    city_df,
    x="Timestamp",
    y="PM2_5_ug_m3",
    title=f"PM2.5 Trend in {selected_city}",
    template="plotly_dark"
)

fig_trend.update_traces(line=dict(color="#00A6FB", width=2))

fig_trend.update_layout(
    plot_bgcolor="#071B33",
    paper_bgcolor="#071B33",
    font=dict(color="white"),
    title_font=dict(size=22, color="#EAF6FF"),
    xaxis_title="Date",
    yaxis_title="PM2.5 Level",
    hovermode="x unified"
)

st.plotly_chart(fig_trend, use_container_width=True)

# =========================
# MAP
# =========================
st.subheader("City Pollution Map")

latest_df = df.sort_values("Timestamp").groupby("City").tail(1)

fig_map = px.scatter_mapbox(
    latest_df,
    lat="Latitude",
    lon="Longitude",
    size="PM2_5_ug_m3",
    color="PM2_5_ug_m3",
    color_continuous_scale="Blues",
    hover_name="City",
    hover_data=["PM2_5_ug_m3", "European_AQI", "Hazardous_Event"],
    zoom=1,
    height=520,
    title="Latest PM2.5 Level by City"
)

fig_map.update_layout(
    mapbox_style="carto-darkmatter",
    paper_bgcolor="#071B33",
    font=dict(color="white"),
    title_font=dict(size=22, color="#EAF6FF"),
    margin=dict(l=0, r=0, t=50, b=0)
)

st.plotly_chart(fig_map, use_container_width=True)

# =========================
# TOP POLLUTED CITIES
# =========================
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
    title="Top 10 Cities by Average PM2.5",
    template="plotly_dark",
    text_auto=".2f"
)

fig_bar.update_traces(
    marker_color="#00A6FB",
    textposition="outside"
)

fig_bar.update_layout(
    plot_bgcolor="#071B33",
    paper_bgcolor="#071B33",
    font=dict(color="white"),
    title_font=dict(size=22, color="#EAF6FF"),
    xaxis_title="City",
    yaxis_title="Average PM2.5"
)

st.plotly_chart(fig_bar, use_container_width=True)

st.divider()

# =========================
# PREDICTION SECTION
# =========================
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

    st.markdown(
        f"""
        <div class="card">
            <h2>Predicted PM2.5 Level: {prediction:.2f} µg/m³</h2>
        </div>
        """,
        unsafe_allow_html=True
    )

    if prediction <= 15:
        st.success("Air quality condition: Good")
    elif prediction <= 35:
        st.info("Air quality condition: Moderate")
    elif prediction <= 55:
        st.warning("Air quality condition: Unhealthy for Sensitive Groups")
    else:
        st.error("Air quality condition: Unhealthy / Dangerous")