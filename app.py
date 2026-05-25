import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
from datetime import datetime

st.set_page_config(
    page_title="Air Quality Prediction Dashboard",
    page_icon="🌍",
    layout="wide"
)

# ======================
# CSS DESIGN
# ======================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #061A33 0%, #082B57 45%, #03101F 100%);
    color: #EAF6FF;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #082B57 0%, #031A35 100%);
    border-right: 1px solid rgba(255,255,255,0.12);
}

.main-title {
    font-size: 38px;
    font-weight: 800;
    color: #FFFFFF;
    margin-bottom: 5px;
}

.subtitle {
    color: #BFDFFF;
    font-size: 16px;
    margin-bottom: 25px;
}

.card {
    background: rgba(13, 52, 96, 0.75);
    border: 1px solid rgba(95, 180, 255, 0.25);
    padding: 22px;
    border-radius: 20px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.35);
}

[data-testid="stMetric"] {
    background: linear-gradient(135deg, #0B3D91, #0077FF);
    padding: 22px;
    border-radius: 20px;
    box-shadow: 0 8px 25px rgba(0,119,255,0.28);
}

[data-testid="stMetricValue"] {
    color: white !important;
    font-size: 30px !important;
    font-weight: 800 !important;
}

[data-testid="stMetricLabel"] {
    color: #D9ECFF !important;
}

h1, h2, h3 {
    color: #FFFFFF;
}

.stButton > button {
    background: linear-gradient(90deg, #00A6FB, #3949FF);
    color: white;
    border: none;
    border-radius: 14px;
    padding: 12px 28px;
    font-weight: 700;
    width: 100%;
}

.stButton > button:hover {
    background: linear-gradient(90deg, #0077FF, #1A35FF);
    color: white;
}

hr {
    border: 1px solid rgba(255,255,255,0.12);
}
</style>
""", unsafe_allow_html=True)

# ======================
# LOAD DATA & MODEL
# ======================
@st.cache_data
def load_data():
    df = pd.read_csv("global_urban_smog_pm25_hourly.csv")
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])

    df["Hour"] = df["Timestamp"].dt.hour
    df["Day"] = df["Timestamp"].dt.day
    df["Month"] = df["Timestamp"].dt.month
    df["DayOfWeek"] = df["Timestamp"].dt.dayofweek

    return df.dropna()

@st.cache_resource
def load_model():
    return joblib.load("pm25_model_small.pkl")

df = load_data()
model = load_model()

# ======================
# SIDEBAR
# ======================
st.sidebar.markdown("## 🌍 Air Quality Dashboard")
st.sidebar.markdown("### Filters")

city_options = ["All Cities"] + sorted(df["City"].unique().tolist())
selected_city = st.sidebar.selectbox("Select City", city_options)

min_date = df["Timestamp"].min().date()
max_date = df["Timestamp"].max().date()

date_range = st.sidebar.date_input(
    "Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

if len(date_range) == 2:
    start_date, end_date = date_range
    filtered_df = df[
        (df["Timestamp"].dt.date >= start_date) &
        (df["Timestamp"].dt.date <= end_date)
    ]
else:
    filtered_df = df.copy()

if selected_city != "All Cities":
    filtered_df = filtered_df[filtered_df["City"] == selected_city]

st.sidebar.markdown("---")
st.sidebar.markdown("### AQI Guide")
st.sidebar.markdown("""
🟢 **0 - 15** Good  
🟡 **16 - 35** Moderate  
🟠 **36 - 55** Unhealthy for Sensitive  
🔴 **56 - 150** Unhealthy  
🟣 **151+** Hazardous  
""")

# ======================
# HEADER
# ======================
st.markdown('<div class="main-title">Air Quality Prediction Dashboard</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Monitor PM2.5 levels, analyze pollution trends, and predict air quality using machine learning.</div>',
    unsafe_allow_html=True
)

# ======================
# METRICS
# ======================
avg_pm25 = filtered_df["PM2_5_ug_m3"].mean()
max_pm25 = filtered_df["PM2_5_ug_m3"].max()
avg_aqi = filtered_df["European_AQI"].mean()
hazard_count = filtered_df["Hazardous_Event"].sum()

col1, col2, col3, col4 = st.columns(4)

col1.metric("Average PM2.5", f"{avg_pm25:.2f}")
col2.metric("Maximum PM2.5", f"{max_pm25:.2f}")
col3.metric("Average AQI", f"{avg_aqi:.2f}")
col4.metric("Hazardous Events", int(hazard_count))

st.divider()

# ======================
# GLOBAL MAP - NO REPEATED CONTINENTS
# ======================
st.subheader("Global PM2.5 Pollution Map")
st.caption("Latest PM2.5 level by city. This map uses geo projection to avoid repeated continents.")

latest_df = df.sort_values("Timestamp").groupby("City").tail(1)

fig_map = px.scatter_geo(
    latest_df,
    lat="Latitude",
    lon="Longitude",
    size="PM2_5_ug_m3",
    color="PM2_5_ug_m3",
    hover_name="City",
    hover_data={
        "PM2_5_ug_m3": ":.2f",
        "European_AQI": ":.2f",
        "Hazardous_Event": True,
        "Latitude": False,
        "Longitude": False
    },
    color_continuous_scale=["#00D1FF", "#2ECC71", "#F1C40F", "#E67E22", "#E74C3C", "#8E44AD"],
    projection="natural earth",
    height=550
)

fig_map.update_geos(
    showland=True,
    landcolor="#0B2345",
    showocean=True,
    oceancolor="#031A35",
    showcountries=True,
    countrycolor="#1E5A99",
    showcoastlines=True,
    coastlinecolor="#2B7FC3",
    bgcolor="#061A33"
)

fig_map.update_layout(
    paper_bgcolor="#061A33",
    plot_bgcolor="#061A33",
    font=dict(color="white"),
    margin=dict(l=0, r=0, t=20, b=0),
    coloraxis_colorbar=dict(
        title="PM2.5",
        tickfont=dict(color="white"),
        titlefont=dict(color="white")
    )
)

st.plotly_chart(fig_map, use_container_width=True)

# ======================
# TREND + TOP CITIES
# ======================
left_col, right_col = st.columns([1.2, 1])

with left_col:
    st.subheader("PM2.5 Trend Over Time")

    fig_trend = px.line(
        filtered_df,
        x="Timestamp",
        y="PM2_5_ug_m3",
        color="City" if selected_city == "All Cities" else None,
        title="PM2.5 Trend",
        template="plotly_dark"
    )

    fig_trend.update_layout(
        paper_bgcolor="#061A33",
        plot_bgcolor="#061A33",
        font=dict(color="white"),
        title_font=dict(color="#FFFFFF", size=20),
        hovermode="x unified"
    )

    fig_trend.update_traces(line=dict(width=2))

    st.plotly_chart(fig_trend, use_container_width=True)

with right_col:
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
        x="PM2_5_ug_m3",
        y="City",
        orientation="h",
        text="PM2_5_ug_m3",
        template="plotly_dark",
        color="PM2_5_ug_m3",
        color_continuous_scale="Blues"
    )

    fig_bar.update_layout(
        paper_bgcolor="#061A33",
        plot_bgcolor="#061A33",
        font=dict(color="white"),
        yaxis=dict(autorange="reversed"),
        xaxis_title="Average PM2.5",
        yaxis_title="City",
        showlegend=False,
        coloraxis_showscale=False
    )

    fig_bar.update_traces(texttemplate="%{text:.2f}", textposition="outside")

    st.plotly_chart(fig_bar, use_container_width=True)

st.divider()

# ======================
# PREDICTION SECTION
# ======================
st.subheader("Predict PM2.5 Level")
st.info(
    "Prediction is based on the input values below. "
    "City filter is used for dashboard analysis, but the prediction depends on the environmental values you enter."
)

if selected_city != "All Cities" and len(filtered_df) > 0:
    default_data = filtered_df
else:
    default_data = df

col1, col2, col3 = st.columns(3)

with col1:
    pm10 = st.number_input("PM10 (µg/m³)", value=float(default_data["PM10_ug_m3"].mean()))
    carbon = st.number_input("Carbon Monoxide (µg/m³)", value=float(default_data["Carbon_Monoxide_ug_m3"].mean()))
    nitrogen = st.number_input("Nitrogen Dioxide (µg/m³)", value=float(default_data["Nitrogen_Dioxide_ug_m3"].mean()))

with col2:
    ozone = st.number_input("Ozone (µg/m³)", value=float(default_data["Ozone_ug_m3"].mean()))
    dust = st.number_input("Dust (µg/m³)", value=float(default_data["Dust_ug_m3"].mean()))
    uv = st.number_input("UV Index", value=float(default_data["UV_Index"].mean()))

with col3:
    aqi = st.number_input("European AQI", value=float(default_data["European_AQI"].mean()))
    hour = st.slider("Hour", 0, 23, int(default_data["Hour"].mean()))
    month = st.slider("Month", 1, 12, int(default_data["Month"].mean()))

day = datetime.today().day
day_of_week = datetime.today().weekday()

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

    if prediction <= 15:
        status = "Good"
        message_type = "success"
    elif prediction <= 35:
        status = "Moderate"
        message_type = "info"
    elif prediction <= 55:
        status = "Unhealthy for Sensitive Groups"
        message_type = "warning"
    else:
        status = "Unhealthy / Dangerous"
        message_type = "error"

    st.markdown(
        f"""
        <div class="card">
            <h2>Predicted PM2.5 Level: {prediction:.2f} µg/m³</h2>
            <p>This prediction is based on your selected environmental input values.</p>
            <h3>Air Quality Status: {status}</h3>
        </div>
        """,
        unsafe_allow_html=True
    )

    if message_type == "success":
        st.success(f"Air quality condition: {status}")
    elif message_type == "info":
        st.info(f"Air quality condition: {status}")
    elif message_type == "warning":
        st.warning(f"Air quality condition: {status}")
    else:
        st.error(f"Air quality condition: {status}")

# ======================
# FOOTER
# ======================

st.markdown("---")

st.markdown(
    """
    <div style="
        text-align:center;
        padding:20px;
        margin-top:30px;
        border-top:1px solid rgba(255,255,255,0.1);
        color:#BFDFFF;
        font-size:14px;
    ">
        Air Quality Prediction Dashboard © 2026<br>
        <span style="
            color:white;
            font-size:18px;
            font-weight:700;
        ">
            Created by Agnes Jeni Makay
        </span>
    </div>
    """,
    unsafe_allow_html=True
)