# Air Quality Prediction Dashboard

Live Demo:  
https://air-quality-prediction-dashboard-isx8qdwcwfxfar6li9ncx8.streamlit.app/

GitHub Repository:  
https://github.com/nesaugust/air-quality-prediction-dashboard

A machine learning dashboard built with Python and Streamlit to predict PM2.5 air pollution levels and analyze global urban air quality trends.

---

## Project Overview

This project uses hourly air quality data from multiple global cities to predict PM2.5 levels using a Random Forest regression model. The dashboard enables users to monitor pollution trends, explore highly polluted cities, visualize pollution levels on an interactive global map, and generate PM2.5 predictions based on environmental indicators.

The application combines machine learning, data visualization, and interactive analytics within a modern Streamlit dashboard interface.

---

## Features

- PM2.5 prediction using machine learning
- Interactive global pollution dashboard
- City and date filtering
- PM2.5 trend visualization over time
- Top polluted cities analysis
- Interactive global pollution map
- Hazardous air quality event monitoring
- User input prediction tool
- Modern dark blue dashboard UI
- Real-time interactive visualizations

---

## Dataset

The dataset contains hourly urban air quality measurements, including:

- PM2.5
- PM10
- Carbon Monoxide
- Nitrogen Dioxide
- Ozone
- Dust
- UV Index
- European AQI
- City coordinates and location
- Hazardous event indicators
- Timestamp-based environmental data

---

## Tech Stack

- Python
- Pandas
- Scikit-learn
- Random Forest Regressor
- Streamlit
- Plotly
- Joblib

---

## Machine Learning Model

The target variable is:

```text
PM2_5_ug_m3
```

The model was trained using a Random Forest Regressor with engineered time-based features, including:

- Hour
- Day
- Month
- Day of Week

The trained model is saved using Joblib for efficient deployment and fast prediction within the Streamlit application.

## Dashboard Components

### Global Pollution Map

Interactive visualization of PM2.5 levels across global cities using geographic mapping.

### PM2.5 Trend Analysis

Time-series visualization of PM2.5 pollution trends.

### Top Polluted Cities

Ranking of cities with the highest average PM2.5 levels.

### Prediction Tool

Interactive machine learning prediction form using environmental indicators.

### KPI Metrics

Summary statistics including:

- Average PM2.5
- Maximum PM2.5
- Average AQI
- Hazardous event count

---

## How to Run Locally

Clone the repository:

```bash
git clone https://github.com/nesaugust/air-quality-prediction-dashboard.git
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit app:

```bash
streamlit run app.py
```

---

## Project Structure

```text
air-quality-prediction-dashboard/
│
├── app.py
├── train_model.py
├── pm25_model_small.pkl
├── requirements.txt
├── README.md
├── global_urban_smog_pm25_hourly.csv
```

---

## Author

Created by Agnes Jeni Makay

Built with Python, Machine Learning, Streamlit, and interactive data visualization tools.