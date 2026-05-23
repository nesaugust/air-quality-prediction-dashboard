# Air Quality Prediction Dashboard

A machine learning dashboard built with Python and Streamlit to predict PM2.5 air pollution levels and analyze global urban air quality trends.

## Project Overview

This project uses hourly air quality data from multiple cities to predict PM2.5 levels using a Random Forest regression model. The dashboard allows users to explore pollution trends, identify highly polluted cities, view pollution levels on an interactive map, and make PM2.5 predictions based on environmental indicators.

## Features

- PM2.5 prediction using machine learning
- Interactive city filter
- PM2.5 trend visualization
- Top polluted cities analysis
- Interactive pollution map
- Hazardous event summary
- User input prediction tool

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
- City location
- Hazardous event indicator

## Tech Stack

- Python
- Pandas
- Scikit-learn
- Random Forest Regressor
- Streamlit
- Plotly
- Joblib

## Machine Learning Model

The target variable is:

```text
PM2_5_ug_m3