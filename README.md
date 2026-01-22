# Traffic Congestion Prediction in Beijing (2007–2012)

This project implements a **traffic congestion prediction system for Beijing** using historical GPS trajectory data from the **GeoLife dataset**.  
It applies **machine learning on spatio-temporal features** to identify congestion hotspots and predict congestion along real routes entered by users.

The system supports:
- Large-scale GPS trajectory analysis
- Congestion labeling based on spatial–temporal density
- Supervised model training
- Route-level congestion prediction from natural language addresses
- Map-based visualization of predicted traffic conditions

---

## Project Overview

- **City**: Beijing, China  
- **Time span**: 2007–2012  
- **Dataset size**: ~18 million GPS points  
- **Main model**: Random Forest  
- **Secondary model**: Logistic Regression (custom IRLS)  

The full methodology, experiments, and visual analysis are described in `Report.pdf`.

---

## Repository Structure

.
├── geolife.py # Main script: data loading, training, prediction, visualization
├── Report.pdf # Full project report
├── README.md
└── .gitignore

Large datasets and trained models are **not stored in this repository**.

---

## Models & Dataset (Kaggle)

All required `.pkl` files are hosted on Kaggle due to GitHub size limits.

**Kaggle link:**  
https://www.kaggle.com/models/iawaf1/trajectory-models/

### Files provided on Kaggle
- `Beijing_user_trajectory.pkl` — preprocessed GPS trajectory dataset  
- `rf_model.pkl` — trained Random Forest congestion model  
- `logreg_model.pkl` — trained Logistic Regression model  

---

## How to Run the Project

You can either **download pretrained models** or **train everything yourself**.

---

### Option 1: Use Pretrained Models (Recommended)

1. Download all three `.pkl` files from Kaggle.
2. Place them in the project root:

.
├── geolife.py
├── Report.pdf
├── Beijing_user_trajectory.pkl
├── rf_model.pkl
└── logreg_model.pkl

3. Run:

```bash

python geolife.py

```

### Option 2: Train Models Yourself

1. Download only:

    * Beijing_user_trajectory.pkl

2. Place it in the project root.

3. Run:

```bash

python geolife.py --retrain

```

**This will:**

* Load and preprocess the dataset

* Label congestion using grid-hour density

* Train Logistic Regression and Random Forest models

* Save rf_model.pkl and logreg_model.pkl

⚠️ Training requires significant RAM and time due to dataset size.

## Route Congestion Prediction

Users can predict congestion along a route using natural language input.

**Input format**

Place1, Place2, YYYY-MM-DD HH:MM


**Example**

Beijing Institute of Technology, Forbidden City, 2025-06-21 09:00


**Process**

1. Geocode locations (OpenStreetMap / Nominatim)

2. Generate route (OpenRouteService)

3. Predict congestion at route points

4. Display interactive map:

    🟢 Green — free traffic

    🔴 Red — congested traffic
## Features Used for Prediction

* Latitude

* Longitude

* Hour of day

* Month

* Weekday

Congestion labels are assigned when more than 100 GPS points occur in the same spatial grid and hour.

## Requirements

* Python 3.8+

* Libraries:

    * numpy

    * pandas

    * scikit-learn

    * joblib

    * matplotlib

    * seaborn

    * geopy

    * openrouteservice

    * folium

## License

* Code: Academic / educational use

* Dataset & models: CC BY-NC 4.0 (via Kaggle)

    * Attribution required

    * Non-commercial use only

## Acknowledgments

Microsoft Research Asia — GeoLife GPS Trajectory Dataset

Beijing Institute of Technology — academic support

OpenStreetMap & OpenRouteService — geospatial services

## Authors

**Runar Kenzhekeyev**

**Kazhybek Asset**