# geolife.py
import os, json, argparse
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Viz & IO
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import folium

# Geocoding & Routing
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
import openrouteservice
from openrouteservice.exceptions import ApiError

# Sparse linear algebra for Laplacian smoothing
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# ML (for training)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve, confusion_matrix, brier_score_loss
)
from sklearn.calibration import calibration_curve

import time

def tic(msg):
    print(f"\n▶ {msg} ...", flush=True)
    return time.perf_counter()

def toc(t0, tag=""):
    dt = time.perf_counter() - t0
    print(f"✓ {tag or 'done'} in {dt:.1f}s", flush=True)
    return dt

# Helpers: data load, filtering, grids, features
def excel_date_to_datetime(excel_date):
    return datetime(1899, 12, 30) + timedelta(days=float(excel_date))

def load_trajectory_data(file_path):
    data_dict = pd.read_pickle(file_path)
    all_data = []
    for user_id, trajectories in data_dict.items():
        for trajectory in trajectories:
            for point in trajectory:
                lat, lon, timestamp = point
                dt = excel_date_to_datetime(timestamp)
                all_data.append({
                    'UserID': user_id,
                    'Latitude': float(lat),
                    'Longitude': float(lon),
                    'Timestamp': dt,
                    'Hour': dt.hour,
                    'Date': dt.date(),
                    'Month': dt.month,
                    'Year': dt.year
                })
    return pd.DataFrame(all_data)

def apply_filters(df, lat_min, lat_max, lon_min, lon_max, start_datetime, end_datetime):
    return df[
        (df['Latitude'] >= lat_min) & (df['Latitude'] <= lat_max) &
        (df['Longitude'] >= lon_min) & (df['Longitude'] <= lon_max) &
        (df['Timestamp'] >= start_datetime) & (df['Timestamp'] <= end_datetime)
    ].copy()

def build_congestion_dataset(df, grid_size=0.01, min_points_for_congestion=100):
    df = df.copy()
    df['GridLat'] = (df['Latitude'] // grid_size).astype(float) * grid_size
    df['GridLon'] = (df['Longitude'] // grid_size).astype(float) * grid_size
    df['Hour'] = df['Timestamp'].dt.hour
    df['DateHour'] = df['Timestamp'].dt.floor('h')

    grouped = df.groupby(['GridLat', 'GridLon', 'DateHour']).size().reset_index(name='PointCount')
    grouped['Congested'] = (grouped['PointCount'] > min_points_for_congestion).astype(int)

    df = df.merge(grouped, on=['GridLat', 'GridLon', 'DateHour'], how='left')
    df['Congested'] = df['Congested'].fillna(0).astype(int)
    return df

def generate_visual_insights(df):
    print("Generating visual insights...")
    sns.set(style="whitegrid")

    plt.figure(figsize=(12, 6))
    sns.countplot(data=df, x='Month')
    plt.title("Monthly Data Distribution")
    plt.savefig("monthly_distribution.png"); print("Saved: monthly_distribution.png")

    plt.figure(figsize=(12, 6))
    sns.countplot(data=df, x='Hour')
    plt.title("Hourly Activity")
    plt.savefig("hourly_activity.png"); print("Saved: hourly_activity.png")

    plt.figure(figsize=(12, 6))
    sns.countplot(data=df, x='Hour', hue='Congested')
    plt.title("Congestion by Hour")
    plt.legend(title="Congested")
    plt.savefig("congestion_by_hour.png"); print("Saved: congestion_by_hour.png")

    plt.figure(figsize=(8, 6))
    sample_n = min(10000, len(df))
    sns.scatterplot(
        x='Longitude', y='Latitude', hue='Congested',
        data=df.sample(n=sample_n, random_state=42),
        alpha=0.5
    )
    plt.title("Spatial Congestion Map (sampled)")
    plt.savefig("spatial_congestion.png"); print("Saved: spatial_congestion.png")

    plt.figure(figsize=(10, 6))
    weekday_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    df['Weekday'] = df['Timestamp'].dt.weekday
    sns.countplot(x='Weekday', data=df)
    plt.xticks(ticks=range(7), labels=weekday_labels)
    plt.title("Activity per Weekday"); plt.xlabel("Weekday"); plt.ylabel("Point Count")
    plt.savefig("weekday_activity.png"); print("Saved: weekday_activity.png")

    plt.figure(figsize=(12, 6))
    time_series = df.groupby('DateHour')['Congested'].sum()
    time_series.rolling(window=24).mean().plot()
    plt.title("Smoothed Congestion Over Time (Rolling 24h avg)")
    plt.xlabel("Time"); plt.ylabel("Number of Congested Points")
    plt.grid(True); plt.tight_layout()
    plt.savefig("congestion_over_time.png"); print("Saved: congestion_over_time.png")

    pivot = df.pivot_table(index=df['Timestamp'].dt.weekday,
                           columns='Hour',
                           values='UserID',
                           aggfunc='count').fillna(0)
    plt.figure(figsize=(14, 6))
    sns.heatmap(pivot, cmap="YlOrRd", linewidths=0.1)
    plt.title("Heatmap of Activity: Weekday vs Hour")
    plt.ylabel("Weekday (0=Mon)"); plt.xlabel("Hour")
    plt.savefig("weekday_hour_heatmap.png"); print("Saved: weekday_hour_heatmap.png")

# Features for Logistic (engineered features)
def _cyc_enc(x, period):
    x = np.asarray(x, dtype=float)
    ang = 2.0 * np.pi * (x / period)
    return np.sin(ang), np.cos(ang)

def make_features_numpy(lat, lon, hour, month, weekday):
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    hour = np.asarray(hour, dtype=float)
    month = np.asarray(month, dtype=float)
    weekday = np.asarray(weekday, dtype=float)

    # Cyclical encodings
    h_s, h_c = _cyc_enc(hour, 24.0)
    m_s, m_c = _cyc_enc(month - 1.0, 12.0)  # months 1..12 -> 0..11
    w_s, w_c = _cyc_enc(weekday, 7.0)

    # Spatial polynomial terms
    lat2 = lat * lat
    lon2 = lon * lon
    latlon = lat * lon

    # Cross terms with time (subset)
    lat_hs = lat * h_s
    lon_hs = lon * h_s
    lat_hc = lat * h_c
    lon_hc = lon * h_c

    feats = np.column_stack([
        lat, lon, lat2, lon2, latlon,
        h_s, h_c, m_s, m_c, w_s, w_c,
        lat_hs, lon_hs, lat_hc, lon_hc
    ])
    return feats

def dataframe_to_features(df_like):
    return make_features_numpy(
        df_like['Latitude'].to_numpy(),
        df_like['Longitude'].to_numpy(),
        df_like['Hour'].to_numpy(),
        df_like['Month'].to_numpy(),
        df_like['WeekdayNum'].to_numpy()
    )

# Custom Logistic Regression via IRLS (now trainable)
class Standardizer:
    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_[self.std_ == 0] = 1.0
        return self
    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return (X - self.mean_) / self.std_
    def fit_transform(self, X):
        return self.fit(X).transform(X)

def _sigmoid(z):
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    pos = z >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[neg])
    out[neg] = ez / (1.0 + ez)
    return out

class LogisticRegressionIRLSNP:
    def __init__(self, reg_lambda=1.0, max_iter=50, tol=1e-6, class_weight=None, verbose=False):
        self.reg_lambda = float(reg_lambda)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.class_weight = class_weight
        self.verbose = verbose

    def _add_bias(self, X):
        return np.hstack([np.ones((X.shape[0], 1), dtype=float), X])

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int).ravel()
        n, p = X.shape

        self.scaler_ = Standardizer().fit(X)
        Xs = self.scaler_.transform(X)
        Xb = self._add_bias(Xs)

        self.w_ = np.zeros(p + 1, dtype=float)

        if self.class_weight is None:
            w_class0 = 1.0; w_class1 = 1.0
        else:
            w_class0 = float(self.class_weight.get(0, 1.0))
            w_class1 = float(self.class_weight.get(1, 1.0))

        lam = self.reg_lambda

        for it in range(self.max_iter):
            z = Xb @ self.w_
            p_hat = _sigmoid(z)

            sample_w = np.where(y == 1, w_class1, w_class0)

            r = (p_hat - y) * sample_w
            grad = Xb.T @ r
            w_tilde = self.w_.copy(); w_tilde[0] = 0.0
            grad += lam * w_tilde

            W = p_hat * (1.0 - p_hat) * sample_w
            Xb_weighted = Xb * W[:, None]
            H = Xb.T @ Xb_weighted
            R = np.eye(p + 1); R[0, 0] = 0.0
            H += lam * R

            try:
                delta = np.linalg.solve(H, grad)
            except np.linalg.LinAlgError:
                H += 1e-6 * np.eye(p + 1)
                delta = np.linalg.solve(H, grad)

            self.w_ = self.w_ - delta
            step_norm = np.linalg.norm(delta)

            if self.verbose:
                eps = 1e-12
                ll = np.sum(sample_w * (y * np.log(p_hat + eps) + (1 - y) * np.log(1 - p_hat + eps)))
                reg = 0.5 * lam * np.dot(w_tilde, w_tilde)
                obj = -(ll - reg)
                print(f"Iter {it+1:02d}: obj={obj:.4f}, step={step_norm:.3e}")

            if step_norm < self.tol:
                break

        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        Xs = self.scaler_.transform(X)
        Xb = self._add_bias(Xs)
        z = Xb @ self.w_
        p = _sigmoid(z)
        return np.column_stack([1.0 - p, p])

    def predict(self, X, threshold=0.5):
        proba = self.predict_proba(X)[:, 1]
        return (proba >= threshold).astype(int)

# Training utilities (integrated)
def _compute_class_weights(y: np.ndarray):
    y = np.asarray(y, dtype=int)
    pos = int((y == 1).sum())
    neg = int((y == 0).sum())
    if pos == 0 or neg == 0:
        return {0: 1.0, 1: 1.0}
    tot = pos + neg
    return {0: 0.5 * tot / neg, 1: 0.5 * tot / pos}

def _plot_roc_pr(y_true, p_rf, p_lr, out_dir="model_reports"):
    os.makedirs(out_dir, exist_ok=True)
    # ROC
    plt.figure(figsize=(6,5))
    for name, p in [("RF", p_rf), ("LogReg", p_lr)]:
        fpr, tpr, _ = roc_curve(y_true, p)
        auc = roc_auc_score(y_true, p)
        plt.plot(fpr, tpr, label=f'{name} (AUC={auc:.3f})')
    plt.plot([0,1], [0,1], 'k--', lw=1)
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC"); plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, "roc.png")); plt.close()

    # PR
    plt.figure(figsize=(6,5))
    base = float((np.asarray(y_true)==1).mean())
    for name, p in [("RF", p_rf), ("LogReg", p_lr)]:
        prec, rec, _ = precision_recall_curve(y_true, p)
        ap = average_precision_score(y_true, p)
        plt.plot(rec, prec, label=f'{name} (AP={ap:.3f})')
    plt.hlines(base, 0, 1, colors="gray", linestyles="--", label=f"Baseline={base:.3f}")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title("Precision–Recall"); plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, "pr.png")); plt.close()

def _plot_confusions(y_true, yhat_rf, yhat_lr, out_dir="model_reports"):
    os.makedirs(out_dir, exist_ok=True)
    for name, yhat in [("rf", yhat_rf), ("logreg", yhat_lr)]:
        cm = confusion_matrix(y_true, yhat, labels=[0,1])
        plt.figure(figsize=(4,4))
        plt.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                plt.text(j, i, str(cm[i, j]),
                         ha="center", va="center",
                         color="white" if cm[i, j] > cm.max()/2 else "black")
        plt.xticks([0,1], ["Pred 0","Pred 1"])
        plt.yticks([0,1], ["True 0","True 1"])
        plt.title(f"Confusion Matrix: {name.upper()}")
        plt.tight_layout(); plt.savefig(os.path.join(out_dir, f"cm_{name}.png")); plt.close()

def _plot_calibration(y_true, p_rf, p_lr, out_dir="model_reports", n_bins=12):
    os.makedirs(out_dir, exist_ok=True)
    plt.figure(figsize=(6,5))
    for name, p in [("RF", p_rf), ("LogReg", p_lr)]:
        prob_true, prob_pred = calibration_curve(y_true, p, n_bins=n_bins, strategy='uniform')
        plt.plot(prob_pred, prob_true, marker='o', label=name)
    plt.plot([0,1], [0,1], 'k--', lw=1, label="Perfect")
    plt.xlabel("Predicted probability"); plt.ylabel("Observed frequency")
    plt.title("Calibration (Reliability) Curve"); plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, "calibration.png")); plt.close()

def _plot_rf_importance(rf: RandomForestClassifier, out_dir="model_reports", top_k=15):
    os.makedirs(out_dir, exist_ok=True)
    names = ['Latitude','Longitude','Hour','Month','WeekdayNum']
    imp = rf.feature_importances_
    order = np.argsort(imp)[::-1][:top_k]
    plt.figure(figsize=(6,5))
    plt.barh([names[i] for i in order][::-1], imp[order][::-1])
    plt.title("Random Forest Feature Importance")
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, "rf_feature_importance.png")); plt.close()

def _plot_lr_coefficients(lr: "LogisticRegressionIRLSNP", out_dir="model_reports", top_k=15):
    os.makedirs(out_dir, exist_ok=True)
    names = [
        "lat","lon","lat^2","lon^2","lat*lon",
        "sin(hour)","cos(hour)","sin(month)","cos(month)","sin(weekday)","cos(weekday)",
        "lat*sin(hour)","lon*sin(hour)","lat*cos(hour)","lon*cos(hour)"
    ]
    coef = lr.w_[1:]  # exclude bias
    order = np.argsort(np.abs(coef))[::-1][:top_k]
    plt.figure(figsize=(7,5))
    plt.barh([names[i] for i in order][::-1], coef[order][::-1])
    plt.title("Logistic Coefficients (standardized features)")
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, "lr_coefficients.png")); plt.close()

def _plot_prob_hist(p_rf, p_lr, out_dir="model_reports"):
    os.makedirs(out_dir, exist_ok=True)
    plt.figure(figsize=(6,5))
    plt.hist(p_rf, bins=30, alpha=0.6, label="RF")
    plt.hist(p_lr, bins=30, alpha=0.6, label="LogReg")
    plt.xlabel("Predicted probability"); plt.ylabel("Count")
    plt.title("Predicted Probability Distribution")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "prob_hist.png")); plt.close()

def train_or_load_both_models(
    df_cong: pd.DataFrame,
    rf_path: str = "rf_model.pkl",
    lr_path: str = "logreg_model.pkl",
    out_dir: str = "model_reports",
    test_size: float = 0.2,
    random_state: int = 42,
    use_all: bool = True,              # <-- train on ALL rows by default
    rf_n_estimators: int = 500,        # same as before
):
    """
    Train/load RandomForest + IRLS-Logistic on a shared held-out split.
    Saves metrics & plots to `out_dir`, returns (rf, lr).
    """
    os.makedirs(out_dir, exist_ok=True)

    work = df_cong.copy()
    work['WeekdayNum'] = work['Timestamp'].dt.weekday
    base_cols = ['Latitude','Longitude','Hour','Month','WeekdayNum','Congested']
    dfm = work[base_cols].dropna().drop_duplicates()

    # no cap unless you set use_all=False
    if not use_all:
        cap = 2_000_000
        if len(dfm) > cap:
            dfm = dfm.sample(n=cap, random_state=random_state)

    print(f"[trainer] Rows used for supervised learning: {len(dfm):,}")

    X_rf_full = dfm[['Latitude','Longitude','Hour','Month','WeekdayNum']].values
    y_full     = dfm['Congested'].to_numpy(dtype=int)
    X_lr_full  = dataframe_to_features(dfm)

    # shared split
    n = len(dfm)
    rng = np.random.default_rng(random_state)
    perm = rng.permutation(n)
    n_test   = max(1, int(test_size * n))
    test_idx = perm[:n_test]
    train_idx= perm[n_test:]
    print(f"[trainer] Split: train={n - n_test:,}, test={n_test:,} (test_size={test_size})")

    Xtr_rf, Xte_rf = X_rf_full[train_idx], X_rf_full[test_idx]
    Xtr_lr, Xte_lr = X_lr_full[train_idx], X_lr_full[test_idx]
    ytr,    yte    = y_full[train_idx],    y_full[test_idx]

    # Random Forest
    if os.path.exists(rf_path):
        print(f"[trainer] Loading RF from {rf_path}...")
        rf = joblib.load(rf_path)
    else:
        print("[trainer] Training RandomForest…")
        rf = RandomForestClassifier(
            n_estimators=rf_n_estimators,
            min_samples_leaf=10,
            class_weight='balanced',
            n_jobs=-1,
            random_state=random_state
        )
        t0 = tic(f"Fitting RandomForest (n_estimators={rf_n_estimators}, min_samples_leaf=10)")
        rf.fit(Xtr_rf, ytr)
        toc(t0, "RF fit")
        joblib.dump(rf, rf_path)
        print(f"[trainer] Saved RF to {rf_path}")

    # Logistic (IRLS)
    if os.path.exists(lr_path):
        print(f"[trainer] Loading LogReg from {lr_path}…")
        lr = joblib.load(lr_path)
    else:
        print("[trainer] Training Logistic (IRLS)…")
        class_weight = _compute_class_weights(ytr)
        lr = LogisticRegressionIRLSNP(
            reg_lambda=5.0, max_iter=60, tol=1e-6,
            class_weight=class_weight, verbose=False
        )
        t0 = tic("Fitting IRLS Logistic (max_iter=60, reg_lambda=5.0)")
        lr.fit(Xtr_lr, ytr)
        toc(t0, "IRLS fit")
        joblib.dump(lr, lr_path)
        print(f"[trainer] Saved LogReg to {lr_path}")

    # Evaluation & reports
    p_rf = rf.predict_proba(Xte_rf)[:, 1]
    p_lr = lr.predict_proba(Xte_lr)[:, 1]
    thr = 0.5
    yhat_rf = (p_rf >= thr).astype(int)
    yhat_lr = (p_lr >= thr).astype(int)

    def _safe_roc_auc(y, p):
        try: return float(roc_auc_score(y, p))
        except ValueError: return float('nan')
    def _safe_ap(y, p):
        try: return float(average_precision_score(y, p))
        except ValueError: return float('nan')

    metrics = {
        "samples": int(n),
        "prevalence": float((yte == 1).mean()),
        "rf": {
            "acc": float(accuracy_score(yte, yhat_rf)),
            "roc_auc": _safe_roc_auc(yte, p_rf),
            "pr_auc": _safe_ap(yte, p_rf),
            "brier": float(brier_score_loss(yte, p_rf)),
        },
        "logreg": {
            "acc": float(accuracy_score(yte, yhat_lr)),
            "roc_auc": _safe_roc_auc(yte, p_lr),
            "pr_auc": _safe_ap(yte, p_lr),
            "brier": float(brier_score_loss(yte, p_lr)),
        }
    }
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"[trainer] Saved metrics to {os.path.join(out_dir, 'metrics.json')}")

    _plot_roc_pr(yte, p_rf, p_lr, out_dir=out_dir)
    _plot_confusions(yte, yhat_rf, yhat_lr, out_dir=out_dir)
    _plot_calibration(yte, p_rf, p_lr, out_dir=out_dir, n_bins=12)
    _plot_rf_importance(rf, out_dir=out_dir)
    _plot_lr_coefficients(lr, out_dir=out_dir)
    _plot_prob_hist(p_rf, p_lr, out_dir=out_dir)
    print(f"[trainer] Reports saved in '{out_dir}'")
    return rf, lr

# Graph Laplacian on a rectangular grid
def laplacian_from_grid(h, w):
    """4-neighbor Laplacian on an h x w grid."""
    N = h * w
    rows, cols, data = [], [], []
    def idx(r,c): return r*w + c

    for r in range(h):
        for c in range(w):
            i = idx(r,c)
            deg = 0
            for rr,cc in ((r-1,c), (r+1,c), (r,c-1), (r,c+1)):
                if 0 <= rr < h and 0 <= cc < w:
                    j = idx(rr,cc)
                    rows.append(i); cols.append(j); data.append(-1.0)
                    deg += 1
            rows.append(i); cols.append(i); data.append(float(deg))
    return sp.csr_matrix((data,(rows,cols)), shape=(N,N))

def laplacian_smooth_field(prob_grid, lam=0.5):
    """
    Minimize ||u - f||^2 + lam * u^T L u  => (I + lam L) u = f
    prob_grid: (h, w) array of probabilities in [0,1]
    """
    h, w = prob_grid.shape
    L = laplacian_from_grid(h, w)
    A = sp.eye(h*w, format='csr') + lam * L
    b = prob_grid.ravel()
    u = spla.spsolve(A, b)
    u = np.clip(u, 0.0, 1.0)
    return u.reshape(h, w)

# Geocode / Route helper
def geocode_address(address):
    geolocator = Nominatim(user_agent="congestion_predictor")
    try:
        location = geolocator.geocode(address, timeout=10)
        if location:
            print(f"{address} -> Coordinates: {location.latitude}, {location.longitude}")
            return location.latitude, location.longitude
        else:
            print(f"Failed to locate: {address}")
            return None
    except GeocoderTimedOut:
        print(f"Geocoder timed out for: {address}")
        return None

def get_route_from_ors(start_coords, end_coords, api_key, mode="foot-walking"):
    try:
        client = openrouteservice.Client(key=api_key)
        route = client.directions(
            coordinates=[(start_coords[1], start_coords[0]), (end_coords[1], end_coords[0])],
            profile=mode,
            format='geojson'
        )
        coords = [(pt[1], pt[0]) for pt in route['features'][0]['geometry']['coordinates']]
        return coords
    except ApiError as e:
        print(f"OpenRouteService error: {e}")
        return []

# Build a grid, predict prob field, smooth, sample route
def build_grid_axes(lat_min, lat_max, lon_min, lon_max, grid_size=0.01):
    lat_vals = np.arange(lat_min, lat_max + 1e-12, grid_size, dtype=float)
    lon_vals = np.arange(lon_min, lon_max + 1e-12, grid_size, dtype=float)
    return lat_vals, lon_vals

def blend_probabilities(p_rf, p_lr, alpha=0.5):
    return np.clip(alpha * p_rf + (1.0 - alpha) * p_lr, 0.0, 1.0)

def predict_prob_field_for_time(rf, lr, lat_vals, lon_vals, dt, alpha=0.5, batch=200000):
    H, W = len(lat_vals), len(lon_vals)
    LATS, LONS = np.meshgrid(lat_vals, lon_vals, indexing='ij')
    lat_flat = LATS.ravel()
    lon_flat = LONS.ravel()

    df_rf = pd.DataFrame({
        'Latitude': lat_flat,
        'Longitude': lon_flat,
        'Hour': dt.hour,
        'Month': dt.month,
        'WeekdayNum': dt.weekday()
    })
    X_rf = df_rf[['Latitude','Longitude','Hour','Month','WeekdayNum']].values

    df_lr = df_rf.copy()
    X_lr = dataframe_to_features(df_lr)

    prf = np.zeros(len(lat_flat), dtype=float)
    plr = np.zeros(len(lat_flat), dtype=float)

    for start in range(0, len(lat_flat), batch):
        end = min(start+batch, len(lat_flat))
        prf[start:end] = rf.predict_proba(X_rf[start:end])[:,1]
        plr[start:end] = lr.predict_proba(X_lr[start:end])[:,1]

    p = blend_probabilities(prf, plr, alpha=alpha)
    return p.reshape(H, W)

def nearest_index(val_array, x):
    idx = np.searchsorted(val_array, x)
    if idx <= 0: return 0
    if idx >= len(val_array): return len(val_array)-1
    if abs(val_array[idx]-x) < abs(val_array[idx-1]-x): return idx
    return idx-1

def sample_route_from_field(prob_grid, lat_vals, lon_vals, route_coords):
    samples = []
    for (lat, lon) in route_coords:
        i = nearest_index(lat_vals, lat)
        j = nearest_index(lon_vals, lon)
        samples.append(prob_grid[i, j])
    return np.array(samples, dtype=float)

# Main route prediction (with Laplacian smoothing)
def predict_congestion_by_address_ensemble(rf, lr, alpha, api_key,
                                           lat_vals, lon_vals,
                                           lam=0.5, threshold=0.5,
                                           grid_size=0.01):
    print("\nEnter route as: Place1, Place2, YYYY-MM-DD HH:MM")
    print("Example: Beijing Institute of Technology, Forbidden City, 2012-06-21 09:00")
    try:
        line = input("> ").strip()
        if not line:
            print("Empty input.")
            return
        addr1, addr2, time_str = map(str.strip, line.split(",", 2))
        dt = datetime.strptime(time_str, "%Y-%m-%d %H:%M")

        coord1 = geocode_address(addr1)
        coord2 = geocode_address(addr2)
        if coord1 is None or coord2 is None:
            print("Failed to get coordinates.")
            return

        route_points = get_route_from_ors(coord1, coord2, api_key)
        if not route_points:
            print("No route found between the given addresses.")
            return

        print("Predicting probability field over grid...")
        prob_grid = predict_prob_field_for_time(rf, lr, lat_vals, lon_vals, dt, alpha=alpha)

        print("Applying Graph-Laplacian smoothing...")
        prob_grid_sm = laplacian_smooth_field(prob_grid, lam=lam)

        p_route = sample_route_from_field(prob_grid_sm, lat_vals, lon_vals, route_points)
        yhat = (p_route >= threshold).astype(int)

        route_map = folium.Map(location=route_points[0], zoom_start=13)
        for idx, ((lat, lon), p, yy) in enumerate(zip(route_points, p_route, yhat)):
            color = "red" if yy else "green"
            folium.CircleMarker(
                location=(lat, lon),
                radius=6, color=color, fill=True, fill_color=color, fill_opacity=0.7,
                popup=f"Point {idx+1}: P(cong)={p:.2f}"
            ).add_to(route_map)

        folium.PolyLine(route_points, color="blue", weight=2.5, opacity=0.6).add_to(route_map)
        route_map.save("route_map_ors.html")
        print("Route map saved as route_map_ors.html")

        congested_count = int(yhat.sum())
        if   congested_count == 0: level = "Fully clear"
        elif congested_count < len(yhat)/3: level = "Mostly clear"
        elif congested_count < 2*len(yhat)/3: level = "Moderate congestion"
        else: level = "Heavily congested"
        print(f"Overall route congestion level: {level}  (alpha={alpha:.2f}, lam={lam:.2f}, thr={threshold:.2f})")
    except Exception as e:
        print(f"Error: {e}")


# simple EDA plot
def plot_yearly_activity(df):
    yearly_counts = df.groupby(df['Timestamp'].dt.year).size()
    yearly_counts = yearly_counts.loc[yearly_counts.index.isin(range(2007, 2013))]
    plt.figure(figsize=(10, 6))
    bars = plt.bar(yearly_counts.index.astype(str), yearly_counts.values, color='skyblue', edgecolor='black')
    plt.title("Yearly Trajectory Activity (2007–2012)", fontsize=16)
    plt.xlabel("Year", fontsize=14); plt.ylabel("Number of Data Points", fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 1000, f"{yval:,}", ha='center', va='bottom', fontsize=10)
    plt.tight_layout(); plt.savefig("yearly_activity.png"); plt.show()
    print("Saved as 'yearly_activity.png'")

# main
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Geolife congestion trainer + predictor")
    parser.add_argument("--data", default="Beijing_user_trajectory.pkl")
    parser.add_argument("--lat-min", type=float, default=39.5)
    parser.add_argument("--lat-max", type=float, default=40.1)
    parser.add_argument("--lon-min", type=float, default=115.9)
    parser.add_argument("--lon-max", type=float, default=116.7)
    parser.add_argument("--start",   default="2007-04-01")
    parser.add_argument("--end",     default="2012-08-31 23:59")
    parser.add_argument("--grid-size", type=float, default=0.01)
    parser.add_argument("--min-points", type=int, default=100)
    parser.add_argument("--rf-path", default="rf_model.pkl")
    parser.add_argument("--lr-path", default="logreg_model.pkl")
    parser.add_argument("--reports", default="model_reports")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--retrain", action="store_true", help="Force retraining models")
    parser.add_argument("--alpha", type=float, default=0.5, help="RF/LR blend weight")
    parser.add_argument("--lam", type=float, default=0.5, help="Laplacian smoothing lambda")
    parser.add_argument("--thr", type=float, default=0.5, help="Decision threshold")
    parser.add_argument("--ors-key", default="5b3ce3597851110001cf6248597b5c09ec034530b20e35d1edadc320")
    args = parser.parse_args()

    # Load & prepare
    if not os.path.exists(args.data):
        raise FileNotFoundError(f"Data file not found: {args.data}")
    df = load_trajectory_data(args.data)
    print(f"Loaded data points: {len(df):,}")

    start_dt = datetime.fromisoformat(args.start)
    end_dt   = datetime.fromisoformat(args.end)
    filtered_df = apply_filters(df, args.lat_min, args.lat_max, args.lon_min, args.lon_max, start_dt, end_dt)
    print(f"After filtering: {len(filtered_df):,} points.")
    plot_yearly_activity(filtered_df)

    df_cong = build_congestion_dataset(filtered_df, grid_size=args.grid_size,
                                       min_points_for_congestion=args.min_points)
    generate_visual_insights(df_cong)

    # Train or load models
    need_train = args.retrain or not (os.path.exists(args.rf_path) and os.path.exists(args.lr_path))
    if need_train:
        print("[main] Training models (either forced or missing artifacts)…")
        rf, lr = train_or_load_both_models(
            df_cong=df_cong,
            rf_path=args.rf_path,
            lr_path=args.lr_path,
            out_dir=args.reports,
            test_size=args.test_size,
            random_state=args.seed,
            use_all=True,
            rf_n_estimators=500
        )

    else:
        print("[main] Loading existing models…")
        rf = joblib.load(args.rf_path)
        lr = joblib.load(args.lr_path)

    # Build grid axes for probability field & smoothing
    lat_axis, lon_axis = build_grid_axes(args.lat_min, args.lat_max, args.lon_min, args.lon_max, grid_size=args.grid_size)

    # Interactive route prediction with spatial smoothing
    predict_congestion_by_address_ensemble(
        rf=rf, lr=lr, alpha=args.alpha, api_key=args.ors_key,
        lat_vals=lat_axis, lon_vals=lon_axis,
        lam=args.lam, threshold=args.thr,
        grid_size=args.grid_size
    )
