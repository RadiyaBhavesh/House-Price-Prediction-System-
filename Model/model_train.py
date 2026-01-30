import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# =============================
# 1. LOAD DATA
# =============================
data = pd.read_csv("../dataset/house_data.csv")
print("✅ Dataset Loaded")

# =============================
# 2. DATA CLEANING
# =============================
data.dropna(inplace=True)
data.drop_duplicates(inplace=True)

# Remove price outliers (IQR method)
Q1 = data["Price"].quantile(0.25)
Q3 = data["Price"].quantile(0.75)
IQR = Q3 - Q1
data = data[(data["Price"] >= Q1 - 1.5 * IQR) &
            (data["Price"] <= Q3 + 1.5 * IQR)]

# =============================
# 3. FEATURE ENGINEERING
# =============================
le = LabelEncoder()
data["Location"] = le.fit_transform(data["Location"])

# Log transform price (industry standard)
data["Price"] = np.log1p(data["Price"])

X = data[["Area", "BHK", "Bathroom", "Location"]]
y = data["Price"]

# =============================
# 4. TRAIN TEST SPLIT
# =============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =============================
# 5. LINEAR REGRESSION PIPELINE
# =============================
lr_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("lr", LinearRegression())
])

lr_pipeline.fit(X_train, y_train)

# =============================
# 6. RANDOM FOREST MODEL
# =============================
rf_model = RandomForestRegressor(
    n_estimators=300,
    max_depth=20,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

# =============================
# 7. HYBRID PREDICTION (WEIGHTED)
# =============================
lr_pred = lr_pipeline.predict(X_test)
rf_pred = rf_model.predict(X_test)

# Weighted hybrid (RF stronger)
hybrid_pred = (0.4 * lr_pred) + (0.6 * rf_pred)

# =============================
# 8. EVALUATION (REAL ₹ SCALE)
# =============================
y_test_real = np.expm1(y_test)
hybrid_real = np.expm1(hybrid_pred)

r2 = r2_score(y_test, hybrid_pred)
mae = mean_absolute_error(y_test_real, hybrid_real)
rmse = np.sqrt(mean_squared_error(y_test_real, hybrid_real))

print("\n📊 HYBRID MODEL PERFORMANCE")
print("R2 Score        :", round(r2, 3))
print("MAE (₹)         :", round(mae, 2))
print("RMSE (₹)        :", round(rmse, 2))

# =============================
# 9. CROSS VALIDATION
# =============================
cv_scores = cross_val_score(
    lr_pipeline, X, y, cv=5, scoring="r2"
)
print("CV R2 Avg       :", round(cv_scores.mean(), 3))

# =============================
# 10. SAVE MODELS
# =============================
pickle.dump(lr_pipeline, open("linear_model.pkl", "wb"))
pickle.dump(rf_model, open("rf_model.pkl", "wb"))
pickle.dump(le, open("location_encoder.pkl", "wb"))

print("\n✅ Hybrid models & encoder saved successfully")
