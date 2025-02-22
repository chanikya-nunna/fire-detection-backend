import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import shap
import smogn  # For regression oversampling

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_validate
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import RFE
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# ✅ Load Dataset
df = pd.read_csv("forestfires.csv")
print("✅ Dataset loaded successfully!")

# ✅ Encode Categorical Features
df['month'] = df['month'].astype('category').cat.codes
df['day'] = df['day'].astype('category').cat.codes

# ✅ Feature Scaling
num_features = ['X', 'Y', 'FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind', 'rain', 'month', 'day']
scaler = StandardScaler()
df[num_features] = scaler.fit_transform(df[num_features])

# ✅ Transform Target Variable
df['area_cbrt'] = np.cbrt(df['area'])

# ✅ Apply SMOGN for Oversampling
df_resampled = smogn.smoter(data=df, y='area_cbrt', k=7, samp_method='extreme')

# ✅ Splitting Data
X = df_resampled[num_features]  # Only use numerical features
y = df_resampled['area_cbrt']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Feature Selection with RFE
selector = RFE(GradientBoostingRegressor(n_estimators=50, random_state=42), n_features_to_select=8)
selector.fit(X_train, y_train)
selected_features = X_train.columns[selector.support_]

# ✅ Filter only selected features
X_train = X_train[selected_features]
X_test = X_test[selected_features]

# ✅ Train Random Forest Regressor
reg = RandomForestRegressor(n_estimators=100, random_state=42)
reg.fit(X_train, y_train)

# ✅ Model Evaluation
reg_mse = mean_squared_error(y_test, reg.predict(X_test))
reg_r2 = r2_score(y_test, reg.predict(X_test))
print(f"🔥 Random Forest MSE: {reg_mse:.4f}")
print(f"🔥 Random Forest R² Score: {reg_r2:.4f}")

# ✅ Save Best Model and Feature Names
joblib.dump(reg, "fire_detection_model.pkl")
joblib.dump(selected_features.tolist(), "selected_features.pkl")
print("✅ Model and selected features saved successfully!")
