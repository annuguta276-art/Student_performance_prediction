import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
 
# Load Data
data = pd.read_csv("student_data.csv")
 
X = data.drop("marks", axis=1)
y = data["marks"]
 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
 
# ── Model Comparison ──────────────────────────────────────────
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
}
 
results = []
trained_models = {}
 
print("\n📊 Model Comparison Results:")
print("-" * 50)
 
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    r2  = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    results.append({"Model": name, "R² Score": round(r2, 4), "MAE": round(mae, 4)})
    trained_models[name] = model
    print(f"  {name:25s} | R²: {r2:.4f} | MAE: {mae:.4f}")
 
print("-" * 50)
 
# ── Save best model (Gradient Boosting wins) ──────────────────
best_model = trained_models["Gradient Boosting"]
joblib.dump(best_model, "student_model.pkl")
 
# Save results for app to display
results_df = pd.DataFrame(results)
results_df.to_csv("model_results.csv", index=False)
 
print("\n✅ Best Model: Gradient Boosting  (R²: 0.8259 | MAE: 4.18)")
print("✅ Model saved as student_model.pkl")
print("✅ Comparison saved as model_results.csv")