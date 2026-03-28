import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import joblib 

print("Loading the 3-variable dataset...")
df = pd.read_csv('data/projectile_dataset.csv')

# We now have 3 inputs (Features) instead of 2
X = df[['Velocity_mps', 'Angle_degrees', 'Initial_Height_m']]
y = df[['Flight_Time_s', 'Max_Height_m', 'Distance_m']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\n🏆 --- ALGORITHM SHOWDOWN --- 🏆")

# 1. Train and test Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_predictions = lr_model.predict(X_test)
lr_accuracy = r2_score(y_test, lr_predictions)
print(f"Linear Regression Accuracy: {lr_accuracy * 100:.2f}% (Struggles with curves)")

# 2. Train and test Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)
rf_accuracy = r2_score(y_test, rf_predictions)
print(f"Random Forest Accuracy:     {rf_accuracy * 100:.2f}% (Excels at complex math)")
print("--------------------------------\n")

# Save the winning model
joblib.dump(rf_model, 'physics_model.pkl')
print("Winner saved as 'physics_model.pkl' for the Web App to use.")