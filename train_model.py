
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Find dataset files
base = Path(__file__).parent

def find_file(filename):
    for folder in [base / "dataset", base]:
        path = folder / filename
        if path.exists():
            return path
    raise FileNotFoundError(f"Cannot find {filename}")

# 2. Load datasets
print("Loading datasets...")

train = pd.read_csv(find_file("train.csv"))
store = pd.read_csv(find_file("store.csv"))

# 3. Combine sales and store information
data = train.merge(store, on="Store", how="left")

# 4. Data preprocessing
data["Date"] = pd.to_datetime(data["Date"])

data = data[data["Open"] == 1].copy()

# 5. Extract date features
data["Year"] = data["Date"].dt.year
data["Month"] = data["Date"].dt.month
data["Day"] = data["Date"].dt.day
data["WeekOfYear"] = data["Date"].dt.isocalendar().week.astype(int)

# 6. Select input features
features = [
    "Store", "DayOfWeek", "Promo", "StateHoliday",
    "SchoolHoliday", "Year", "Month", "Day", "WeekOfYear",
    "StoreType", "Assortment", "CompetitionDistance",
    "Promo2"
]

X = data[features].copy()
y = data["Sales"]

# 7. Handle missing values
categorical = ["StateHoliday", "StoreType", "Assortment"]

category_maps = {}

for col in categorical:
    X[col] = X[col].fillna("Unknown").astype(str)
    categories = sorted(X[col].unique())
    category_maps[col] = {
        value: index for index, value in enumerate(categories)
    }
    X[col] = X[col].map(category_maps[col])

X = X.fillna(0)

# 8. Reduce training size for faster execution
if len(X) > 150000:
    X = X.sample(n=150000, random_state=42)
    y = y.loc[X.index]

print("Dataset ready:", X.shape)

# 9. Split training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 10. Train Random Forest model
print("Training Machine Learning model...")

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1,
    min_samples_leaf=2
)

model.fit(X_train, y_train)

# 11. Evaluate model
predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print("\nMODEL PERFORMANCE")
print("-------------------------")
print("MAE:", round(mae, 2))
print("RMSE:", round(rmse, 2))
print("R2 Score:", round(r2, 4))

# 12. Save trained model
joblib.dump(
    {
        "model": model,
        "features": features,
        "category_maps": category_maps
    },
    base / "model.pkl"
)

print("\nModel saved successfully as model.pkl!")
