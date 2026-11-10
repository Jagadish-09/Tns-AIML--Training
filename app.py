
from flask import Flask, request, jsonify
import pandas as pd
import joblib
from pathlib import Path

app = Flask(__name__)

# Project folder
BASE_DIR = Path(__file__).parent

# Load trained ML model
saved_data = joblib.load(BASE_DIR / "model.pkl")

model = saved_data["model"]
features = saved_data["features"]
category_maps = saved_data["category_maps"]

# Load store information
store_data = pd.read_csv(BASE_DIR / "store.csv")
store_data = store_data.set_index("Store")


from flask import render_template

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    try:
        # Get user input
        user_data = request.get_json()

        store_id = int(user_data["store"])
        date = pd.to_datetime(user_data["date"])

        # Check store availability
        if store_id not in store_data.index:
            return jsonify({"error": "Store ID not found"}), 400

        store = store_data.loc[store_id]

        # Create input features
        row = {
            "Store": store_id,
            "DayOfWeek": date.day_of_week + 1,
            "Promo": int(user_data.get("promo", 0)),
            "StateHoliday": str(user_data.get("state_holiday", "0")),
            "SchoolHoliday": int(user_data.get("school_holiday", 0)),
            "Year": date.year,
            "Month": date.month,
            "Day": date.day,
            "WeekOfYear": int(date.isocalendar().week),
            "StoreType": str(store["StoreType"]),
            "Assortment": str(store["Assortment"]),
            "CompetitionDistance": store["CompetitionDistance"],
            "Promo2": store["Promo2"]
        }

        # Convert categorical values
        for col in ["StateHoliday", "StoreType", "Assortment"]:
            value = str(row[col])
            mapping = category_maps[col]

            if value not in mapping:
                return jsonify({
                    "error": f"Unknown category for {col}: {value}"
                }), 400

            row[col] = mapping[value]

        # Prepare model input
        input_df = pd.DataFrame([row])[features]
        input_df = input_df.fillna(0)

        # Predict sales
        prediction = float(model.predict(input_df)[0])

        return jsonify({
            "store": store_id,
            "date": date.strftime("%Y-%m-%d"),
            "predicted_sales": round(max(0, prediction), 2),
            "message": "Sales prediction successful"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True)
