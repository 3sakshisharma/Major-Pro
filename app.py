# app.py
import os
import joblib
import pandas as pd
from flask import Flask, render_template, request

BASE = os.path.dirname(__file__) or "."
MODEL_PATH = os.path.join(BASE, "house_price_model.pkl")
DATA_PATH = os.path.join(BASE, "house_data.csv")

app = Flask(__name__)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Model not found. Run train_model.py first to create house_price_model.pkl")

model = joblib.load(MODEL_PATH)
df = pd.read_csv(DATA_PATH)
df.columns = [c.strip() for c in df.columns]

def unique_sorted(col):
    return sorted(df[col].dropna().unique().tolist())

@app.route("/")
def home():
    cities = unique_sorted('City')
    types = unique_sorted('PropertyType')
    return render_template("index.html", cities=cities, types=types)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        city = request.form.get("City","").strip()
        areaname = request.form.get("AreaName","").strip()
        areasqft = float(request.form.get("AreaSqft","0"))
        bedrooms = int(request.form.get("Bedrooms","0"))
        ptype = request.form.get("PropertyType","").strip()
    except Exception as e:
        return render_template("index.html", error=f"Invalid input: {e}", cities=unique_sorted('City'), types=unique_sorted('PropertyType'))

    X_input = pd.DataFrame([{
        'City': city,
        'AreaName': areaname,
        'AreaSqft': areasqft,
        'Bedrooms': bedrooms,
        'PropertyType': ptype
    }])

    try:
        pred = model.predict(X_input)[0]
    except Exception as e:
        return render_template("index.html", error=f"Prediction error: {e}", cities=unique_sorted('City'), types=unique_sorted('PropertyType'))

    # Neighborhood score (simple heuristic)
    neighborhood_score = round(min(100, (areasqft/1500)*50 + bedrooms*12), 2)

    # Suggestions (top 3 similar in same city & type)
    similar = df[(df['City']==city) & (df['PropertyType']==ptype)].copy()
    suggestions = []
    if len(similar) > 0:
        similar['PriceDiff'] = (similar['Price'] - pred).abs()
        top = similar.sort_values('PriceDiff').head(3)
        suggestions = top[['AreaName','AreaSqft','Bedrooms','Price']].to_dict(orient='records')

    pred_text = f"Predicted Price: ₹{pred:,.0f}"
    return render_template("index.html",
                           prediction_text=pred_text,
                           neighborhood_score=neighborhood_score,
                           suggestions=suggestions,
                           cities=unique_sorted('City'),
                           types=unique_sorted('PropertyType'))

if __name__ == "__main__":
    app.run(debug=True)