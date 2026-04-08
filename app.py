

from flask import Flask, render_template, request
import pickle
import pandas as pd

from utils.neighborhood import get_score
from utils.suggestions import get_similar
from utils.facilities import nearby_facilities
from utils.email_service import send_email

app = Flask(__name__)

model, city_enc, area_enc, type_enc = pickle.load(open("model.pkl","rb"))

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        city = request.form['city'].strip().lower()
        area = request.form['area'].strip().lower()
        sqft = float(request.form['sqft'])
        bedroom = int(request.form['bedroom'])
        ptype = request.form['ptype'].strip().lower()
        email = request.form.get('email')

        city_e = city_enc.transform([city])[0]
        area_e = area_enc.transform([area])[0]
        type_e = type_enc.transform([ptype])[0]

        features = pd.DataFrame([[city_e, area_e, sqft, bedroom, type_e]],
        columns=['city','areaname','areasqft','bedrooms','propertytype'])

        price = int(model.predict(features)[0])
        price = max(price, 0)

        # 🔥 NEW FEATURES
        low = int(price * 0.9)
        high = int(price * 1.1)

        score = get_score(city, area)
        confidence = score * 10

        suggestions = get_similar(city, area, sqft)
        facilities = nearby_facilities(area)

        # SAVE HISTORY
        with open("history.txt", "a") as f:
            f.write(f"{city}, {area}, {price}\n")

        # EMAIL
        if email and email.strip() != "":
            send_email(email, price, city, area)

        return render_template("index.html",
            price=price,
            low=low,
            high=high,
            score=score,
            confidence=confidence,
            suggestions=suggestions,
            facilities=facilities,
            city=city,
            area=area,
            sqft=sqft,
            bedroom=bedroom,
            ptype=ptype,
            email=email
        )

    except Exception as e:
        return render_template("index.html", error="Invalid input or data not found")

if __name__ == "__main__":
    app.run(debug=True)