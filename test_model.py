# test_model.py
import joblib
import pandas as pd

model = joblib.load("house_price_model.pkl")

print("Enter property details:")
City = input("City: ").strip()
AreaName = input("Area/Colony: ").strip()
AreaSqft = float(input("Area (sqft): ").strip())
Bedrooms = int(input("Bedrooms: ").strip())
PropertyType = input("PropertyType (Buy/Rent): ").strip()

X = pd.DataFrame([{
    "City": City,
    "AreaName": AreaName,
    "AreaSqft": AreaSqft,
    "Bedrooms": Bedrooms,
    "PropertyType": PropertyType
}])

pred = model.predict(X)[0]
print(f"\nPredicted Price: ₹ {pred:,.0f}")