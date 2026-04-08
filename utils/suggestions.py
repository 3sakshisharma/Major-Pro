import pandas as pd

data = pd.read_csv("house_data.csv")

for col in ['city','areaname']:
    data[col] = data[col].str.lower().str.strip()

def get_similar(city, area, sqft):

    filtered = data[
        (data['city']==city) &
        (data['areaname']==area)
    ].copy()

    if len(filtered)==0:
        return ["No similar properties found"]

    filtered['diff'] = abs(filtered['areasqft'] - sqft)
    filtered = filtered.sort_values(by='diff')

    results = filtered.head(3)

    return [
        f"{r['bedrooms']} BHK | {r['areasqft']} sqft | ₹{r['price']}"
        for _,r in results.iterrows()
    ]