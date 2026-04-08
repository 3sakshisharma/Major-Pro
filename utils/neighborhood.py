

def get_score(city, area):

    base = {
        "delhi": 8,
        "gurgaon": 9,
        "noida": 7,
        "ghaziabad": 6
    }

    score = base.get(city, 5)

    if "sector" in area:
        score += 1
    if "village" in area:
        score -= 1

    return min(score, 10)