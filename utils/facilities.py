
def nearby_facilities(area):

    if "sector" in area:
        return ["School", "Hospital", "Mall", "Metro"]

    elif "city" in area:
        return ["Market", "Hospital", "Bus Stop"]

    else:
        return ["Local Market", "Clinic"]