def get_maintenance_cost(severity):
    """
    Returns estimated maintenance cost based on pothole severity.
    """

    severity = severity.lower()

    if severity == "major_pothole":
        return 15000

    elif severity == "medium_pothole":
        return 8000

    elif severity == "minor_pothole":
        return 3000

    return 0