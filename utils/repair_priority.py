def get_repair_priority(severity):
    """
    Returns repair priority based on predicted severity.
    """

    severity = severity.lower()

    if severity == "major_pothole":
        return "High Priority"

    elif severity == "medium_pothole":
        return "Medium Priority"

    elif severity == "minor_pothole":
        return "Low Priority"

    return "Unknown"