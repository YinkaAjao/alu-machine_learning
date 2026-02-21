#!/usr/bin/env python3
"""rocket frequency"""

import requests


if __name__ == '__main__':
    launches_url = "https://api.spacexdata.com/v4/launches"
    rockets_url = "https://api.spacexdata.com/v4/rockets"

    # Fetch all launches
    launches = requests.get(launches_url).json()

    # Count launches per rocket ID
    rocket_counts = {}
    for launch in launches:
        rocket_id = launch["rocket"]
        rocket_counts[rocket_id] = rocket_counts.get(rocket_id, 0) + 1

    # Fetch all rockets to map ID -> name
    rockets = requests.get(rockets_url).json()
    rocket_names = {rocket["id"]: rocket["name"] for rocket in rockets}

    # Build sortable list: (name, count)
    result = [
        (rocket_names[rocket_id], count)
        for rocket_id, count in rocket_counts.items()
    ]

    # Sort by count desc, then name asc
    result.sort(key=lambda x: (-x[1], x[0]))

    # Print result
    for name, count in result:
        print(f"{name}: {count}")
