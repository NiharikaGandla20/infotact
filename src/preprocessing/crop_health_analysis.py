import numpy as np


def analyze_crop_health(ndvi):
    ndvi = np.asarray(ndvi, dtype=float)

    health_map = np.full(ndvi.shape, "Healthy Vegetation", dtype=object)

    health_map[ndvi < 0.2] = "Poor Vegetation"
    health_map[(ndvi >= 0.2) & (ndvi < 0.4)] = "High Stress"
    health_map[(ndvi >= 0.4) & (ndvi < 0.6)] = "Moderate Health"

    return health_map


def generate_health_summary(health_map):
    categories, counts = np.unique(health_map, return_counts=True)

    total = health_map.size
    summary = {}

    for category, count in zip(categories, counts):
        summary[category] = {
            "pixel_count": int(count),
            "percentage": round(float(count / total * 100), 2)
        }

    return summary


if __name__ == "__main__":
    sample_ndvi = np.random.uniform(0, 1, (5, 5))

    health_map = analyze_crop_health(sample_ndvi)
    summary = generate_health_summary(health_map)

    print("Crop Health Map:")
    print(health_map)

    print("\nCrop Health Summary:")
    for category, details in summary.items():
        print(category, details)