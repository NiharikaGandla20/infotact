import numpy as np


def normalize_band(band):
    band = np.asarray(band, dtype=float)

    minimum = np.min(band)
    maximum = np.max(band)

    if maximum == minimum:
        return np.zeros_like(band)

    normalized = (band - minimum) / (maximum - minimum)

    return normalized


def calculate_band_statistics(cube):
    cube = np.asarray(cube, dtype=float)

    if cube.ndim != 3:
        raise ValueError("Input must be a 3D hyperspectral cube.")

    statistics = {}

    for i in range(cube.shape[2]):
        band = cube[:, :, i]

        statistics[f"Band_{i + 1}"] = {
            "mean": round(float(np.mean(band)), 4),
            "minimum": round(float(np.min(band)), 4),
            "maximum": round(float(np.max(band)), 4)
        }

    return statistics


if __name__ == "__main__":
    # Generate sample hyperspectral data
    cube = np.random.rand(128, 128, 200)

    band = cube[:, :, 50]

    normalized_band = normalize_band(band)

    statistics = calculate_band_statistics(cube)

    print("Selected Band: 51")
    print("Normalized Band Shape:", normalized_band.shape)

    print("\nFirst Five Band Statistics:")

    for name, values in list(statistics.items())[:5]:
        print(name, values)