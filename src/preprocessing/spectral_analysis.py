"""
TerraSpectra - Spectral Analysis Module
Day 2: Analyze hyperspectral signatures.
"""

import numpy as np


def calculate_spectral_signature(cube):
    """
    Calculate average reflectance across
    all pixels for each spectral band.
    """

    signature = np.mean(cube, axis=(1, 2))

    return signature


def detect_spectral_anomalies(cube, threshold=0.10):
    """
    Identify pixels with unusually low
    average reflectance.

    This is a basic experimental anomaly
    detection method, not disease diagnosis.
    """

    mean_reflectance = np.mean(cube, axis=0)

    overall_mean = np.mean(mean_reflectance)

    anomalies = mean_reflectance < (
        overall_mean * (1 - threshold)
    )

    return anomalies


def analyze_hyperspectral_cube(cube):

    signature = calculate_spectral_signature(cube)

    anomalies = detect_spectral_anomalies(cube)

    print("TerraSpectra Spectral Analysis")
    print("--------------------------------")

    print("Total spectral bands:", cube.shape[0])

    print(
        "Average reflectance:",
        round(float(np.mean(signature)), 4)
    )

    print(
        "Potential anomalous pixels:",
        int(np.sum(anomalies))
    )

    return signature, anomalies


if __name__ == "__main__":

    # Generate sample hyperspectral data
    cube = np.random.default_rng(42).random(
        (200, 128, 128)
    ).astype(np.float32)

    signature, anomalies = analyze_hyperspectral_cube(cube)

    print("Spectral analysis completed!")
