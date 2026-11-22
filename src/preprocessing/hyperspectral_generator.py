
"""
TerraSpectra - Hyperspectral Data Generator
Day 1: Initial data generation module

Purpose:
Generate a synthetic hyperspectral image cube
for future crop stress detection experiments.
"""

import numpy as np

def generate_hyperspectral_data(
    height=128,
    width=128,
    bands=200
):
    """Generate a synthetic hyperspectral data cube."""

    # Reproducible random data
    np.random.seed(42)

    # Create an empty spectral cube
    cube = np.zeros(
        (bands, height, width),
        dtype=np.float32
    )

    # Simulate spectral reflectance
    for band in range(bands):
        reflectance = 0.2 + (band / bands) * 0.5

        cube[band] = (
            reflectance
            + np.random.normal(
                0, 0.02,
                (height, width)
            )
        )

    # Simulate an early crop stress region
    cube[:, 50:80, 50:80] *= 0.85

    # Keep reflectance values within range
    cube = np.clip(cube, 0, 1)

    return cube

if __name__ == "__main__":

    hyperspectral_cube = generate_hyperspectral_data()

    print("TerraSpectra Data Generator")
    print("---------------------------")
    print("Bands:", hyperspectral_cube.shape[0])
    print("Height:", hyperspectral_cube.shape[1])
    print("Width:", hyperspectral_cube.shape[2])
    print("Data type:", hyperspectral_cube.dtype)
    print("Data generation completed!")
