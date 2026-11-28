import numpy as np


# 1. Calculate Mean Spectral Signature
def calculate_spectral_signature(cube):
    """
    Calculate the average spectral signature
    from a hyperspectral data cube.
    """

    cube = np.asarray(cube, dtype=float)

    if cube.ndim != 3:
        raise ValueError("Input must be a 3D hyperspectral cube.")

    return np.mean(cube, axis=(0, 1))


# 2. Detect Spectral Anomalies
def detect_spectral_anomaly(cube, threshold=2.0):
    """
    Identify pixels with unusual spectral intensity.
    """

    cube = np.asarray(cube, dtype=float)

    if cube.ndim != 3:
        raise ValueError("Input must be a 3D hyperspectral cube.")

    mean = np.mean(cube)
    std = np.std(cube)

    if std == 0:
        return np.zeros(cube.shape[:2], dtype=bool)

    pixel_mean = np.mean(cube, axis=2)

    anomaly_map = np.abs(pixel_mean - mean) > threshold * std

    return anomaly_map


# 3. Calculate Spectral Angle Mapper (SAM)
def calculate_spectral_angle(signature1, signature2):
    """
    Calculate the spectral angle between two signatures.
    A smaller angle indicates greater spectral similarity.
    """

    signature1 = np.asarray(signature1, dtype=float)
    signature2 = np.asarray(signature2, dtype=float)

    if signature1.shape != signature2.shape:
        raise ValueError(
            "Both spectral signatures must have the same dimensions."
        )

    if signature1.ndim != 1:
        raise ValueError("Spectral signatures must be one-dimensional.")

    norm1 = np.linalg.norm(signature1)
    norm2 = np.linalg.norm(signature2)

    if norm1 == 0 or norm2 == 0:
        raise ValueError("Spectral signatures cannot be zero vectors.")

    cosine_similarity = np.dot(signature1, signature2) / (norm1 * norm2)

    cosine_similarity = np.clip(cosine_similarity, -1.0, 1.0)

    angle = np.arccos(cosine_similarity)

    return float(np.degrees(angle))


# 4. Compare Spectral Signatures
def compare_spectral_signatures(signature1, signature2):
    """
    Compare two spectral signatures using SAM.
    """

    angle = calculate_spectral_angle(signature1, signature2)

    if angle < 5:
        status = "Very Similar"
    elif angle < 15:
        status = "Moderately Similar"
    else:
        status = "Different Spectral Signatures"

    return {
        "spectral_angle": round(angle, 2),
        "status": status
    }


# 5. Generate Spectral Analysis Summary
def generate_spectral_summary(cube):
    """
    Generate statistical information about hyperspectral bands.
    """

    cube = np.asarray(cube, dtype=float)

    if cube.ndim != 3:
        raise ValueError("Input must be a 3D hyperspectral cube.")

    summary = {
        "height": cube.shape[0],
        "width": cube.shape[1],
        "number_of_bands": cube.shape[2],
        "mean_reflectance": round(float(np.mean(cube)), 4),
        "minimum_reflectance": round(float(np.min(cube)), 4),
        "maximum_reflectance": round(float(np.max(cube)), 4)
    }

    return summary


# 6. Main Execution
if __name__ == "__main__":

    # Generate synthetic hyperspectral data
    hyperspectral_cube = np.random.rand(128, 128, 200)

    print("===== TerraSpectra Spectral Analysis =====")

    # Calculate spectral signature
    signature = calculate_spectral_signature(hyperspectral_cube)

    print("\nMean Spectral Signature:")
    print("Number of bands:", len(signature))

    # Detect anomalies
    anomalies = detect_spectral_anomaly(hyperspectral_cube)

    print("\nSpectral Anomaly Detection:")
    print("Anomalous pixels:", int(np.sum(anomalies)))

    # Compare healthy and stressed signatures
    healthy_signature = np.random.uniform(0.4, 0.9, 200)
    stressed_signature = np.random.uniform(0.1, 0.5, 200)

    comparison = compare_spectral_signatures(
        healthy_signature,
        stressed_signature
    )

    print("\nSpectral Signature Comparison:")
    print("Spectral Angle:", comparison["spectral_angle"])
    print("Similarity:", comparison["status"])

    # Generate summary
    summary = generate_spectral_summary(hyperspectral_cube)

    print("\nHyperspectral Data Summary:")

    for key, value in summary.items():
        print(f"{key}: {value}")