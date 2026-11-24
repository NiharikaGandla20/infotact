```python
"""
TerraSpectra - Vegetation Index Analysis

Calculates common vegetation indices from hyperspectral
or multispectral satellite imagery.
"""

import numpy as np


def calculate_ndvi(nir, red):
    """
    Calculate Normalized Difference Vegetation Index.

    NDVI = (NIR - Red) / (NIR + Red)
    """
    denominator = nir + red

    return np.divide(
        nir - red,
        denominator,
        out=np.zeros_like(denominator, dtype=float),
        where=denominator != 0
    )


def calculate_gndvi(nir, green):
    """
    Calculate Green Normalized Difference Vegetation Index.

    GNDVI = (NIR - Green) / (NIR + Green)
    """
    denominator = nir + green

    return np.divide(
        nir - green,
        denominator,
        out=np.zeros_like(denominator, dtype=float),
        where=denominator != 0
    )


def classify_vegetation(ndvi):
    """
    Classify vegetation condition using NDVI values.

    These are illustrative thresholds, not crop-specific
    disease or stress diagnosis.
    """
    conditions = np.full(ndvi.shape, "Bare soil or water", dtype=object)

    conditions[(ndvi >= 0.2) & (ndvi < 0.4)] = "Sparse vegetation"
    conditions[(ndvi >= 0.4) & (ndvi < 0.6)] = "Moderate vegetation"
    conditions[ndvi >= 0.6] = "Dense vegetation"

    return conditions


def summarize_ndvi(ndvi):
    """Return basic statistics for an NDVI array."""

    return {
        "mean": float(np.mean(ndvi)),
        "minimum": float(np.min(ndvi)),
        "maximum": float(np.max(ndvi)),
        "standard_deviation": float(np.std(ndvi))
    }
