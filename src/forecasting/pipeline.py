"""
Forecast Pipeline
"""

from .inference import predict
from .recursive import recursive_forecast


def forecast(
    model,
    features,
    recursive=False
):
    """
    Forecast wrapper.
    """

    if recursive:

        return recursive_forecast(
            model,
            features
        )

    return predict(
        model,
        features
    )