"""
Submission Utilities
"""

import pandas as pd


def create_submission(
    ids,
    predictions
):
    """
    Create Kaggle submission file.
    """

    submission = pd.DataFrame({

        "id": ids,

        "sales": predictions

    })

    return submission


def save_submission(
    submission,
    filepath
):

    submission.to_csv(
        filepath,
        index=False
    )