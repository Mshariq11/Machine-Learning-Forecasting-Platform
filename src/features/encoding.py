"""
Encoding Features
"""

import pandas as pd


def encode_categorical_features(
    df: pd.DataFrame
):

    data = df.copy()

    mappings = {}

    columns = [

        "family",
        "city",
        "state",
        "type",
        "cluster"

    ]

    for col in columns:

        if col in data.columns:

            categories = sorted(

                data[col]
                .astype(str)
                .unique()

            )

            mapping = {

                value: idx

                for idx, value

                in enumerate(categories)

            }

            data[col] = (

                data[col]
                .astype(str)
                .map(mapping)

            )

            mappings[col] = mapping

    return data, mappings