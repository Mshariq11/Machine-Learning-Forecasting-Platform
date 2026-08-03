"""
=========================================================
Data Audit Module
=========================================================

Reusable functions for auditing project datasets.

Author : Shariq Zia
Project: Store Sales Forecasting
"""

from pathlib import Path
from typing import Dict

import pandas as pd


# =========================================================
# DATA LOADING
# =========================================================

def load_datasets(raw_data_dir: Path) -> Dict[str, pd.DataFrame]:
    """
    Load all raw project datasets.

    Parameters
    ----------
    raw_data_dir : Path
        Path to data/raw directory.

    Returns
    -------
    dict
        Dictionary containing project datasets.
    """

    datasets = {

        "train":
            pd.read_csv(raw_data_dir / "train.csv"),

        "test":
            pd.read_csv(raw_data_dir / "test.csv"),

        "stores":
            pd.read_csv(raw_data_dir / "stores.csv"),

        "oil":
            pd.read_csv(raw_data_dir / "oil.csv"),

        "holidays":
            pd.read_csv(raw_data_dir / "holidays_events.csv")

    }

    return datasets


# =========================================================
# DATASET INVENTORY
# =========================================================

def dataset_inventory(
    datasets: Dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """
    Generate inventory of all datasets.

    Parameters
    ----------
    datasets : dict

    Returns
    -------
    pd.DataFrame
    """

    inventory = []

    for name, df in datasets.items():

        inventory.append({

            "Dataset": name,

            "Rows": df.shape[0],

            "Columns": df.shape[1],

            "Memory (MB)": round(

                df.memory_usage(deep=True)
                  .sum() / 1024**2,

                2

            )

        })

    return pd.DataFrame(inventory)


# =========================================================
# DATASET PREVIEW
# =========================================================

def preview_dataset(
    df: pd.DataFrame,
    rows: int = 5
) -> pd.DataFrame:
    """
    Preview a single dataset.
    """

    return df.head(rows)


def preview_all_datasets(
    datasets: Dict[str, pd.DataFrame],
    rows: int = 5
) -> Dict[str, pd.DataFrame]:
    """
    Preview every dataset.

    Parameters
    ----------
    datasets : dict
        Dictionary containing project datasets.

    rows : int, default=5
        Number of rows to preview.

    Returns
    -------
    Dict[str, pd.DataFrame]
        Dictionary of dataset previews.
    """

    previews = {}

    for name, df in datasets.items():

        previews[name] = preview_dataset(df, rows)

    return previews
# =========================================================
# DATASET INFORMATION
# =========================================================

def dataset_info(
    datasets: Dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """
    Generate structural information for each dataset.
    """

    summary = []

    for name, df in datasets.items():

        summary.append({

            "Dataset": name,

            "Rows": len(df),

            "Columns": df.shape[1],

            "Numeric": len(
                df.select_dtypes(include="number").columns
            ),

            "Categorical": len(
                df.select_dtypes(include=["object", "category"]).columns
            ),

            "Datetime": len(
                df.select_dtypes(include=["datetime64[ns]"]).columns
            ),

            "Memory (MB)": round(
                df.memory_usage(deep=True).sum() / 1024**2,
                2
            )

        })

    return pd.DataFrame(summary)


# =========================================================
# DATA TYPES
# =========================================================

def datatype_summary(
    datasets: Dict[str, pd.DataFrame]
) -> Dict[str, pd.DataFrame]:
    """
    Display data types for each dataset.
    """

    summary = {}

    for name, df in datasets.items():

        summary[name] = pd.DataFrame({

            "Column": df.columns,

            "Data Type": df.dtypes.astype(str)

        })

    return summary


# =========================================================
# MISSING VALUES
# =========================================================

def missing_summary(
    datasets: Dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """
    Missing value summary across all datasets.
    """

    records = []

    for name, df in datasets.items():

        for column in df.columns:

            missing = df[column].isna().sum()

            if missing > 0:

                records.append({

                    "Dataset": name,

                    "Column": column,

                    "Missing": missing,

                    "Missing (%)":
                        round(
                            missing / len(df) * 100,
                            2
                        )

                })

    return pd.DataFrame(records)


# =========================================================
# DUPLICATE SUMMARY
# =========================================================

def duplicate_summary(
    datasets: Dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """
    Duplicate record summary.
    """

    summary = []

    for name, df in datasets.items():

        duplicate_count = df.duplicated().sum()

        summary.append({

            "Dataset": name,

            "Duplicate Rows": duplicate_count,

            "Duplicate (%)":
                round(
                    duplicate_count / len(df) * 100,
                    2
                )

        })

    return pd.DataFrame(summary)


# =========================================================
# MEMORY SUMMARY
# =========================================================

def memory_summary(
    datasets: Dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """
    Dataset memory usage.
    """

    summary = []

    for name, df in datasets.items():

        summary.append({

            "Dataset": name,

            "Memory (MB)":
                round(
                    df.memory_usage(deep=True).sum() / 1024**2,
                    2
                )

        })

    return pd.DataFrame(summary)

# =========================================================
# RUN COMPLETE DATA AUDIT
# =========================================================

def run_data_audit(datasets: Dict[str, pd.DataFrame]) -> None:
    """
    Execute the complete data audit workflow.

    Parameters
    ----------
    datasets : Dict[str, pd.DataFrame]
        Dictionary containing all project datasets.

    Returns
    -------
    None
    """

    sections = [
        ("DATASET INVENTORY", dataset_inventory(datasets)),
        ("DATASET INFORMATION", dataset_info(datasets)),
        ("MISSING VALUE SUMMARY", missing_summary(datasets)),
        ("DUPLICATE SUMMARY", duplicate_summary(datasets)),
        ("MEMORY SUMMARY", memory_summary(datasets)),
    ]

    for title, result in sections:

        print("\n" + "=" * 70)
        print(title)
        print("=" * 70)

        if result.empty:
            print("No records found.")
        else:
            print(result)

    print("\n" + "=" * 70)
    print("DATA TYPES")
    print("=" * 70)

    datatype_reports = datatype_summary(datasets)

    for name, report in datatype_reports.items():

        print(f"\n{name.upper()} DATASET")
        print("-" * 70)
        print(report)

    print("\n" + "=" * 70)
    print("DATA AUDIT COMPLETED SUCCESSFULLY")
    print("=" * 70)