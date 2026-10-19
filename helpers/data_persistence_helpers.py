"""Save and load processed dataframes for faster notebook iteration."""

import os

import pandas as pd


def save_processed_dataframes(tdm_df, augmented_dfs):
    """Save base TDM and augmented dataframes (one CSV per filter method)."""
    output_dir = "./output_files/processed_dataframes"
    os.makedirs(output_dir, exist_ok=True)

    print("Saving processed dataframes...")

    tdm_df.to_csv(f"{output_dir}/tdm_base.csv", index=True)
    print(f"  ✓ Saved tdm_base.csv ({tdm_df.shape})")

    for filter_name, df in augmented_dfs.items():
        filename = f"{output_dir}/augmented_{filter_name}.csv"
        df.to_csv(filename, index=True)
        print(f"  ✓ Saved augmented_{filter_name}.csv ({df.shape})")

    print(f"\n✓ All dataframes saved to {output_dir}/")


def load_processed_dataframes():
    """Load previously saved TDM and augmented dataframes."""
    input_dir = "./output_files/processed_dataframes"

    print("Loading processed dataframes...")

    tdm_df = pd.read_csv(f"{input_dir}/tdm_base.csv", index_col=0)
    print(f"  ✓ Loaded tdm_base.csv ({tdm_df.shape})")

    augmented_dfs = {}
    for filter_name in ["variance", "tfidf", "term_freq"]:
        filename = f"{input_dir}/augmented_{filter_name}.csv"
        augmented_dfs[filter_name] = pd.read_csv(filename, index_col=0)
        print(f"  ✓ Loaded augmented_{filter_name}.csv ({augmented_dfs[filter_name].shape})")

    print("\n✓ All dataframes loaded!")

    return tdm_df, augmented_dfs
