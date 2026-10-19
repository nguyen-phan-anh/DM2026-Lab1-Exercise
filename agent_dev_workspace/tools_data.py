import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import word_tokenize
import numpy as np
import os
from typing import Optional
import pandas as pd
import seaborn as sns
from langchain_core.tools import tool


def make_tools(session):
    @tool
    def describe_data_tool() -> dict:
        """Computes text length statistics overall and per category, building a grouped boxplot.

        Calculates character length text_length = df['text'].apply(len) on session.dataframe,
        computes .describe() statistics overall and grouped by category_name, renders a
        seaborn boxplot on session.pending_figure, and returns the summary stats.

        Returns:
            dict: Summary containing result_id, overall_stats, and category_stats (or error dict).
        """
        if session.dataframe is None or session.dataframe.empty:
            return {"error": "No dataset loaded in session. Call load_dataset_tool first."}

        if "text" not in session.dataframe.columns:
            return {"error": "DataFrame does not contain a 'text' column."}

        df = session.dataframe.copy()
        # Compute character length text_length = df['text'].apply(len)
        df["text_length"] = df["text"].fillna("").astype(str).apply(len)

        # 1. Compute overall .describe() standard stats
        overall_desc = df["text_length"].describe()
        overall_stats = {
            "count": int(overall_desc["count"]),
            "mean": round(float(overall_desc["mean"]), 4),
            "std": round(float(overall_desc["std"]), 4) if pd.notnull(overall_desc["std"]) else 0.0,
            "min": float(overall_desc["min"]),
            "25%": float(overall_desc["25%"]),
            "50%": float(overall_desc["50%"]),
            "75%": float(overall_desc["75%"]),
            "max": float(overall_desc["max"]),
        }

        # 2. Compute category_name .describe() stats if column exists
        category_stats = {}
        has_categories = "category_name" in df.columns and df["category_name"].notnull().any()

        fig, ax = plt.subplots(figsize=(8, 5))

        if has_categories:
            grouped = df.groupby("category_name")["text_length"]
            for cat_name, cat_series in grouped:
                cat_desc = cat_series.describe()
                category_stats[str(cat_name)] = {
                    "count": int(cat_desc["count"]),
                    "mean": round(float(cat_desc["mean"]), 4),
                    "std": round(float(cat_desc["std"]), 4) if pd.notnull(cat_desc["std"]) else 0.0,
                    "min": float(cat_desc["min"]),
                    "25%": float(cat_desc["25%"]),
                    "50%": float(cat_desc["50%"]),
                    "75%": float(cat_desc["75%"]),
                    "max": float(cat_desc["max"]),
                }

            # Grouped boxplot by category_name using hue="category_name" and legend=False
            sns.boxplot(
                data=df,
                x="category_name",
                y="text_length",
                hue="category_name",
                legend=False,
                ax=ax,
            )
            ax.set_title("Text Length Distribution by Category")
            ax.set_xlabel("Category")
        else:
            # Single boxplot overall if no category_name column exists
            sns.boxplot(
                data=df,
                y="text_length",
                ax=ax,
            )
            ax.set_title("Text Length Distribution (Overall)")

        ax.set_ylabel("Text Length (Characters)")
        plt.tight_layout()

        # Set pending_figure on session and close pyplot figure
        session.pending_figure = fig
        plt.close(fig)

        args_dict = {}

        summary = {
            "result_id": session.next_result_id("describe_data"),
            "overall_stats": overall_stats,
            "category_stats": category_stats,
        }

        session.store_result("describe_data_tool", args_dict, summary)
        return summary

    @tool
    def check_missing_tool() -> dict:
        """Checks for missing, empty, or whitespace-only text entries in the loaded dataset.

        Scans session.dataframe['text'] for NaN/None values, empty strings, and strings
        containing only whitespace, returning counts, percentages, and missing row indices.

        Returns:
            dict: Summary containing result_id, total_rows, missing_count, missing_pct,
                  and missing_indices (or error dict).
        """
        if session.dataframe is None or session.dataframe.empty:
            return {"error": "No dataset loaded in session. Call load_dataset_tool first."}

        if "text" not in session.dataframe.columns:
            return {"error": "DataFrame does not contain a 'text' column."}

        text_series = session.dataframe["text"]
        total_rows = len(text_series)

        # Detect NaN/None, empty string, or whitespace-only string
        missing_mask = text_series.isna() | (text_series.astype(str).str.strip() == "")

        missing_count = int(missing_mask.sum())
        missing_pct = round(float(100.0 * missing_count / total_rows), 4) if total_rows > 0 else 0.0

        # Get missing row indices as plain python ints
        missing_indices = [int(idx) for idx in text_series.index[missing_mask].tolist()]

        args_dict = {}

        summary = {
            "result_id": session.next_result_id("check_missing"),
            "total_rows": total_rows,
            "missing_count": missing_count,
            "missing_pct": missing_pct,
            "missing_indices": missing_indices,
        }

        session.store_result("check_missing_tool", args_dict, summary)
        return summary

    @tool
    def check_duplicates_tool(drop: bool = False) -> dict:
        """Checks for duplicate text documents in the dataset and optionally drops them.

        When dropping duplicates (drop=True), keep=False is used to remove every copy of a duplicated row.

        Args:
            drop (bool): If True, removes all copies of duplicated rows from session.dataframe. Default is False.

        Returns:
            dict: Summary containing result_id, duplicate_count, duplicate_row_indices,
                  total_rows_before, total_rows_after, and rows_removed (or error dict).
        """
        if session.dataframe is None or session.dataframe.empty:
            return {"error": "No dataset loaded in session. Call load_dataset_tool first."}

        if "text" not in session.dataframe.columns:
            return {"error": "DataFrame does not contain a 'text' column."}

        df = session.dataframe
        total_rows_before = len(df)

        # Count extra copies using keep="first" (so 1 duplicate pair = 1 duplicate count)
        duplicate_count = int(df.duplicated(subset=["text"], keep="first").sum())

        # Get index labels of ALL rows belonging to a duplicate group using keep=False
        all_dup_mask = df.duplicated(subset=["text"], keep=False)
        duplicate_row_indices = [int(idx) for idx in df.index[all_dup_mask].tolist()]

        rows_removed = 0
        if drop and duplicate_count > 0:
            # Drop every copy matching Master's drop_duplicates(keep=False, inplace=True) behavior exactly
            df.drop_duplicates(subset=["text"], keep=False, inplace=True)
            rows_removed = total_rows_before - len(df)

        total_rows_after = len(session.dataframe)

        args_dict = {"drop": drop}

        summary = {
            "result_id": session.next_result_id("check_duplicates"),
            "duplicate_count": duplicate_count,
            "duplicate_row_indices": duplicate_row_indices,
            "total_rows_before": total_rows_before,
            "total_rows_after": total_rows_after,
            "rows_removed": rows_removed,
        }

        session.store_result("check_duplicates_tool", args_dict, summary)
        return summary

    @tool
    def tokenize_tool(preview_cap: int = 5, token_preview_cap: int = 10) -> dict:
        """Tokenizes each document's text into lowercased unigrams.

        Uses nltk.word_tokenize on lowercased text, assigning full token lists to
        session.dataframe['unigrams'] and session.artifacts['unigrams'].

        Args:
            preview_cap (int): Maximum number of document token lists to include in unigrams_sample. Default is 5.
            token_preview_cap (int): Maximum number of tokens per sampled document list in unigrams_sample. Default is 10.

        Returns:
            dict: Summary containing result_id, total_documents, total_tokens,
                  avg_tokens_per_doc, and unigrams_sample (or error dict).
        """
        if session.dataframe is None or session.dataframe.empty:
            return {"error": "No dataset loaded in session. Call load_dataset_tool first."}

        if "text" not in session.dataframe.columns:
            return {"error": "DataFrame does not contain a 'text' column."}

        tokens_list = []
        for val in session.dataframe["text"]:
            if pd.isna(val) or str(val).strip() == "":
                tokens_list.append([])
            else:
                lowercased = str(val).lower()
                tokens = word_tokenize(lowercased)
                tokens_list.append(tokens)

        # Store full unigram token lists in session dataframe and artifacts
        session.dataframe["unigrams"] = tokens_list
        session.artifacts["unigrams"] = tokens_list

        total_documents = len(tokens_list)
        total_tokens = sum(len(t) for t in tokens_list)
        avg_tokens_per_doc = (
            round(float(total_tokens / total_documents), 4) if total_documents > 0 else 0.0
        )

        unigrams_sample = [
            doc_tokens[:token_preview_cap] for doc_tokens in tokens_list[:preview_cap]
        ]

        args_dict = {"preview_cap": preview_cap, "token_preview_cap": token_preview_cap}

        summary = {
            "result_id": session.next_result_id("tokenize"),
            "total_documents": total_documents,
            "total_tokens": total_tokens,
            "avg_tokens_per_doc": avg_tokens_per_doc,
            "unigrams_sample": unigrams_sample,
        }

        session.store_result("tokenize_tool", args_dict, summary)
        return summary

    @tool
    def sample_data_tool(
        n: Optional[int] = None,
        frac: Optional[float] = None,
        random_state: Optional[int] = None,
        preview_cap: int = 5,
        text_preview_length: int = 50,
    ) -> dict:
        """Subsamples rows from session.dataframe without modifying the original session state.

        Uses pandas DataFrame.sample to select random rows for inspection, returning sampled
        indices and truncated text previews while leaving session.dataframe intact.

        Args:
            n (Optional[int]): Number of items from axis to return.
            frac (Optional[float]): Fraction of axis items to return.
            random_state (Optional[int]): Seed integer for random number generator reproducibility.
            preview_cap (int): Maximum number of sampled rows to include in sampled_rows preview. Default is 5.
            text_preview_length (int): Maximum character length for truncated text preview. Default is 50.

        Returns:
            dict: Summary containing result_id, total_rows_before, sampled_count,
                  sampled_indices, and sampled_rows (or error dict).
        """
        if session.dataframe is None or session.dataframe.empty:
            return {"error": "No dataset loaded in session. Call load_dataset_tool first."}

        df = session.dataframe
        total_rows_before = len(df)

        if n is None and frac is None:
            n = min(5, total_rows_before)

        sampled_df = df.sample(n=n, frac=frac, random_state=random_state)
        sampled_count = len(sampled_df)

        # Capped sampled indices preview list
        sampled_indices = [int(idx) for idx in sampled_df.index[:preview_cap].tolist()]

        # Build preview list of sampled rows
        sampled_rows = []
        preview_slice = sampled_df.head(preview_cap)

        for idx, row in preview_slice.iterrows():
            row_dict = {"index": int(idx)}

            if "category_name" in row and pd.notnull(row["category_name"]):
                row_dict["category_name"] = str(row["category_name"])

            if "text" in row and pd.notnull(row["text"]):
                text_str = str(row["text"])
                row_dict["text_preview"] = (
                    text_str[:text_preview_length] + "..."
                    if len(text_str) > text_preview_length
                    else text_str
                )
            else:
                row_dict["text_preview"] = ""

            sampled_rows.append(row_dict)

        args_dict = {
            "n": n,
            "frac": frac,
            "random_state": random_state,
            "preview_cap": preview_cap,
            "text_preview_length": text_preview_length,
        }

        summary = {
            "result_id": session.next_result_id("sample_data"),
            "total_rows_before": total_rows_before,
            "sampled_count": sampled_count,
            "sampled_indices": sampled_indices,
            "sampled_rows": sampled_rows,
        }

        session.store_result("sample_data_tool", args_dict, summary)
        return summary

    @tool
    def list_files_tool(
        subdirectory: str = ".",
        preview_cap: int = 50,
    ) -> dict:
        """Lists files and directories relative to the repository root.

        Discovers dataset paths and files in the repository, refusing directory traversal
        outside the repo root and returning sorted directory/file lists.

        Args:
            subdirectory (str): Subdirectory path relative to the repo root (default is ".").
            preview_cap (int): Maximum number of files and directories to include in returned lists. Default is 50.

        Returns:
            dict: Summary containing result_id, subdirectory, total_directories, total_files,
                  directories, and files (or error dict).
        """
        # Refuse path traversal attempts containing ".."
        if ".." in subdirectory.replace("\\", "/").split("/"):
            return {"error": "Invalid subdirectory: path traversal ('..') is not allowed."}

        # Resolve against current working directory (repo root)
        repo_root = os.path.abspath(os.getcwd())
        target_path = os.path.abspath(os.path.join(repo_root, subdirectory))

        # Ensure target_path does not escape repo_root
        if not target_path.startswith(repo_root):
            return {"error": "Invalid subdirectory: path outside repository root is not allowed."}

        if not os.path.exists(target_path) or not os.path.isdir(target_path):
            return {"error": f"Subdirectory '{subdirectory}' does not exist or is not a directory."}

        try:
            entries = os.listdir(target_path)
        except Exception as e:
            return {"error": f"Failed to list directory '{subdirectory}': {str(e)}"}

        all_dirs = []
        all_files = []

        for entry in sorted(entries):
            full_entry_path = os.path.join(target_path, entry)
            if os.path.isdir(full_entry_path):
                all_dirs.append(entry)
            elif os.path.isfile(full_entry_path):
                all_files.append(entry)

        total_directories = len(all_dirs)
        total_files = len(all_files)

        args_dict = {"subdirectory": subdirectory, "preview_cap": preview_cap}

        summary = {
            "result_id": session.next_result_id("list_files"),
            "subdirectory": subdirectory,
            "total_directories": total_directories,
            "total_files": total_files,
            "directories": all_dirs[:preview_cap],
            "files": all_files[:preview_cap],
        }

        session.store_result("list_files_tool", args_dict, summary)
        return summary

    @tool
    def inspect_data_tool(
        n: int = 5,
        text_preview_length: int = 50,
        list_preview_items: int = 5,
    ) -> dict:
        """Inspects the loaded dataset's structure, column data types, and head rows.

        Returns dataset dimensions, column names and types, and preview dicts for the first
        n rows with strings, lists, numpy values, and NaNs safely formatted for JSON.

        Args:
            n (int): Number of head rows to inspect (capped at max 20). Default is 5.
            text_preview_length (int): Maximum character length for string previews. Default is 50.
            list_preview_items (int): Maximum number of items for list previews (e.g. unigrams). Default is 5.

        Returns:
            dict: Summary containing result_id, total_rows, total_columns, columns,
                  and head_rows (or error dict).
        """
        if session.dataframe is None or session.dataframe.empty:
            return {"error": "No dataset loaded in session. Call load_dataset_tool first."}

        df = session.dataframe
        total_rows = len(df)
        total_columns = len(df.columns)

        # Cap n between 1 and 20
        actual_n = min(max(1, n), 20)

        columns_info = [
            {"name": str(col), "dtype": str(df[col].dtype)}
            for col in df.columns
        ]

        head_df = df.head(actual_n)
        head_rows = []

        for idx, row in head_df.iterrows():
            row_dict = {"index": int(idx)}

            for col in df.columns:
                val = row[col]

                # Check for list / tuple BEFORE pd.isna check to avoid ambiguous array truth value errors
                if isinstance(val, (list, tuple)):
                    val_list = list(val)
                    total_len = len(val_list)
                    clipped_items = [
                        str(item)[:text_preview_length] for item in val_list[:list_preview_items]
                    ]
                    if total_len > list_preview_items:
                        row_dict[str(col)] = f"{clipped_items} (+{total_len - list_preview_items} more items)"
                    else:
                        row_dict[str(col)] = clipped_items
                # Handle NaN / None / NaT safely for scalar values
                elif pd.isna(val):
                    row_dict[str(col)] = None
                # Handle string cells
                elif isinstance(val, str):
                    if len(val) > text_preview_length:
                        row_dict[str(col)] = val[:text_preview_length] + "..."
                    else:
                        row_dict[str(col)] = val
                # Handle numpy numeric / boolean types for JSON serialization
                elif isinstance(val, (np.integer, int)):
                    row_dict[str(col)] = int(val)
                elif isinstance(val, (np.floating, float)):
                    row_dict[str(col)] = round(float(val), 4)
                elif isinstance(val, (np.bool_, bool)):
                    row_dict[str(col)] = bool(val)
                else:
                    str_val = str(val)
                    if len(str_val) > text_preview_length:
                        row_dict[str(col)] = str_val[:text_preview_length] + "..."
                    else:
                        row_dict[str(col)] = str_val

            head_rows.append(row_dict)

        args_dict = {
            "n": n,
            "text_preview_length": text_preview_length,
            "list_preview_items": list_preview_items,
        }

        summary = {
            "result_id": session.next_result_id("inspect_data"),
            "total_rows": total_rows,
            "total_columns": total_columns,
            "columns": columns_info,
            "head_rows": head_rows,
        }

        session.store_result("inspect_data_tool", args_dict, summary)
        return summary

    return [
        describe_data_tool,
        check_missing_tool,
        check_duplicates_tool,
        tokenize_tool,
        sample_data_tool,
        list_files_tool,
        inspect_data_tool,
    ]
