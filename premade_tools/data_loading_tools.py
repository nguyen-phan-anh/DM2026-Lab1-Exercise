"""Provided tool for loading the working dataset: load_dataset_tool.

This is one of two tools in the whole pipeline given to you already
built, rather than something you write yourself (the other is
visualize_result_tool, in premade_tools/visualization_tools.py). Every
other tool (eighteen of them) is yours to build in the developer-stage
notebook. load_dataset_tool is an exception on purpose: it's the first
step in the pipeline, the one every other tool's input shape depends
on, so it serves as a single worked example of what a finished tool
looks like (the docstring, the type hints, the session.store_result(...)
call) before you go build the rest yourself.

Two datasets are supported, matching what the Master notebook and its
homework actually use: the built-in 20-newsgroups subset, or the
homework's local Reddit-stock-sentiment.csv. Either way, the result is
normalized into the same text/category/category_name shape in
session.dataframe, so nothing downstream needs to know or care which
mode was used to load it.
"""

import os

import pandas as pd
from langchain_core.tools import tool
from sklearn.datasets import fetch_20newsgroups

from helpers import data_mining_helpers as dmh


def _resolve_within_repo(path):
    """Resolve `path` relative to the repo root (the notebook's cwd),
    refusing anything that escapes it. The agent can browse and load
    repo files, not arbitrary filesystem paths."""
    root = os.path.abspath(os.getcwd())
    target = os.path.abspath(os.path.join(root, path))
    if target != root and not target.startswith(root + os.sep):
        return None
    return target


def make_tools(session):
    @tool
    def load_dataset_tool(
        categories: list[str] = None,
        file_path: str = None,
        text_column: str = None,
        label_column: str = None,
    ) -> dict:
        """Load a dataset to work with. Two mutually exclusive modes --
        provide exactly one:

        1. Built-in 20-newsgroups: pass `categories`.
        2. A local CSV file (e.g. the Lab 1 homework's
           newdataset/Reddit-stock-sentiment.csv): pass `file_path`
           plus `text_column` and `label_column` naming the CSV's text
           and label/category columns. The loaded data is normalized
           into the same text/category/category_name shape either way,
           so every other tool works unchanged regardless of which mode
           was used.

        This is the first tool of the lab; every later stage depends on
        it having run.

        Args:
            categories: newsgroup category names, e.g. ["alt.atheism",
                "soc.religion.christian", "comp.graphics", "sci.med"].
                Mode 1 only.
            file_path: path to a local CSV, relative to the repo root.
                Mode 2 only.
            text_column: name of the CSV column with the document text.
                Required with file_path.
            label_column: name of the CSV column with the category/label.
                Required with file_path.
        """
        if (categories is None) == (file_path is None):
            return {"error": "Provide exactly one of categories or file_path."}

        if categories is not None:
            dataset = fetch_20newsgroups(subset="train", categories=categories, shuffle=True, random_state=42)
            df = pd.DataFrame.from_records(dmh.format_rows(dataset), columns=["text"])
            df["category"] = dataset.target
            df["category_name"] = [dataset.target_names[t] for t in dataset.target]
            session.raw_dataset = dataset
            args = {"categories": categories}
            source_summary = {"categories_requested": categories}
        else:
            if not text_column or not label_column:
                return {"error": "file_path requires both text_column and label_column."}
            target = _resolve_within_repo(file_path)
            if target is None or not os.path.isfile(target):
                return {"error": f"File not found within the repo: '{file_path}'."}
            raw_df = pd.read_csv(target)
            missing = [c for c in (text_column, label_column) if c not in raw_df.columns]
            if missing:
                return {
                    "error": f"Column(s) not found in {file_path}: {missing}",
                    "available_columns": list(raw_df.columns),
                }
            df = pd.DataFrame(
                {
                    # fillna before astype(str): a blank CSV cell reads as
                    # NaN, and astype(str) on NaN produces the literal
                    # string "nan" rather than an empty one, which would
                    # make check_missing_tool blind to genuinely missing
                    # rows loaded from a CSV.
                    "text": raw_df[text_column].fillna("").astype(str),
                    "category_name": raw_df[label_column].fillna("").astype(str),
                }
            )
            df["category"] = pd.factorize(df["category_name"])[0]
            session.raw_dataset = None
            args = {"file_path": file_path, "text_column": text_column, "label_column": label_column}
            source_summary = {"file_path": file_path, "text_column": text_column, "label_column": label_column}

        session.dataframe = df

        result_id = session.next_result_id("load_dataset")
        counts = df["category_name"].value_counts().to_dict()
        summary = {
            "result_id": result_id,
            **source_summary,
            "n_documents": int(len(df)),
            "counts_per_category": counts,
        }
        session.store_result("load_dataset", args, summary, full_report=df)
        return summary

    return [load_dataset_tool]
