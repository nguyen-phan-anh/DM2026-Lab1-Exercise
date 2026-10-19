from typing import Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from langchain_core.tools import tool
from sklearn.metrics.pairwise import cosine_similarity


def make_tools(session):
    @tool
    def cosine_similarity_tool(
        doc1_idx: int,
        doc2_idx: int,
    ) -> dict:
        """Computes cosine similarity between two documents in the loaded dataset.

        Retrieves row vectors from session.feature_matrix (built by build_dtm_tool)
        for the given document indices and computes their pairwise cosine similarity.

        Args:
            doc1_idx: Row index of the first document in the dataset.
            doc2_idx: Row index of the second document in the dataset.

        Returns:
            dict: Summary containing result_id, doc1_idx, doc2_idx, category1, category2, and similarity.
        """
        if session.feature_matrix is None or session.feature_names is None:
            raise ValueError(
                "Document-term matrix not built yet. Call build_dtm_tool first."
            )

        n_docs = session.feature_matrix.shape[0]
        if doc1_idx < 0 or doc1_idx >= n_docs:
            raise ValueError(f"doc1_idx {doc1_idx} out of bounds (0 to {n_docs - 1}).")
        if doc2_idx < 0 or doc2_idx >= n_docs:
            raise ValueError(f"doc2_idx {doc2_idx} out of bounds (0 to {n_docs - 1}).")

        # Extract sparse row vectors
        v1 = session.feature_matrix[doc1_idx]
        v2 = session.feature_matrix[doc2_idx]

        # Compute cosine similarity (sklearn returns 0.0 for zero vectors natively)
        sim_matrix = cosine_similarity(v1, v2)
        sim_val = float(sim_matrix[0, 0])
        sim_val = float(np.nan_to_num(sim_val, nan=0.0))

        # Retrieve document category labels if available
        cat1 = None
        cat2 = None
        if session.dataframe is not None and "category_name" in session.dataframe.columns:
            cat1 = str(session.dataframe.iloc[doc1_idx]["category_name"])
            cat2 = str(session.dataframe.iloc[doc2_idx]["category_name"])

        args_dict = {"doc1_idx": doc1_idx, "doc2_idx": doc2_idx}

        summary = {
            "result_id": session.next_result_id("cosine_similarity"),
            "doc1_idx": doc1_idx,
            "doc2_idx": doc2_idx,
            "category1": cat1,
            "category2": cat2,
            "similarity": round(sim_val, 4),
        }

        session.store_result("cosine_similarity_tool", args_dict, summary)
        return summary

    @tool
    def feature_correlation_matrix_tool(top_n: int = 20) -> dict:
        """Computes feature-vs-feature correlation matrix and heatmap for top variance terms.

        Selects top-n terms by variance across all documents in session.feature_matrix,
        computes their pairwise Pearson correlation matrix, sets seaborn heatmap on
        session.pending_figure, and returns the correlation matrix in summary.

        Args:
            top_n: Maximum number of top terms by variance to include (default 20).

        Returns:
            dict: Summary containing result_id, n_terms, terms, and correlation_matrix (or error dict).
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"error": "Document-term matrix not built yet. Call build_dtm_tool first."}

        matrix = session.feature_matrix
        terms = np.array(session.feature_names)

        # Compute feature variances across all documents
        means = np.array(matrix.mean(axis=0)).flatten()
        means_sq = np.array(matrix.power(2).mean(axis=0)).flatten()
        variances = np.clip(means_sq - np.square(means), 0.0, None)

        # Sort indices purely by variance descending to match Known Answer 19 in TEST_FIXTURE.md
        # (alpha: 1.1875, gamma: 1.109375, delta: 0.25, beta: 0.234375, always: 0.109375)
        df_var = pd.DataFrame({"term": terms, "variance": variances, "orig_idx": np.arange(len(terms))})
        df_var = df_var.sort_values(by="variance", ascending=False).reset_index(drop=True)

        top_df = df_var.head(top_n)
        top_indices = top_df["orig_idx"].values
        selected_terms = top_df["term"].tolist()

        k = len(selected_terms)
        if k == 0:
            return {"error": "Document-term matrix contains no terms."}

        # Extract submatrix for selected top-variance terms
        submatrix = matrix[:, top_indices].toarray()

        # Compute Pearson correlation matrix (k x k)
        corr_matrix = np.corrcoef(submatrix, rowvar=False)

        # Check if any NaN values exist due to zero variance
        has_nan = np.isnan(corr_matrix).any()

        if k == 1:
            corr_matrix = np.array([[1.0]])
        else:
            corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
            np.fill_diagonal(corr_matrix, 1.0)

        corr_matrix_list = [
            [round(float(val), 4) for val in row] for row in corr_matrix
        ]

        # Render seaborn heatmap
        fig, ax = plt.subplots(figsize=(max(6, k * 0.4), max(5, k * 0.35)))
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            vmin=-1.0,
            vmax=1.0,
            xticklabels=selected_terms,
            yticklabels=selected_terms,
            ax=ax,
        )
        ax.set_title("Feature Correlation Matrix (Top Variance Terms)")
        plt.tight_layout()

        # Set pending_figure on session and close pyplot figure
        session.pending_figure = fig
        plt.close(fig)

        args_dict = {"top_n": top_n}

        summary = {
            "result_id": session.next_result_id("feature_correlation"),
            "n_terms": k,
            "terms": selected_terms,
            "correlation_matrix": corr_matrix_list,
        }

        if has_nan:
            summary["note"] = "Zero-variance feature(s) detected; NaN correlation values were replaced with 0.0."

        session.store_result("feature_correlation_matrix_tool", args_dict, summary)
        return summary

    return [cosine_similarity_tool, feature_correlation_matrix_tool]
