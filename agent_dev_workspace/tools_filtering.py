from typing import Optional
import numpy as np
import pandas as pd
from scipy import stats
from langchain_core.tools import tool


def make_tools(session):
    @tool
    def variance_filter_tool(
        threshold: float,
        preview_cap: int = 10,
    ) -> dict:
        """Flags terms in the document-term matrix by variance below a given threshold.

        Computes feature variances across all documents in session.feature_matrix,
        identifying low-variance (near-constant) terms for potential removal.

        Args:
            threshold: Minimum variance required for a term to be kept (no default; caller must specify).
            preview_cap: Maximum number of terms to include in kept/removed samples (default 10).

        Returns:
            dict: Summary containing result_id, threshold, n_total_terms, n_kept, n_removed,
                  kept_terms_sample, and removed_terms_sample.
        """
        if session.feature_matrix is None or session.feature_names is None:
            raise ValueError(
                "Document-term matrix not built yet. Call build_dtm_tool first."
            )

        # Compute per-term variance across all documents in session.feature_matrix
        matrix = session.feature_matrix
        means = np.array(matrix.mean(axis=0)).flatten()
        means_sq = np.array(matrix.power(2).mean(axis=0)).flatten()
        variances = means_sq - np.square(means)

        # Clip negative float precision noise to exactly 0.0
        variances = np.clip(variances, 0.0, None)

        terms = session.feature_names

        # Build full_report DataFrame sorted descending by variance, then ascending by term
        df = pd.DataFrame({"term": terms, "variance": variances})
        df = df.sort_values(by=["variance", "term"], ascending=[False, True]).reset_index(
            drop=True
        )

        # Partition into kept and removed sets
        kept_df = df[df["variance"] >= threshold]
        removed_df = df[df["variance"] < threshold]

        n_total_terms = len(df)
        n_kept = len(kept_df)
        n_removed = len(removed_df)

        # Kept preview: top kept terms sorted descending by variance
        kept_terms_sample = kept_df.head(preview_cap)["term"].tolist()

        # Removed preview: most constant removed terms sorted ascending by variance
        removed_df_asc = removed_df.sort_values(
            by=["variance", "term"], ascending=[True, True]
        )
        removed_terms_sample = removed_df_asc.head(preview_cap)["term"].tolist()

        args_dict = {"threshold": threshold, "preview_cap": preview_cap}

        summary = {
            "result_id": session.next_result_id("variance_filter"),
            "threshold": threshold,
            "n_total_terms": n_total_terms,
            "n_kept": n_kept,
            "n_removed": n_removed,
            "kept_terms_sample": kept_terms_sample,
            "removed_terms_sample": removed_terms_sample,
        }

        # Store full report matching spec convention: 1st column term, 2nd column variance, sorted
        session.store_result("variance_filter_tool", args_dict, summary, full_report=df)
        return summary

    @tool
    def pearson_filter_tool(
        target_class: str,
        preview_cap: int = 10,
    ) -> dict:
        """Computes Pearson correlation between each term count and a target class label.

        Evaluates feature-class linear correlation (one-vs-rest binary indicator) across all
        documents, returning ranked terms with r values and p-values.

        Args:
            target_class: Category label to correlate terms against (one-vs-rest).
            preview_cap: Maximum number of top terms to include in top_terms sample (default 10).

        Returns:
            dict: Summary containing result_id, target_class, n_total_terms, and top_terms sample.
        """
        if session.feature_matrix is None or session.feature_names is None:
            raise ValueError(
                "Document-term matrix not built yet. Call build_dtm_tool first."
            )

        if session.dataframe is None or "category_name" not in session.dataframe.columns:
            raise ValueError(
                "Dataset or category_name column missing in session. Call load_dataset_tool first."
            )

        if target_class not in session.dataframe["category_name"].values:
            raise ValueError(
                f"Target class '{target_class}' not found in category_name column."
            )

        matrix = session.feature_matrix
        n_docs, n_terms = matrix.shape

        # Build binary target vector y (1.0 for target_class, 0.0 otherwise)
        y = (session.dataframe["category_name"] == target_class).astype(float).values

        if len(y) != n_docs:
            raise ValueError(
                f"Target vector length ({len(y)}) does not match feature matrix rows ({n_docs})."
            )

        y_std = np.std(y)
        if y_std == 0:
            raise ValueError(
                f"Target class '{target_class}' results in a constant target vector across all documents."
            )

        y_mean = np.mean(y)

        # Vectorized sparse matrix operations:
        # cov(X_j, Y) = E[X_j * Y] - E[X_j] * E[Y] = (X^T @ y) / N - mean_x * mean_y
        means_x = np.array(matrix.mean(axis=0)).flatten()
        means_sq_x = np.array(matrix.power(2).mean(axis=0)).flatten()
        stds_x = np.sqrt(np.maximum(0.0, means_sq_x - np.square(means_x)))

        # Matrix-vector multiplication on sparse matrix
        sum_xy = np.array(matrix.T.dot(y)).flatten()
        means_xy = sum_xy / n_docs

        cov_xy = means_xy - (means_x * y_mean)

        # Compute Pearson r vector safely
        with np.errstate(divide="ignore", invalid="ignore"):
            r_values = cov_xy / (stds_x * y_std)

        # Replace zero-variance feature NaN/inf with 0.0 and clip [-1.0, 1.0] precision noise
        r_values = np.nan_to_num(r_values, nan=0.0, posinf=0.0, neginf=0.0)
        r_values = np.clip(r_values, -1.0, 1.0)

        # Compute p-values vectorized via student's t-distribution:
        # t = r * sqrt((N - 2) / (1 - r^2))
        dof = n_docs - 2
        p_values = np.ones(n_terms, dtype=float)
        if dof > 0:
            with np.errstate(divide="ignore", invalid="ignore"):
                r_sq = np.square(r_values)
                # Where r == +/-1.0, p-value is 0.0
                perfect_mask = np.isclose(r_sq, 1.0)
                denom = np.maximum(0.0, 1.0 - r_sq)
                t_stats = np.where(perfect_mask, np.inf, np.abs(r_values) * np.sqrt(dof / denom))
                p_vals = 2.0 * stats.t.sf(t_stats, df=dof)
                p_vals[perfect_mask] = 0.0
                p_values = np.nan_to_num(p_vals, nan=1.0)

        terms = session.feature_names

        df = pd.DataFrame(
            {"term": terms, "pearson_r": r_values, "pearson_p": p_values}
        )

        # Rank terms by absolute correlation descending, then term name ascending
        df["abs_r"] = df["pearson_r"].abs()
        df = df.sort_values(by=["abs_r", "term"], ascending=[False, True]).reset_index(
            drop=True
        )
        df_report = df.drop(columns=["abs_r"])

        top_df = df_report.head(preview_cap)
        top_terms = [
            {
                "term": str(row["term"]),
                "pearson_r": round(float(row["pearson_r"]), 4),
                "pearson_p": float(row["pearson_p"]),  # Not rounded so tiny p-values aren't lost to 0.0
            }
            for _, row in top_df.iterrows()
        ]

        args_dict = {"target_class": target_class, "preview_cap": preview_cap}

        summary = {
            "result_id": session.next_result_id("pearson_filter"),
            "target_class": target_class,
            "n_total_terms": len(df_report),
            "top_terms": top_terms,
        }

        # Store full report matching spec convention: 1st column term, 2nd column pearson_r, sorted
        session.store_result("pearson_filter_tool", args_dict, summary, full_report=df_report)
        return summary

    @tool
    def spearman_filter_tool(
        target_class: str,
        preview_cap: int = 10,
    ) -> dict:
        """Computes Spearman rank correlation between each term count and a target class label.

        Evaluates feature-class monotonic correlation (one-vs-rest binary indicator) across all
        documents using average ranks for ties, returning ranked terms with r values and p-values.

        Args:
            target_class: Category label to correlate terms against (one-vs-rest).
            preview_cap: Maximum number of top terms to include in top_terms sample (default 10).

        Returns:
            dict: Summary containing result_id, target_class, n_total_terms, and top_terms sample.
        """
        if session.feature_matrix is None or session.feature_names is None:
            raise ValueError(
                "Document-term matrix not built yet. Call build_dtm_tool first."
            )

        if session.dataframe is None or "category_name" not in session.dataframe.columns:
            raise ValueError(
                "Dataset or category_name column missing in session. Call load_dataset_tool first."
            )

        if target_class not in session.dataframe["category_name"].values:
            raise ValueError(
                f"Target class '{target_class}' not found in category_name column."
            )

        matrix = session.feature_matrix.tocsc()
        n_docs, n_terms = matrix.shape

        # Build binary target vector y (1.0 for target_class, 0.0 otherwise)
        y = (session.dataframe["category_name"] == target_class).astype(float).values

        if len(y) != n_docs:
            raise ValueError(
                f"Target vector length ({len(y)}) does not match feature matrix rows ({n_docs})."
            )

        y_std = np.std(y)
        if y_std == 0:
            raise ValueError(
                f"Target class '{target_class}' results in a constant target vector across all documents."
            )

        # Precompute target ranks once (with average ranks for ties)
        rank_y = stats.rankdata(y, method="average")
        rank_y_mean = np.mean(rank_y)
        rank_y_std = np.std(rank_y)

        r_values = []
        p_values = []

        dof = n_docs - 2

        for j in range(n_terms):
            col = matrix.getcol(j)
            nz_indices = col.indices
            nz_values = col.data

            k = len(nz_values)
            z = n_docs - k

            # Build rank vector for feature column j efficiently
            rank_x = np.full(n_docs, (z + 1) / 2.0, dtype=float)
            if k > 0:
                nz_ranks = stats.rankdata(nz_values, method="average") + z
                rank_x[nz_indices] = nz_ranks

            std_x = np.std(rank_x)

            if std_x == 0 or rank_y_std == 0:
                r_values.append(0.0)
                p_values.append(1.0)
            else:
                cov = np.mean((rank_x - np.mean(rank_x)) * (rank_y - rank_y_mean))
                r = cov / (std_x * rank_y_std)
                r = float(np.clip(r, -1.0, 1.0))

                if dof > 0:
                    r_sq = r * r
                    if np.isclose(r_sq, 1.0):
                        p = 0.0
                    else:
                        t_stat = abs(r) * np.sqrt(dof / (1.0 - r_sq))
                        p = float(2.0 * stats.t.sf(t_stat, df=dof))
                else:
                    p = 1.0

                r_values.append(r)
                p_values.append(p)

        terms = session.feature_names

        df = pd.DataFrame(
            {"term": terms, "spearman_r": r_values, "spearman_p": p_values}
        )

        # Rank terms by absolute correlation descending, then term name ascending
        df["abs_r"] = df["spearman_r"].abs()
        df = df.sort_values(by=["abs_r", "term"], ascending=[False, True]).reset_index(
            drop=True
        )
        df_report = df.drop(columns=["abs_r"])

        top_df = df_report.head(preview_cap)
        top_terms = [
            {
                "term": str(row["term"]),
                "spearman_r": round(float(row["spearman_r"]), 4),
                "spearman_p": float(row["spearman_p"]),  # Not rounded so tiny p-values aren't lost to 0.0
            }
            for _, row in top_df.iterrows()
        ]

        args_dict = {"target_class": target_class, "preview_cap": preview_cap}

        summary = {
            "result_id": session.next_result_id("spearman_filter"),
            "target_class": target_class,
            "n_total_terms": len(df_report),
            "top_terms": top_terms,
        }

        # Store full report matching spec convention: 1st column term, 2nd column spearman_r, sorted
        session.store_result("spearman_filter_tool", args_dict, summary, full_report=df_report)
        return summary

    return [variance_filter_tool, pearson_filter_tool, spearman_filter_tool]
