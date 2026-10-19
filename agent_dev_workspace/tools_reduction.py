import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from langchain_core.tools import tool


def make_tools(session):
    @tool
    def reduce_dimensions_tool(
        method: str,
        n_components: int = 2,
        random_state: int = 42,
        perplexity: float = 30.0,
        n_neighbors: int = 15,
    ) -> dict:
        """Applies dimensionality reduction (PCA, t-SNE, or UMAP) to the feature matrix.

        Reduces session.feature_matrix down to 2D/3D coordinates for visualization
        or downstream analysis, plots a scatter plot colored by category on
        session.pending_figure, and stores the full reduced coordinates in
        session.artifacts.

        Args:
            method: Reduction algorithm to use ('pca', 'tsne', or 'umap').
            n_components: Number of target dimensions (default is 2).
            random_state: Random seed for reproducibility.
            perplexity: Perplexity parameter for t-SNE (must be < n_documents).
            n_neighbors: Number of neighbors for UMAP (must be < n_documents).

        Returns:
            Dict with result summary including reduced coordinates preview and metrics.
        """
        if session.feature_matrix is None:
            return {"error": "No feature matrix available. Run build_dtm_tool first."}

        X = session.feature_matrix
        if hasattr(X, "toarray"):
            X = X.toarray()

        n_samples = X.shape[0]
        method_lower = method.lower()

        if method_lower == "tsne":
            if perplexity >= n_samples:
                return {
                    "error": f"perplexity ({perplexity}) must be strictly less than the number of documents ({n_samples})."
                }
            model = TSNE(
                n_components=n_components,
                perplexity=perplexity,
                random_state=random_state,
            )
            coords = model.fit_transform(X)
            metrics = {}
        elif method_lower == "pca":
            model = PCA(
                n_components=n_components,
                random_state=random_state,
            )
            coords = model.fit_transform(X)
            metrics = {
                "explained_variance_ratio": [
                    float(x) for x in model.explained_variance_ratio_
                ]
            }
        elif method_lower == "umap":
            if n_neighbors >= n_samples:
                return {
                    "error": f"n_neighbors ({n_neighbors}) must be strictly less than the number of documents ({n_samples})."
                }
            model = umap.UMAP(
                n_components=n_components,
                n_neighbors=n_neighbors,
                random_state=random_state,
            )
            coords = model.fit_transform(X)
            metrics = {}
        else:
            return {
                "error": f"Unsupported reduction method '{method}'. Choose from 'pca', 'tsne', or 'umap'."
            }

        # Save full coordinates in artifacts
        artifact_key = f"reduced_coords_{method_lower}"
        coords_list = coords.tolist()
        session.artifacts[artifact_key] = coords_list

        # Plotting requirement: build scatter plot and assign to session.pending_figure
        fig, ax = plt.subplots(figsize=(8, 6))
        labels = None
        if session.dataframe is not None and "category_name" in session.dataframe.columns:
            labels = session.dataframe["category_name"].tolist()
        elif session.labels is not None:
            labels = list(session.labels)

        if labels is not None:
            unique_labels = sorted(list(set(labels)))
            for cat in unique_labels:
                indices = [j for j, l in enumerate(labels) if l == cat]
                cat_coords = coords[indices]
                ax.scatter(
                    cat_coords[:, 0],
                    cat_coords[:, 1],
                    label=str(cat),
                    alpha=0.7,
                    s=50,
                )
            ax.legend(title="Category")
        else:
            ax.scatter(coords[:, 0], coords[:, 1], alpha=0.7, s=50)

        ax.set_title(f"{method.upper()} Dimensionality Reduction")
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        ax.grid(True, linestyle="--", alpha=0.5)
        fig.tight_layout()

        session.pending_figure = fig
        plt.close(fig)

        # Generate result ID and store summary
        res_id = session.next_result_id("reduction")
        summary = {
            "result_id": res_id,
            "method": method_lower,
            "n_components": n_components,
            "n_samples": n_samples,
            "coordinates_preview": coords_list[:5],  # Preview only (first 5 rows)
            "artifact_key": artifact_key,
            **metrics,
        }

        session.store_result(
            tool_name="reduce_dimensions_tool",
            args={
                "method": method,
                "n_components": n_components,
                "random_state": random_state,
                "perplexity": perplexity,
                "n_neighbors": n_neighbors,
            },
            summary=summary,
        )

        return summary

    @tool
    def binarize_labels_tool() -> dict:
        """One-hot encodes the category labels of the loaded dataset.

        Reads category_name from session.dataframe, generates an explicit 0/1 integer
        one-hot encoded matrix, stores it as a 2D NumPy integer array in
        session.artifacts['binarized_labels'], and returns a summary containing
        the label names in order, matrix shape, and document counts per label.

        Returns:
            Dict containing result summary with label names, matrix shape, and document counts.
        """
        if session.dataframe is None or "category_name" not in session.dataframe.columns:
            return {"error": "No dataset loaded or 'category_name' column missing."}

        category_series = session.dataframe["category_name"]
        dummies_df = pd.get_dummies(category_series, dtype=int)

        label_names = list(dummies_df.columns)
        matrix = dummies_df.to_numpy(dtype=int)

        session.artifacts["binarized_labels"] = matrix

        counts_per_label = {
            str(col): int(dummies_df[col].sum()) for col in dummies_df.columns
        }

        res_id = session.next_result_id("binarize_labels")
        summary = {
            "result_id": res_id,
            "labels": label_names,
            "shape": list(matrix.shape),
            "counts_per_label": counts_per_label,
            "artifact_key": "binarized_labels",
        }

        session.store_result(
            tool_name="binarize_labels_tool",
            args={},
            summary=summary,
        )

        return summary

    return [reduce_dimensions_tool, binarize_labels_tool]
