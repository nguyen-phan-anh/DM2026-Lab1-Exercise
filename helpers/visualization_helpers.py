"""Plotting helpers for PAMI stats and dimensionality reduction."""

from __future__ import annotations

import os
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401, registers 3D projection
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

try:
    import umap
except ImportError:
    umap = None

DEFAULT_CATEGORY_COLORS = ["coral", "blue", "black", "orange"]
DEFAULT_CAMERA_ANGLES = [(30, 45), (45, 135), (60, 225)]


def custom_plot_graphs(
    self,
    allTicks: bool = False,
    max_items: Optional[int] = 200,
    show_all: bool = False,
    static: bool = False,
    output_dir: Optional[str] = None,
    file_name: Optional[str] = None,
    save_format: str = "html",
    show_in_notebook: bool = True,
) -> None:
    """Overridden PAMI plotGraphs: cap items and save figures to disk."""
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    def _save_figure(fig, suffix: str):
        if not output_dir:
            return
        prefix = f"{file_name}_" if file_name else ""
        ext = save_format.lower().lstrip(".")
        target_path = os.path.join(output_dir, f"{prefix}{suffix}.{ext}")
        if ext == "html":
            fig.write_html(target_path)
        else:
            fig.write_image(target_path)
        print(f"Saved: {target_path}")

    itemFrequencies = self.getSortedListOfItemFrequencies()
    if not itemFrequencies:
        print("Warning: Vocabulary is empty. Skipping item frequency graph.")
    else:
        total_unique_items = len(itemFrequencies)
        if show_all or max_items is None:
            items = list(itemFrequencies.keys())
            frequencies = list(itemFrequencies.values())
            chart_title = f"Item Frequencies (All {total_unique_items} Items)"
        else:
            items = list(itemFrequencies.keys())[:max_items]
            frequencies = list(itemFrequencies.values())[:max_items]
            chart_title = f"Item Frequencies (Top {len(items)} of {total_unique_items})"

        ranks = list(range(1, len(items) + 1))
        itemDf = pd.DataFrame({"Rank": ranks, "Item": items, "Frequency": frequencies})
        show_markers = len(items) <= 50
        itemFig = px.line(
            itemDf,
            x="Rank",
            y="Frequency",
            markers=show_markers,
            title=chart_title,
            hover_name="Item",
            hover_data={"Item": False, "Rank": True, "Frequency": True},
        )
        itemFig.update_layout(xaxis_title="No of items (rank)", yaxis_title="Frequency")
        _save_figure(itemFig, "item_frequencies")
        if show_in_notebook:
            itemFig.show(renderer="png" if static else None)

    trx_len_dist = self.getTransanctionalLengthDistribution()
    if trx_len_dist:
        lengths = list(trx_len_dist.keys())
        counts = list(trx_len_dist.values())
        lengthDf = pd.DataFrame({"Length": lengths, "Frequency": counts})
        lengthFig = px.line(
            lengthDf,
            x="Length",
            y="Frequency",
            markers=True,
            title="Transaction Length Distribution",
            hover_data={"Length": True, "Frequency": True},
        )
        lengthFig.update_layout(xaxis_title="Length (#items)", yaxis_title="Frequency")
        if allTicks:
            lengthFig.update_xaxes(tickmode="array", tickvals=lengths)
        _save_figure(lengthFig, "transaction_length")
        if show_in_notebook:
            lengthFig.show(renderer="png" if static else None)


def register_pami_custom_plot_graphs():
    """Bind custom_plot_graphs onto PAMI TransactionalDatabase."""
    from PAMI.extras.dbStats import TransactionalDatabase as tds

    tds.TransactionalDatabase.plotGraphs = custom_plot_graphs


def _category_mask(category_labels: pd.Series, category: str) -> np.ndarray:
    return (category_labels == category).values


def plot_scatter_2d(ax, X_reduced, category_labels, categories, title, colors=None):
    """Single 2D scatter subplot colored by category."""
    colors = colors or DEFAULT_CATEGORY_COLORS
    for c, category in zip(colors, categories):
        mask = _category_mask(category_labels, category)
        ax.scatter(
            X_reduced[mask, 0],
            X_reduced[mask, 1],
            c=c,
            marker="o",
            label=category,
            alpha=0.7,
            s=50,
        )
    ax.grid(color="gray", linestyle=":", linewidth=1, alpha=0.3)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Dimension 1", fontsize=11)
    ax.set_ylabel("Dimension 2", fontsize=11)


def compute_dimred_2d_for_filters(augmented_dfs, random_state=42):
    """Run PCA, t-SNE, UMAP (2D) for each filter key in augmented_dfs."""
    if umap is None:
        raise ImportError("umap-learn is required for UMAP visualizations")

    dimred_results = {}
    for filter_name, augmented_df in augmented_dfs.items():
        dimred_results[filter_name] = {
            "pca": PCA(n_components=2).fit_transform(augmented_df.values),
            "tsne": TSNE(n_components=2, random_state=random_state).fit_transform(
                augmented_df.values
            ),
            "umap": umap.UMAP(n_components=2, random_state=random_state).fit_transform(
                augmented_df.values
            ),
        }
    return dimred_results


def plot_dimred_2d_comparison_grid(
    dimred_results,
    category_labels,
    output_path="./output_files/dimensionality_reduction/all_filters_comparison_3x3.png",
    show=True,
):
    """Create 3×3 grid: rows = filters, columns = PCA / t-SNE / UMAP."""
    categories_list = category_labels.unique()
    col = DEFAULT_CATEGORY_COLORS

    fig, axes = plt.subplots(3, 3, figsize=(30, 30))
    fig.suptitle(
        "Dimensionality Reduction Comparison Across Filtering Methods",
        fontsize=20,
        fontweight="bold",
        y=0.995,
    )

    filter_names_display = ["VARIANCE", "TF-IDF", "TERM FREQUENCY"]
    filter_keys = ["variance", "tfidf", "term_freq"]
    dimred_techniques = ["pca", "tsne", "umap"]
    dimred_names = ["PCA", "t-SNE", "UMAP"]

    for row, (filter_name, filter_display) in enumerate(zip(filter_keys, filter_names_display)):
        for col_idx, (technique, tech_name) in enumerate(zip(dimred_techniques, dimred_names)):
            X_reduced = dimred_results[filter_name][technique]
            plot_scatter_2d(
                axes[row, col_idx],
                X_reduced,
                category_labels,
                categories_list,
                tech_name,
                col,
            )
            if col_idx == 0:
                axes[row, col_idx].text(
                    -0.15,
                    0.5,
                    filter_display,
                    transform=axes[row, col_idx].transAxes,
                    fontsize=14,
                    fontweight="bold",
                    rotation=90,
                    verticalalignment="center",
                    horizontalalignment="center",
                )
            if row == 0 and col_idx == 2:
                axes[row, col_idx].legend(loc="upper right", fontsize=10, framealpha=0.9)

    plt.tight_layout(rect=[0, 0, 1, 0.99])
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()
    return output_path


def create_3d_scatter(ax, X_3d, category_labels, categories, title, colors=None):
    """Draw 3D scatter on a given Axes3D."""
    colors = colors or DEFAULT_CATEGORY_COLORS
    for c, category in zip(colors, categories):
        mask = _category_mask(category_labels, category)
        ax.scatter(
            X_3d[mask, 0],
            X_3d[mask, 1],
            X_3d[mask, 2],
            c=c,
            marker="o",
            label=category,
            alpha=0.7,
            s=40,
        )
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    ax.set_zlabel("Dim 3")


def compute_dimred_3d(augmented_df, random_state=42):
    """PCA, t-SNE, UMAP with 3 components."""
    if umap is None:
        raise ImportError("umap-learn is required for UMAP visualizations")
    return {
        "pca": PCA(n_components=3).fit_transform(augmented_df.values),
        "tsne": TSNE(n_components=3, random_state=random_state).fit_transform(
            augmented_df.values
        ),
        "umap": umap.UMAP(n_components=3, random_state=random_state).fit_transform(
            augmented_df.values
        ),
    }


def plot_dimred_3d_multi_angle(
    X_pca_3d,
    X_tsne_3d,
    X_umap_3d,
    category_labels,
    filter_name,
    output_dir="./output_files/dimensionality_reduction/3d_plots",
    camera_angles: Sequence[tuple] = DEFAULT_CAMERA_ANGLES,
    show=False,
):
    """
    One figure per filter: 3×3 grid (methods × camera angles). Saves PNG.
    """
    os.makedirs(output_dir, exist_ok=True)
    categories = category_labels.unique()
    methods = [("PCA", X_pca_3d), ("t-SNE", X_tsne_3d), ("UMAP", X_umap_3d)]

    fig = plt.figure(figsize=(24, 24))
    fig.suptitle(
        f"3D Dimensionality Reduction: {filter_name.replace('_', ' ').title()}",
        fontsize=18,
        fontweight="bold",
        y=0.995,
    )

    for row, (elev, azim) in enumerate(camera_angles):
        for col, (method_name, X_3d) in enumerate(methods):
            ax = fig.add_subplot(3, 3, row * 3 + col + 1, projection="3d")
            create_3d_scatter(
                ax,
                X_3d,
                category_labels,
                categories,
                f"{method_name} (elev={elev}°, azim={azim}°)",
            )
            ax.view_init(elev=elev, azim=azim)
            if row == 0 and col == 2:
                ax.legend(loc="upper left", fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.99])
    out_path = os.path.join(output_dir, f"{filter_name}_3d_comparison_angles.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()
    return out_path


def run_all_3d_dimred_plots(augmented_dfs, category_labels, random_state=42, show=False):
    """Generate 3D plots for variance, tfidf, and term_freq (3 angles × 3 methods each)."""
    paths = []
    for filter_name in ["variance", "tfidf", "term_freq"]:
        coords = compute_dimred_3d(augmented_dfs[filter_name], random_state=random_state)
        path = plot_dimred_3d_multi_angle(
            coords["pca"],
            coords["tsne"],
            coords["umap"],
            category_labels,
            filter_name,
            show=show,
        )
        paths.append(path)
        print(f"Saved 3D comparison: {path}")
    return paths
