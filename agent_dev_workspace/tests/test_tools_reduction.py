import json
import numpy as np
import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools as make_dtm_tools
from tools_reduction import make_tools as make_reduction_tools


def test_reduce_dimensions_pca():
    session = SessionState()

    # Load premade load_dataset_tool and load sample_fixture.csv
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(
        t for t in premade_tools if t.name == "load_dataset_tool"
    )
    load_dataset_tool.invoke(
        {
            "file_path": "agent_dev/sample_fixture.csv",
            "text_column": "text",
            "label_column": "label",
        }
    )

    # Build DTM
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    # Run reduce_dimensions_tool with PCA
    reduction_tools = make_reduction_tools(session)
    reduce_dimensions_tool = next(
        t for t in reduction_tools if t.name == "reduce_dimensions_tool"
    )

    res = reduce_dimensions_tool.invoke({"method": "pca", "n_components": 2})

    # Known answer #7 expected values
    expected_coords = np.array(
        [
            [2.3669, 0.8871],
            [1.7039, 0.2642],
            [1.0408, -0.3587],
            [-2.0759, 1.0129],
            [-1.4554, 0.3405],
            [-0.8349, -0.3319],
            [-0.8349, -0.3319],
            [0.0894, -1.4822],
        ]
    )
    expected_explained_variance = [0.7532, 0.1965]

    coords = np.array(session.artifacts["reduced_coords_pca"])
    assert np.allclose(coords, expected_coords, atol=1e-3)
    assert np.allclose(
        res["explained_variance_ratio"], expected_explained_variance, atol=1e-3
    )
    assert session.pending_figure is not None


# Exact coordinates of Known Answer #8 are not asserted because t-SNE embeddings depend on scikit-learn library versions.
def test_reduce_dimensions_tsne():
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(
        t for t in premade_tools if t.name == "load_dataset_tool"
    )
    load_dataset_tool.invoke(
        {
            "file_path": "agent_dev/sample_fixture.csv",
            "text_column": "text",
            "label_column": "label",
        }
    )

    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    reduction_tools = make_reduction_tools(session)
    reduce_dimensions_tool = next(
        t for t in reduction_tools if t.name == "reduce_dimensions_tool"
    )

    res1 = reduce_dimensions_tool.invoke(
        {"method": "tsne", "perplexity": 2, "random_state": 42}
    )

    coords1 = np.array(session.artifacts["reduced_coords_tsne"])

    # Shape and finite values check
    assert coords1.shape == (8, 2)
    assert np.all(np.isfinite(coords1))

    # Summary checks
    assert len(res1["coordinates_preview"]) == 5
    assert res1["n_samples"] == 8

    # Reproducibility check with same random_state
    res2 = reduce_dimensions_tool.invoke(
        {"method": "tsne", "perplexity": 2, "random_state": 42}
    )
    coords2 = np.array(session.artifacts["reduced_coords_tsne"])
    assert np.allclose(coords1, coords2, atol=1e-3)

    # Duplicate rows (5 and 6) proximity check relative to overall spread
    dist_5_6 = np.linalg.norm(coords1[5] - coords1[6])
    max_pairwise_dist = np.max(
        np.linalg.norm(coords1[:, None, :] - coords1[None, :, :], axis=-1)
    )
    assert dist_5_6 < 0.10 * max_pairwise_dist

    # Plot figure check
    assert session.pending_figure is not None


# Exact coordinates of Known Answer #9 are not asserted because UMAP embeddings depend on umap-learn library versions.
def test_reduce_dimensions_umap():
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(
        t for t in premade_tools if t.name == "load_dataset_tool"
    )
    load_dataset_tool.invoke(
        {
            "file_path": "agent_dev/sample_fixture.csv",
            "text_column": "text",
            "label_column": "label",
        }
    )

    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    reduction_tools = make_reduction_tools(session)
    reduce_dimensions_tool = next(
        t for t in reduction_tools if t.name == "reduce_dimensions_tool"
    )

    res1 = reduce_dimensions_tool.invoke(
        {"method": "umap", "n_neighbors": 3, "random_state": 42}
    )

    coords1 = np.array(session.artifacts["reduced_coords_umap"])

    # Shape and finite values check
    assert coords1.shape == (8, 2)
    assert np.all(np.isfinite(coords1))

    # Summary checks
    assert len(res1["coordinates_preview"]) == 5
    assert res1["n_samples"] == 8

    # Reproducibility check with same random_state
    res2 = reduce_dimensions_tool.invoke(
        {"method": "umap", "n_neighbors": 3, "random_state": 42}
    )
    coords2 = np.array(session.artifacts["reduced_coords_umap"])
    assert np.allclose(coords1, coords2, atol=1e-3)

    # Duplicate rows (5 and 6) proximity check relative to overall spread (threshold 0.25)
    dist_5_6 = np.linalg.norm(coords1[5] - coords1[6])
    max_pairwise_dist = np.max(
        np.linalg.norm(coords1[:, None, :] - coords1[None, :, :], axis=-1)
    )
    assert dist_5_6 < 0.25 * max_pairwise_dist

    # Plot figure check
    assert session.pending_figure is not None


def test_reduce_dimensions_error_handling():
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(
        t for t in premade_tools if t.name == "load_dataset_tool"
    )
    load_dataset_tool.invoke(
        {
            "file_path": "agent_dev/sample_fixture.csv",
            "text_column": "text",
            "label_column": "label",
        }
    )

    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    reduction_tools = make_reduction_tools(session)
    reduce_dimensions_tool = next(
        t for t in reduction_tools if t.name == "reduce_dimensions_tool"
    )

    # Unknown method
    res_unknown = reduce_dimensions_tool.invoke({"method": "invalid_method"})
    assert "error" in res_unknown

    # Perplexity >= n_samples (8 samples)
    res_perplexity = reduce_dimensions_tool.invoke(
        {"method": "tsne", "perplexity": 8.0}
    )
    assert "error" in res_perplexity


def test_binarize_labels_tool():
    # Test error handling when no dataset is loaded
    fresh_session = SessionState()
    reduction_tools_empty = make_reduction_tools(fresh_session)
    binarize_tool_empty = next(
        t for t in reduction_tools_empty if t.name == "binarize_labels_tool"
    )
    res_err = binarize_tool_empty.invoke({})
    assert "error" in res_err

    # Test standard execution on fixture data
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(
        t for t in premade_tools if t.name == "load_dataset_tool"
    )
    load_dataset_tool.invoke(
        {
            "file_path": "agent_dev/sample_fixture.csv",
            "text_column": "text",
            "label_column": "label",
        }
    )

    reduction_tools = make_reduction_tools(session)
    binarize_labels_tool = next(
        t for t in reduction_tools if t.name == "binarize_labels_tool"
    )

    res = binarize_labels_tool.invoke({})

    # Summary checks
    assert res["labels"] == ["catA", "catB"]
    assert res["shape"] == [8, 2]
    assert res["counts_per_label"] == {"catA": 4, "catB": 4}

    # JSON serializability check of summary
    serialized = json.dumps(res)
    assert isinstance(serialized, str)

    # Matrix artifact checks
    matrix = session.artifacts["binarized_labels"]
    assert isinstance(matrix, np.ndarray)
    assert np.issubdtype(matrix.dtype, np.integer)
    assert matrix.shape == (8, 2)
    assert np.all(matrix.sum(axis=1) == 1)

    # DataFrame immutability check
    assert session.dataframe is not None
    assert len(session.dataframe) == 8
