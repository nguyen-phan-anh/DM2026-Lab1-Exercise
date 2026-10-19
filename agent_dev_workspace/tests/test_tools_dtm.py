import matplotlib.pyplot as plt
import numpy as np
import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools


def test_build_dtm_tool():
    # Instantiate single session state
    session = SessionState()

    # Load premade dataset tool
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    # Load sample fixture
    load_res = load_dataset_tool.invoke(
        {
            "file_path": "agent_dev/sample_fixture.csv",
            "text_column": "text",
            "label_column": "label",
        }
    )
    assert load_res["n_documents"] == 8

    # Load student's build_dtm_tool
    tools = make_tools(session)
    build_dtm_tool = next(t for t in tools if t.name == "build_dtm_tool")

    # Run build_dtm_tool
    res = build_dtm_tool.invoke({})

    # Check against Known Answer #4 in TEST_FIXTURE.md
    assert res["n_documents"] == 8
    assert res["n_terms"] == 5
    assert res["non_zero"] == 21
    assert res["total_elements"] == 40
    assert np.isclose(res["sparsity_pct"], 47.5, atol=1e-3)

    # Verify feature names list in session state
    expected_vocab = ["alpha", "always", "beta", "delta", "gamma"]
    assert session.feature_names == expected_vocab

    # Verify count vectorizer artifact was stored
    assert "count_vectorizer" in session.artifacts


def test_term_frequency_tool():
    # Instantiate single session state for independent test execution
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke(
        {
            "file_path": "agent_dev/sample_fixture.csv",
            "text_column": "text",
            "label_column": "label",
        }
    )

    tools = make_tools(session)
    build_dtm_tool = next(t for t in tools if t.name == "build_dtm_tool")
    term_frequency_tool = next(t for t in tools if t.name == "term_frequency_tool")

    # Build DTM first
    build_dtm_tool.invoke({})

    # Run term frequency tool
    res = term_frequency_tool.invoke({"top_n": 10})

    # Assertions based on Known Answer #14 in TEST_FIXTURE.md
    assert res["total_unique_terms"] == 5
    assert res["total_occurrences"] == 27

    # Check top terms and frequencies
    expected_counts = {
        "alpha": 6,
        "always": 7,
        "beta": 3,
        "delta": 4,
        "gamma": 7,
    }

    retrieved_counts = {item["term"]: item["frequency"] for item in res["top_terms"]}
    assert retrieved_counts == expected_counts

    # Check deterministic order for top terms (frequency desc, term asc)
    # always: 7, gamma: 7, alpha: 6, delta: 4, beta: 3
    expected_ordered_terms = ["always", "gamma", "alpha", "delta", "beta"]
    actual_ordered_terms = [item["term"] for item in res["top_terms"]]
    assert actual_ordered_terms == expected_ordered_terms


def test_dtm_heatmap_tool():
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke(
        {
            "file_path": "agent_dev/sample_fixture.csv",
            "text_column": "text",
            "label_column": "label",
        }
    )

    tools = make_tools(session)
    build_dtm_tool = next(t for t in tools if t.name == "build_dtm_tool")
    dtm_heatmap_tool = next(t for t in tools if t.name == "dtm_heatmap_tool")

    # Build DTM first
    build_dtm_tool.invoke({})

    # Test default parameters (n_terms=20, n_documents=20) -> should return full 8x5 matrix
    res_default = dtm_heatmap_tool.invoke({})

    assert res_default["n_documents"] == 8
    assert res_default["n_terms"] == 5
    assert res_default["terms"] == ["alpha", "always", "beta", "delta", "gamma"]
    assert res_default["document_labels"] == ["0", "1", "2", "3", "4", "5", "6", "7"]

    # Check full matrix values against Known Answer #21 in TEST_FIXTURE.md
    expected_full_matrix = [
        [3, 1, 1, 0, 0],
        [2, 1, 1, 0, 0],
        [1, 1, 1, 0, 0],
        [0, 1, 0, 1, 3],
        [0, 1, 0, 1, 2],
        [0, 1, 0, 1, 1],
        [0, 1, 0, 1, 1],
        [0, 0, 0, 0, 0],
    ]
    assert res_default["matrix"] == expected_full_matrix

    # Verify session.pending_figure is set and is a matplotlib figure
    assert session.pending_figure is not None
    assert isinstance(session.pending_figure, plt.Figure)

    # Test custom slice parameters (n_terms=2, n_documents=3)
    res_slice = dtm_heatmap_tool.invoke({"n_terms": 2, "n_documents": 3})

    assert res_slice["n_documents"] == 3
    assert res_slice["n_terms"] == 2
    assert res_slice["terms"] == ["alpha", "always"]
    assert res_slice["document_labels"] == ["0", "1", "2"]

    # Expected top-left 3x2 slice: [[3, 1], [2, 1], [1, 1]]
    expected_slice_matrix = [
        [3, 1],
        [2, 1],
        [1, 1],
    ]
    assert res_slice["matrix"] == expected_slice_matrix
