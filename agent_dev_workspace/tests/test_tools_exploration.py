import matplotlib
import numpy as np
import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools as make_dtm_tools
from tools_exploration import make_tools as make_exploration_tools


def test_cosine_similarity_tool():
    session = SessionState()

    # Load premade dataset tool
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    # Load sample fixture
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

    # Get cosine_similarity_tool
    exploration_tools = make_exploration_tools(session)
    cosine_similarity_tool = next(
        t for t in exploration_tools if t.name == "cosine_similarity_tool"
    )

    # Known answer 16 in TEST_FIXTURE.md:
    # 0 vs 1 (both catA, near-identical text): 0.9847
    res_0_1 = cosine_similarity_tool.invoke({"doc1_idx": 0, "doc2_idx": 1})
    assert np.isclose(res_0_1["similarity"], 0.9847, atol=1e-3)
    assert res_0_1["category1"] == "catA"
    assert res_0_1["category2"] == "catA"

    # 0 vs 3 (catA vs catB, share only "always"): 0.0909
    res_0_3 = cosine_similarity_tool.invoke({"doc1_idx": 0, "doc2_idx": 3})
    assert np.isclose(res_0_3["similarity"], 0.0909, atol=1e-3)

    # 3 vs 4 (both catB, near-identical text): 0.9847
    res_3_4 = cosine_similarity_tool.invoke({"doc1_idx": 3, "doc2_idx": 4})
    assert np.isclose(res_3_4["similarity"], 0.9847, atol=1e-3)

    # 0 vs 7 (catA vs empty text row): exactly 0.0
    res_0_7 = cosine_similarity_tool.invoke({"doc1_idx": 0, "doc2_idx": 7})
    assert res_0_7["similarity"] == 0.0

    # Verify out-of-bounds error
    with pytest.raises(ValueError, match="doc1_idx 99 out of bounds"):
        cosine_similarity_tool.invoke({"doc1_idx": 99, "doc2_idx": 0})


def test_feature_correlation_matrix_tool():
    session = SessionState()

    # Load premade dataset tool
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke(
        {
            "file_path": "agent_dev/sample_fixture.csv",
            "text_column": "text",
            "label_column": "label",
        }
    )

    # Check uninitialized state returns error dict instead of raising
    exploration_tools_uninit = make_exploration_tools(session)
    feature_corr_tool_uninit = next(
        t for t in exploration_tools_uninit if t.name == "feature_correlation_matrix_tool"
    )
    err_res = feature_corr_tool_uninit.invoke({})
    assert "error" in err_res

    # Build DTM
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    exploration_tools = make_exploration_tools(session)
    feature_correlation_matrix_tool = next(
        t for t in exploration_tools if t.name == "feature_correlation_matrix_tool"
    )

    # Call feature_correlation_matrix_tool
    res = feature_correlation_matrix_tool.invoke({"top_n": 20})

    assert res["n_terms"] == 5

    # Known answer 19 in TEST_FIXTURE.md term order
    expected_terms = ["alpha", "gamma", "delta", "beta", "always"]
    assert res["terms"] == expected_terms

    # Known answer 19 in TEST_FIXTURE.md 5x5 matrix
    expected_matrix = [
        [1.0000, -0.5718, -0.6882, 0.8885, 0.2601],
        [-0.5718, 1.0000, 0.8307, -0.6435, 0.3140],
        [-0.6882, 0.8307, 1.0000, -0.7746, 0.3780],
        [0.8885, -0.6435, -0.7746, 1.0000, 0.2928],
        [0.2601, 0.3140, 0.3780, 0.2928, 1.0000],
    ]

    actual_matrix = res["correlation_matrix"]
    assert len(actual_matrix) == 5
    for i in range(5):
        assert len(actual_matrix[i]) == 5
        for j in range(5):
            assert np.isclose(actual_matrix[i][j], expected_matrix[i][j], atol=1e-3)

    # Check session.pending_figure is set to a matplotlib Figure
    assert session.pending_figure is not None
    assert isinstance(session.pending_figure, matplotlib.figure.Figure)
