import numpy as np
import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools as make_dtm_tools
from tools_filtering import make_tools as make_filtering_tools


def test_variance_filter_tool():
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

    # Get variance_filter_tool
    filtering_tools = make_filtering_tools(session)
    variance_filter_tool = next(
        t for t in filtering_tools if t.name == "variance_filter_tool"
    )

    # First call: full preview (preview_cap=10)
    res_full = variance_filter_tool.invoke({"threshold": 0.15, "preview_cap": 10})

    assert res_full["n_total_terms"] == 5
    assert res_full["n_kept"] == 4
    assert res_full["n_removed"] == 1

    # Check kept/removed terms in full preview
    # Kept: alpha (1.1875), gamma (1.109375), delta (0.25), beta (0.234375)
    expected_kept = ["alpha", "gamma", "delta", "beta"]
    assert res_full["kept_terms_sample"] == expected_kept
    assert res_full["removed_terms_sample"] == ["always"]

    # Retrieve full_report from session.results_store
    result_id = res_full["result_id"]
    stored_entry = session.results_store[result_id]
    full_report_df = stored_entry["full_report"]

    # Verify 1st col is term, 2nd col is variance
    assert list(full_report_df.columns) == ["term", "variance"]

    # Check per-term variance values against Known Answer #5 in TEST_FIXTURE.md
    expected_variances = {
        "alpha": 1.1875,
        "always": 0.109375,
        "beta": 0.234375,
        "delta": 0.25,
        "gamma": 1.109375,
    }

    actual_variances = dict(zip(full_report_df["term"], full_report_df["variance"]))
    for term, exp_var in expected_variances.items():
        assert np.isclose(actual_variances[term], exp_var, atol=1e-3)

    # Second call: truncated preview (preview_cap=2)
    res_trunc = variance_filter_tool.invoke({"threshold": 0.15, "preview_cap": 2})

    assert res_trunc["n_total_terms"] == 5
    assert res_trunc["n_kept"] == 4
    assert res_trunc["n_removed"] == 1
    assert len(res_trunc["kept_terms_sample"]) == 2
    assert res_trunc["kept_terms_sample"] == ["alpha", "gamma"]


def test_pearson_filter_tool():
    # Fresh session state for test isolated from previous tests
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

    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    filtering_tools = make_filtering_tools(session)
    pearson_filter_tool = next(
        t for t in filtering_tools if t.name == "pearson_filter_tool"
    )

    # Call pearson_filter_tool for target_class="catB"
    res = pearson_filter_tool.invoke({"target_class": "catB", "preview_cap": 10})

    assert res["n_total_terms"] == 5
    assert res["target_class"] == "catB"

    # Retrieve full_report from session.results_store
    result_id = res["result_id"]
    stored_entry = session.results_store[result_id]
    full_report_df = stored_entry["full_report"]

    # Verify column layout: 1st column "term", 2nd column "pearson_r", 3rd column "pearson_p"
    assert list(full_report_df.columns) == ["term", "pearson_r", "pearson_p"]

    # Verify ranking by absolute correlation: delta, gamma, beta, alpha, always
    expected_order = ["delta", "gamma", "beta", "alpha", "always"]
    assert full_report_df["term"].tolist() == expected_order

    # Known answer 6 in TEST_FIXTURE.md for target_class="catB"
    expected_pearson_r = {
        "delta": 1.0000,
        "gamma": 0.8307,
        "beta": -0.7746,
        "alpha": -0.6882,
        "always": 0.3780,
    }

    actual_r = dict(zip(full_report_df["term"], full_report_df["pearson_r"]))
    for term, exp_r in expected_pearson_r.items():
        assert np.isclose(actual_r[term], exp_r, atol=1e-3)

    # Verify p-value for delta (perfect correlation r=1.0 -> p=0.0, valid float)
    delta_row = full_report_df[full_report_df["term"] == "delta"].iloc[0]
    assert isinstance(float(delta_row["pearson_p"]), float)
    assert not np.isnan(delta_row["pearson_p"])
    assert delta_row["pearson_p"] >= 0.0

    # Verify error raised for unknown target class
    with pytest.raises(ValueError, match="Target class 'non_existent_class' not found"):
        pearson_filter_tool.invoke({"target_class": "non_existent_class"})


def test_spearman_filter_tool():
    # Fresh session state for test isolated from previous tests
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

    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    filtering_tools = make_filtering_tools(session)
    spearman_filter_tool = next(
        t for t in filtering_tools if t.name == "spearman_filter_tool"
    )

    # Call spearman_filter_tool for target_class="catB"
    res = spearman_filter_tool.invoke({"target_class": "catB", "preview_cap": 10})

    assert res["n_total_terms"] == 5
    assert res["target_class"] == "catB"

    # Retrieve full_report from session.results_store
    result_id = res["result_id"]
    stored_entry = session.results_store[result_id]
    full_report_df = stored_entry["full_report"]

    # Verify column layout: 1st column "term", 2nd column "spearman_r", 3rd column "spearman_p"
    assert list(full_report_df.columns) == ["term", "spearman_r", "spearman_p"]

    # Verify ranking by absolute correlation: delta, gamma, beta, alpha, always
    expected_order = ["delta", "gamma", "beta", "alpha", "always"]
    assert full_report_df["term"].tolist() == expected_order

    # Known answer 6 in TEST_FIXTURE.md for target_class="catB" (Spearman r)
    expected_spearman_r = {
        "delta": 1.0000,
        "gamma": 0.9363,
        "beta": -0.7746,
        "alpha": -0.7500,
        "always": 0.3780,
    }

    actual_r = dict(zip(full_report_df["term"], full_report_df["spearman_r"]))
    for term, exp_r in expected_spearman_r.items():
        assert np.isclose(actual_r[term], exp_r, atol=1e-4)

    # Verify p-value for delta (perfect rank correlation r=1.0 -> p=0.0, valid float)
    delta_row = full_report_df[full_report_df["term"] == "delta"].iloc[0]
    assert isinstance(float(delta_row["spearman_p"]), float)
    assert not np.isnan(delta_row["spearman_p"])
    assert delta_row["spearman_p"] >= 0.0

    # Verify error raised for unknown target class
    with pytest.raises(ValueError, match="Target class 'non_existent_class' not found"):
        spearman_filter_tool.invoke({"target_class": "non_existent_class"})
