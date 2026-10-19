import json
import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_data import make_tools as make_data_tools


def test_describe_data_tool():
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

    # Get describe_data_tool
    data_tools = make_data_tools(session)
    describe_data_tool = next(
        t for t in data_tools if t.name == "describe_data_tool"
    )

    # Call describe_data_tool
    res = describe_data_tool.invoke({})

    # 1. Assert overall stats against Known Answer #18
    overall = res["overall_stats"]
    assert overall["count"] == 8
    assert np.isclose(overall["mean"], 19.875, atol=1e-4)
    assert np.isclose(overall["std"], 9.4330, atol=1e-4)
    assert overall["min"] == 0.0
    assert np.isclose(overall["25%"], 17.75, atol=1e-4)
    assert np.isclose(overall["50%"], 20.5, atol=1e-4)
    assert np.isclose(overall["75%"], 25.25, atol=1e-4)
    assert overall["max"] == 30.0

    # 2. Assert per-category stats against Known Answer #18
    cat_stats = res["category_stats"]
    assert "catA" in cat_stats
    assert "catB" in cat_stats

    catA = cat_stats["catA"]
    assert catA["count"] == 4
    assert np.isclose(catA["mean"], 17.25, atol=1e-4)
    assert np.isclose(catA["std"], 12.5, atol=1e-4)
    assert catA["min"] == 0.0
    assert np.isclose(catA["25%"], 12.75, atol=1e-4)
    assert np.isclose(catA["50%"], 20.0, atol=1e-4)
    assert np.isclose(catA["75%"], 24.5, atol=1e-4)
    assert catA["max"] == 29.0

    catB = cat_stats["catB"]
    assert catB["count"] == 4
    assert np.isclose(catB["mean"], 22.5, atol=1e-4)
    assert np.isclose(catB["std"], 5.7446, atol=1e-4)
    assert catB["min"] == 18.0
    assert np.isclose(catB["25%"], 18.0, atol=1e-4)
    assert np.isclose(catB["50%"], 21.0, atol=1e-4)
    assert np.isclose(catB["75%"], 25.5, atol=1e-4)
    assert catB["max"] == 30.0

    # 3. Assert session.pending_figure is populated with a matplotlib Figure instance
    assert session.pending_figure is not None
    assert isinstance(session.pending_figure, Figure)


def test_describe_data_tool_uninitialized():
    # Fresh session with no dataset loaded
    session = SessionState()
    data_tools = make_data_tools(session)
    describe_data_tool = next(
        t for t in data_tools if t.name == "describe_data_tool"
    )

    res = describe_data_tool.invoke({})
    assert "error" in res
    assert res["error"] == "No dataset loaded in session. Call load_dataset_tool first."


def test_check_missing_tool_known_answer_2():
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

    data_tools = make_data_tools(session)
    check_missing_tool = next(
        t for t in data_tools if t.name == "check_missing_tool"
    )

    res = check_missing_tool.invoke({})

    # Known answer #2 assertions
    assert res["total_rows"] == 8
    assert res["missing_count"] == 1
    assert res["missing_indices"] == [7]
    assert np.isclose(res["missing_pct"], 12.5, atol=1e-4)

    # Tool must not modify session.dataframe
    assert len(session.dataframe) == 8


def test_check_missing_tool_uninitialized():
    session = SessionState()
    data_tools = make_data_tools(session)
    check_missing_tool = next(
        t for t in data_tools if t.name == "check_missing_tool"
    )

    res = check_missing_tool.invoke({})
    assert "error" in res
    assert res["error"] == "No dataset loaded in session. Call load_dataset_tool first."


def test_check_missing_tool_various_missing_types():
    session = SessionState()
    session.dataframe = pd.DataFrame(
        {"text": ["valid text", "", "   ", None]}
    )

    data_tools = make_data_tools(session)
    check_missing_tool = next(
        t for t in data_tools if t.name == "check_missing_tool"
    )

    res = check_missing_tool.invoke({})
    assert res["total_rows"] == 4
    assert res["missing_count"] == 3
    assert res["missing_indices"] == [1, 2, 3]
    assert np.isclose(res["missing_pct"], 75.0, atol=1e-4)


def test_check_duplicates_tool_known_answer_3_detect():
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

    data_tools = make_data_tools(session)
    check_duplicates_tool = next(
        t for t in data_tools if t.name == "check_duplicates_tool"
    )

    res = check_duplicates_tool.invoke({"drop": False})

    # Known Answer #3 assertions (detect mode)
    assert res["duplicate_count"] == 1
    assert res["duplicate_row_indices"] == [5, 6]
    assert res["total_rows_before"] == 8
    assert res["total_rows_after"] == 8
    assert res["rows_removed"] == 0


def test_check_duplicates_tool_known_answer_3_drop():
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

    data_tools = make_data_tools(session)
    check_duplicates_tool = next(
        t for t in data_tools if t.name == "check_duplicates_tool"
    )

    res = check_duplicates_tool.invoke({"drop": True})

    # Known Answer #3 assertions (drop mode)
    assert res["duplicate_count"] == 1
    assert res["duplicate_row_indices"] == [5, 6]
    assert res["total_rows_before"] == 8
    assert res["total_rows_after"] == 6
    assert res["rows_removed"] == 2
    assert len(session.dataframe) == 6

    # Specific assertions requested:
    # 1. "always gamma delta" is completely gone from session.dataframe["text"]
    assert "always gamma delta" not in session.dataframe["text"].tolist()
    # 2. Index labels were not reset
    assert session.dataframe.index.tolist() == [0, 1, 2, 3, 4, 7]


def test_check_duplicates_tool_uninitialized():
    session = SessionState()
    data_tools = make_data_tools(session)
    check_duplicates_tool = next(
        t for t in data_tools if t.name == "check_duplicates_tool"
    )

    res = check_duplicates_tool.invoke({})
    assert "error" in res
    assert res["error"] == "No dataset loaded in session. Call load_dataset_tool first."


def test_tokenize_tool_known_answer_10():
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

    data_tools = make_data_tools(session)
    tokenize_tool = next(t for t in data_tools if t.name == "tokenize_tool")

    res = tokenize_tool.invoke({"preview_cap": 8})

    expected_tokens = [
        ["always", "alpha", "alpha", "alpha", "beta"],
        ["always", "alpha", "alpha", "beta"],
        ["always", "alpha", "beta"],
        ["always", "gamma", "gamma", "gamma", "delta"],
        ["always", "gamma", "gamma", "delta"],
        ["always", "gamma", "delta"],
        ["always", "gamma", "delta"],
        [],
    ]

    assert res["total_documents"] == 8
    assert res["total_tokens"] == 27
    assert res["unigrams_sample"] == expected_tokens

    # Verify session.dataframe["unigrams"] holds identical token lists
    assert session.dataframe["unigrams"].tolist() == expected_tokens

    # Verify session.artifacts["unigrams"] holds identical token lists
    assert session.artifacts["unigrams"] == expected_tokens

    # Explicitly check empty-text row 7 gives an empty list []
    assert res["unigrams_sample"][7] == []
    assert session.dataframe["unigrams"].iloc[7] == []


def test_tokenize_tool_uninitialized():
    session = SessionState()
    data_tools = make_data_tools(session)
    tokenize_tool = next(t for t in data_tools if t.name == "tokenize_tool")

    res = tokenize_tool.invoke({})
    assert "error" in res
    assert res["error"] == "No dataset loaded in session. Call load_dataset_tool first."


def test_sample_data_tool_known_answer_13():
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

    data_tools = make_data_tools(session)
    sample_data_tool = next(t for t in data_tools if t.name == "sample_data_tool")

    # 1. Call with n=4, random_state=42 (Known Answer 13)
    res = sample_data_tool.invoke({"n": 4, "random_state": 42})

    assert res["total_rows_before"] == 8
    assert res["sampled_count"] == 4
    assert res["sampled_indices"] == [1, 5, 0, 7]
    assert len(res["sampled_rows"]) == 4

    # Verify session.dataframe was not modified
    assert len(session.dataframe) == 8

    # 2. Call with preview_cap=2 to check preview capping
    res_capped = sample_data_tool.invoke({"n": 4, "random_state": 42, "preview_cap": 2})

    assert res_capped["sampled_count"] == 4
    assert res_capped["sampled_indices"] == [1, 5]
    assert len(res_capped["sampled_rows"]) == 2


def test_sample_data_tool_uninitialized():
    session = SessionState()
    data_tools = make_data_tools(session)
    sample_data_tool = next(t for t in data_tools if t.name == "sample_data_tool")

    res = sample_data_tool.invoke({})
    assert "error" in res
    assert res["error"] == "No dataset loaded in session. Call load_dataset_tool first."


def test_list_files_tool_known_answer_11():
    session = SessionState()

    data_tools = make_data_tools(session)
    list_files_tool = next(t for t in data_tools if t.name == "list_files_tool")

    # 1. Known Answer 11 assertion: subdirectory="newdataset"
    res = list_files_tool.invoke({"subdirectory": "newdataset"})

    assert res["subdirectory"] == "newdataset"
    assert res["total_directories"] == 0
    assert res["total_files"] == 1
    assert res["directories"] == []
    assert res["files"] == ["Reddit-stock-sentiment.csv"]

    # 2. Check path traversal containing ".." returns an error dict
    res_traversal = list_files_tool.invoke({"subdirectory": "../"})
    assert "error" in res_traversal
    assert "path traversal" in res_traversal["error"]

    # 3. Check non-existent directory returns an error dict
    res_nonexistent = list_files_tool.invoke({"subdirectory": "non_existent_folder_xyz"})
    assert "error" in res_nonexistent
    assert "does not exist" in res_nonexistent["error"]


def test_inspect_data_tool_custom():
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

    data_tools = make_data_tools(session)
    tokenize_tool = next(t for t in data_tools if t.name == "tokenize_tool")
    inspect_data_tool = next(t for t in data_tools if t.name == "inspect_data_tool")

    # Tokenize first so "unigrams" column exists in session.dataframe
    tokenize_tool.invoke({})

    # 1. Inspect dataset head on fixture
    res = inspect_data_tool.invoke({"n": 5})

    assert res["total_rows"] == 8
    assert res["total_columns"] >= 3

    col_names = [col["name"] for col in res["columns"]]
    assert "text" in col_names
    assert "category_name" in col_names
    assert "category" in col_names
    assert "unigrams" in col_names

    # Check that row 0 unigrams preview matches first 5 tokens of Known Answer 10
    row0_unigrams = res["head_rows"][0]["unigrams"]
    assert row0_unigrams == ["always", "alpha", "alpha", "alpha", "beta"]

    # Check session.dataframe was not modified
    assert len(session.dataframe) == 8

    # 2. Check list truncation note when list_preview_items=2 (row 0 has 5 tokens)
    res_truncated_list = inspect_data_tool.invoke({"n": 5, "list_preview_items": 2})
    row0_trunc_unigrams = res_truncated_list["head_rows"][0]["unigrams"]
    assert isinstance(row0_trunc_unigrams, str)
    assert "(+3 more items)" in row0_trunc_unigrams

    # 3. Check JSON serializability on n=50 call over the fixture (covers row 7 with empty string text "" and [] unigrams)
    res_fifty_fixture = inspect_data_tool.invoke({"n": 50})
    json_str = json.dumps(res_fifty_fixture)
    assert isinstance(json_str, str)
    assert res_fifty_fixture["head_rows"][7]["text"] == ""
    assert res_fifty_fixture["head_rows"][7]["unigrams"] == []

    # 4. Check capping of n at 20 using a custom session with 25 rows
    session_large = SessionState()
    session_large.dataframe = pd.DataFrame(
        {"text": [f"doc {i}" for i in range(25)]}
    )
    data_tools_large = make_data_tools(session_large)
    inspect_large_tool = next(t for t in data_tools_large if t.name == "inspect_data_tool")

    res_capped_20 = inspect_large_tool.invoke({"n": 50})
    assert res_capped_20["total_rows"] == 25
    assert len(res_capped_20["head_rows"]) == 20


def test_inspect_data_tool_uninitialized():
    session = SessionState()
    data_tools = make_data_tools(session)
    inspect_data_tool = next(t for t in data_tools if t.name == "inspect_data_tool")

    res = inspect_data_tool.invoke({})
    assert "error" in res
    assert res["error"] == "No dataset loaded in session. Call load_dataset_tool first."
