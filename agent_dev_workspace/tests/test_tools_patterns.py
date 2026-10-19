import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_patterns import make_tools

def setup_fixture_session():
    """Helper to create a fresh SessionState and load sample_fixture.csv."""
    session = SessionState()
    premade, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    tools = make_tools(session)
    mine_patterns_tool = next(t for t in tools if t.name == "mine_patterns_tool")
    return session, mine_patterns_tool

def test_fpgrowth_tfidf_known_answer_17():
    """Known Answer #17: FPGrowth with tfidf filtering."""
    session, miner = setup_fixture_session()

    # catA -> tfidf keeps 'alpha' (support 3), found at min_sup 1, 2, 3
    for ms in [1, 2, 3]:
        res = miner.invoke({
            "category_name": "catA",
            "filtering_method": "tfidf",
            "algorithm": "fpgrowth",
            "min_sup": ms
        })
        assert res["n_patterns_found"] == 1
        assert res["preview"][0]["pattern"] == ["alpha"]
        assert res["preview"][0]["support"] == 3

    # catA at min_sup = 4 yields 0 patterns
    res_a4 = miner.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth",
        "min_sup": 4
    })
    assert res_a4["n_patterns_found"] == 0

    # catB -> tfidf keeps 'gamma' (support 4), found at min_sup 1, 2, 3, 4
    for ms in [1, 2, 3, 4]:
        res = miner.invoke({
            "category_name": "catB",
            "filtering_method": "tfidf",
            "algorithm": "fpgrowth",
            "min_sup": ms
        })
        assert res["n_patterns_found"] == 1
        assert res["preview"][0]["pattern"] == ["gamma"]
        assert res["preview"][0]["support"] == 4

    # Verify artifacts saved under lowercased key
    assert "patterns_cata_tfidf_fpgrowth" in session.artifacts
    assert len(session.artifacts["patterns_cata_tfidf_fpgrowth"]) == 0  # from min_sup=4 call
    assert "patterns_catb_tfidf_fpgrowth" in session.artifacts
    assert len(session.artifacts["patterns_catb_tfidf_fpgrowth"]) == 1

def test_zero_terms_survived_variance_and_tf():
    """Known Answer #17: variance and term_frequency keep zero terms for both categories."""
    _, miner = setup_fixture_session()

    for cat in ["catA", "catB"]:
        for fmt in ["variance", "term_frequency"]:
            res = miner.invoke({
                "category_name": cat,
                "filtering_method": fmt,
                "algorithm": "fpgrowth",
                "min_sup": 1
            })
            assert res["n_patterns_found"] == 0
            assert "note" in res
            assert res["note"] == "no terms survived filtering"

def test_topk_known_answer_20():
    """Known Answer #20: topk (FAE) at k=1, 2, 3 returns 1 pattern for both categories."""
    _, miner = setup_fixture_session()

    for k_val in [1, 2, 3]:
        # catA -> alpha (support 3)
        res_a = miner.invoke({
            "category_name": "catA",
            "filtering_method": "tfidf",
            "algorithm": "topk",
            "k": k_val
        })
        assert res_a["n_patterns_found"] == 1
        assert res_a["preview"][0]["pattern"] == ["alpha"]
        assert res_a["preview"][0]["support"] == 3

        # catB -> gamma (support 4)
        res_b = miner.invoke({
            "category_name": "catB",
            "filtering_method": "tfidf",
            "algorithm": "topk",
            "k": k_val
        })
        assert res_b["n_patterns_found"] == 1
        assert res_b["preview"][0]["pattern"] == ["gamma"]
        assert res_b["preview"][0]["support"] == 4

def test_maxfpgrowth_known_answer_20():
    """Known Answer #20: maxfpgrowth returns 0 patterns at min_sup 1 and 4 for both categories."""
    _, miner = setup_fixture_session()

    for cat in ["catA", "catB"]:
        for ms in [1, 4]:
            res = miner.invoke({
                "category_name": cat,
                "filtering_method": "tfidf",
                "algorithm": "maxfpgrowth",
                "min_sup": ms
            })
            assert res["n_patterns_found"] == 0

def test_error_cases():
    """Upfront error validation checks."""
    _, miner = setup_fixture_session()

    # Unknown category (must return available_categories list)
    res_cat = miner.invoke({
        "category_name": "non_existent_category",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth",
        "min_sup": 1
    })
    assert "error" in res_cat
    assert "available_categories" in res_cat
    assert res_cat["available_categories"] == ["catA", "catB"]

    # Bad filtering_method
    res_fmt = miner.invoke({
        "category_name": "catA",
        "filtering_method": "invalid_filtering",
        "algorithm": "fpgrowth",
        "min_sup": 1
    })
    assert "error" in res_fmt

    # Bad algorithm
    res_algo = miner.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "invalid_algo",
        "min_sup": 1
    })
    assert "error" in res_algo

    # Missing / non-positive min_sup for fpgrowth
    res_sup = miner.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth"
    })
    assert "error" in res_sup

    # Missing / non-positive k for topk
    res_k = miner.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "topk"
    })
    assert "error" in res_k
