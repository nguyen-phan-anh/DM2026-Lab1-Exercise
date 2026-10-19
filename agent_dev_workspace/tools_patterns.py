import os
import tempfile
from typing import Optional
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from langchain_core.tools import tool

import helpers.text_filtering_helpers as tfh
from PAMI.extras.convert import DF2DB
import PAMI.frequentPattern.basic.FPGrowth as fpg
import PAMI.frequentPattern.topk.FAE as topk_fae
import PAMI.frequentPattern.maximal.MaxFPGrowth as maxfpg

import traceback

def make_tools(session):
    @tool
    def mine_patterns_tool(
        category_name: str,
        filtering_method: str,
        algorithm: str = "fpgrowth",
        min_sup: Optional[int] = None,
        k: Optional[int] = None,
    ) -> dict:
        """Mine frequent patterns for a single category using PAMI pattern mining algorithms.

        Args:
            category_name: The category to filter and mine patterns from.
            filtering_method: Method to filter vocabulary ("variance", "tfidf", "term_frequency").
            algorithm: Mining algorithm ("fpgrowth", "topk", "maxfpgrowth"). Default "fpgrowth".
            min_sup: Minimum support threshold (required for "fpgrowth" and "maxfpgrowth", > 0).
            k: Top k patterns count (required for "topk", > 0).
        """
        # Step 1: Upfront Argument & Session Validation
        if session.dataframe is None or session.dataframe.empty:
            return {"error": "No dataframe loaded in session."}

        available_categories = []
        if "category_name" in session.dataframe.columns:
            available_categories = sorted(session.dataframe["category_name"].dropna().unique().tolist())
        elif "category" in session.dataframe.columns:
            available_categories = sorted(session.dataframe["category"].dropna().unique().tolist())

        if category_name not in available_categories:
            return {
                "error": f"Category '{category_name}' not found.",
                "available_categories": available_categories
            }

        valid_filtering = ["variance", "tfidf", "term_frequency"]
        if filtering_method not in valid_filtering:
            return {
                "error": f"Invalid filtering_method '{filtering_method}'. Must be one of {valid_filtering}."
            }

        valid_algorithms = ["fpgrowth", "topk", "maxfpgrowth"]
        if algorithm not in valid_algorithms:
            return {
                "error": f"Invalid algorithm '{algorithm}'. Must be one of {valid_algorithms}."
            }

        if algorithm in ["fpgrowth", "maxfpgrowth"]:
            if min_sup is None or min_sup <= 0:
                return {"error": f"Algorithm '{algorithm}' requires min_sup > 0."}
        elif algorithm == "topk":
            if k is None or k <= 0:
                return {"error": "Algorithm 'topk' requires k > 0."}

        full_args = {
            "category_name": category_name,
            "filtering_method": filtering_method,
            "algorithm": algorithm,
            "min_sup": min_sup,
            "k": k
        }
        artifact_key = f"patterns_{category_name}_{filtering_method}_{algorithm}".lower()

        # Step 2: Extract category dataframe & build per-category DTM
        col = "category_name" if "category_name" in session.dataframe.columns else "category"
        cat_df = session.dataframe[session.dataframe[col] == category_name].copy()

        text_series = (
            cat_df["text"].fillna("").astype(str)
            .str.encode("ascii", errors="ignore").str.decode("ascii")
)
        vectorizer = CountVectorizer(
            stop_words="english",
            strip_accents="unicode",
            token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b",
)
        dtm_mat = vectorizer.fit_transform(text_series)
        #vectorizer = CountVectorizer(stop_words="english")
        #dtm_mat = vectorizer.fit_transform(cat_df["text"].fillna(""))

        feature_names = list(vectorizer.get_feature_names_out())

        if len(feature_names) == 0:
            res_id = session.next_result_id("patterns")
            summary = {
                "result_id": res_id,
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "n_patterns_found": 0,
                "note": "no terms survived stopword removal",
                "preview": []
            }
            session.artifacts[artifact_key] = []
            session.store_result("mine_patterns_tool", full_args, summary)
            return summary

        dtm_df = pd.DataFrame(dtm_mat.toarray(), columns=feature_names)

        # Step 3: Vocabulary Filtering via text_filtering_helpers
        filt_dict = {category_name: dtm_df}
        cats = [category_name]

        if filtering_method == "variance":
            filtered_dict = tfh.apply_variance_filtering(filt_dict, cats)
            filtered_df = filtered_dict[category_name]
        elif filtering_method == "tfidf":
            tfidf_mat = TfidfTransformer().fit_transform(dtm_mat)
            tfidf_df = pd.DataFrame(tfidf_mat.toarray(), columns=feature_names)
            tfidf_dict = {category_name: tfidf_df}
            filtered_dict = tfh.apply_tfidf_filtering(tfidf_dict, cats)
            # Match columns back to count matrix
            kept_cols = filtered_dict[category_name].columns
            filtered_df = dtm_df[kept_cols]
        elif filtering_method == "term_frequency":
            filtered_dict = tfh.apply_term_frequency_filtering(filt_dict, cats)
            filtered_df = filtered_dict[category_name]

        if filtered_df.shape[1] == 0:
            res_id = session.next_result_id("patterns")
            summary = {
                "result_id": res_id,
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "n_patterns_found": 0,
                "note": "no terms survived filtering",
                "preview": []
            }
            session.artifacts[artifact_key] = []
            session.store_result("mine_patterns_tool", full_args, summary)
            return summary

        # Step 4: Transactional DB Conversion & Mining
        patterns = []
        db_fd, db_file = tempfile.mkstemp(suffix=".txt")
        os.close(db_fd)

        try:
            #db_converter = DF2DB.DF2DB(filtered_df, DFtype="dense")
            #db_converter.convert2TransactionalDatabase(db_file, ">=", 1)
            cols = filtered_df.columns.to_numpy()
            values = filtered_df.to_numpy()
            with open(db_file, "w", encoding="utf-8", errors="ignore", newline="\n") as f:
                for row in values:
                    items = cols[row >= 1]
                    if len(items) > 0:
                        f.write("\t".join(str(c) for c in items) + "\n")

            if algorithm == "fpgrowth":
                miner = fpg.FPGrowth(iFile=db_file, minSup=min_sup, sep="\t")
            elif algorithm == "topk":
                miner = topk_fae.FAE(iFile=db_file, k=k, sep="\t")
            elif algorithm == "maxfpgrowth":
                miner = maxfpg.MaxFPGrowth(iFile=db_file, minSup=min_sup, sep="\t")

            miner.mine()
            patterns_df = miner.getPatternsAsDataFrame()

            if patterns_df is not None and not patterns_df.empty:
                for _, row in patterns_df.iterrows():
                    pat_str = str(row["Patterns"])
                    terms_list = pat_str.strip().split("\t") if "\t" in pat_str else pat_str.strip().split()
                    patterns.append({
                        "pattern": terms_list,
                        "support": int(row["Support"])
                    })
        except UnicodeError:
            return {"error": "Encoding error while mining", "traceback": traceback.format_exc()}
        finally:
            if os.path.exists(db_file):
                os.remove(db_file)

        # Step 5: Format Output & Store Results
        patterns.sort(key=lambda x: x["support"], reverse=True)

        res_id = session.next_result_id("patterns")
        preview_10 = patterns[:10]

        summary = {
            "result_id": res_id,
            "category_name": category_name,
            "filtering_method": filtering_method,
            "algorithm": algorithm,
            "n_patterns_found": len(patterns),
            "preview": preview_10
        }

        session.artifacts[artifact_key] = patterns
        session.store_result("mine_patterns_tool", full_args, summary)

        return summary

    return [mine_patterns_tool]
