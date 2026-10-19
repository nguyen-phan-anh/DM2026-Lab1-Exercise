from typing import List, Optional, Union
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from langchain_core.tools import tool
from sklearn.feature_extraction.text import CountVectorizer


def make_tools(session):
    @tool
    def build_dtm_tool(
        max_features: Optional[int] = None,
        min_df: Union[int, float] = 1,
        max_df: Union[int, float] = 1.0,
        ngram_range: Optional[List[int]] = None,
    ) -> dict:
        """Builds a document-term matrix (DTM) from the loaded dataset's text column.

        Uses CountVectorizer to tokenize and count word frequencies across documents,
        populating session.feature_matrix and session.feature_names.

        Args:
            max_features: Maximum number of terms to include in the vocabulary.
            min_df: Minimum document frequency for a term to be included (count or fraction).
            max_df: Maximum document frequency for a term to be included (count or fraction).
            ngram_range: Range of n-gram sizes to extract as [min_n, max_n] (defaults to [1, 1]).

        Returns:
            dict: Summary containing result_id, n_documents, n_terms, non_zero,
                  total_elements, and sparsity_pct.
        """
        if session.dataframe is None or session.dataframe.empty:
            raise ValueError("No dataset loaded in session. Call load_dataset_tool first.")

        # Ensure text column exists
        if "text" not in session.dataframe.columns:
            raise ValueError("DataFrame does not contain a 'text' column.")

        # Handle default ngram_range cleanly to avoid mutable default list risk
        if ngram_range is None:
            ngram_range = [1, 1]

        # Fill missing text with empty strings to preserve row count and alignment
        text_data = session.dataframe["text"].fillna("").astype(str)

        # Convert list to tuple for CountVectorizer
        ngram_tuple = tuple(ngram_range)

        # Vectorize text
        vectorizer = CountVectorizer(
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            ngram_range=ngram_tuple,
        )
        feature_matrix = vectorizer.fit_transform(text_data)
        feature_names = vectorizer.get_feature_names_out().tolist()

        # Update session state
        session.feature_matrix = feature_matrix
        session.feature_names = feature_names
        session.artifacts["count_vectorizer"] = vectorizer

        if "category_name" in session.dataframe.columns:
            session.set_labels(session.dataframe["category_name"].values)

        # Sparsity calculation
        n_rows, n_cols = feature_matrix.shape
        non_zero = int(feature_matrix.nnz)
        total_elements = n_rows * n_cols
        sparsity_pct = (
            float(100.0 * (1.0 - non_zero / total_elements)) if total_elements > 0 else 0.0
        )

        args_dict = {
            "max_features": max_features,
            "min_df": min_df,
            "max_df": max_df,
            "ngram_range": ngram_range,
        }

        summary = {
            "result_id": session.next_result_id("build_dtm"),
            "n_documents": n_rows,
            "n_terms": n_cols,
            "non_zero": non_zero,
            "total_elements": total_elements,
            "sparsity_pct": round(sparsity_pct, 4),
        }

        session.store_result("build_dtm_tool", args_dict, summary)
        return summary

    @tool
    def term_frequency_tool(top_n: int = 10) -> dict:
        """Aggregates term frequencies across all documents in the global document-term matrix.

        Sums column counts in session.feature_matrix to compute overall frequency per term,
        returning top terms in the summary and storing a full report DataFrame.

        Args:
            top_n: Number of top terms to include in the returned summary list (default 10).

        Returns:
            dict: Summary containing result_id, total_unique_terms, total_occurrences, and top_terms.
        """
        if session.feature_matrix is None or session.feature_names is None:
            raise ValueError(
                "Document-term matrix not built yet. Call build_dtm_tool first."
            )

        # Compute column sums (term counts across all documents)
        counts = np.array(session.feature_matrix.sum(axis=0)).flatten()
        terms = session.feature_names

        # Create full report DataFrame sorted descending by frequency, then ascending by term for deterministic tie-breaking
        df = pd.DataFrame({"term": terms, "frequency": counts})
        df = df.sort_values(by=["frequency", "term"], ascending=[False, True]).reset_index(
            drop=True
        )

        total_unique_terms = len(df)
        total_occurrences = int(df["frequency"].sum())

        top_df = df.head(top_n)
        top_terms = [
            {"term": str(row["term"]), "frequency": int(row["frequency"])}
            for _, row in top_df.iterrows()
        ]

        args_dict = {"top_n": top_n}

        summary = {
            "result_id": session.next_result_id("term_frequency"),
            "total_unique_terms": total_unique_terms,
            "total_occurrences": total_occurrences,
            "top_terms": top_terms,
        }

        session.store_result("term_frequency_tool", args_dict, summary, full_report=df)
        return summary

    @tool
    def dtm_heatmap_tool(n_terms: int = 20, n_documents: int = 20) -> dict:
        """Renders a heatmap slice of the raw document-term matrix.

        Slices session.feature_matrix by first n_documents rows and first n_terms columns,
        builds a seaborn heatmap figure on session.pending_figure, and returns the slice matrix.

        Args:
            n_terms: Number of terms (columns) to include in the heatmap slice (default 20).
            n_documents: Number of documents (rows) to include in the heatmap slice (default 20).

        Returns:
            dict: Summary containing result_id, n_documents, n_terms, terms, document_labels, and matrix.
        """
        if session.feature_matrix is None or session.feature_names is None:
            raise ValueError(
                "Document-term matrix not built yet. Call build_dtm_tool first."
            )

        # Positional slicing bounded by actual matrix dimensions
        actual_n_docs = min(n_documents, session.feature_matrix.shape[0])
        actual_n_terms = min(n_terms, session.feature_matrix.shape[1])

        matrix_slice = session.feature_matrix[
            0:actual_n_docs, 0:actual_n_terms
        ].toarray()
        terms_slice = session.feature_names[:actual_n_terms]

        # Get string row indices from session.dataframe if available
        if session.dataframe is not None and len(session.dataframe) >= actual_n_docs:
            doc_labels = [str(idx) for idx in session.dataframe.index[:actual_n_docs]]
        else:
            doc_labels = [str(i) for i in range(actual_n_docs)]

        # Render heatmap figure
        fig, ax = plt.subplots(
            figsize=(max(6, actual_n_terms * 0.5), max(4, actual_n_docs * 0.4))
        )
        sns.heatmap(
            matrix_slice,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=terms_slice,
            yticklabels=doc_labels,
            ax=ax,
        )
        ax.set_title("Document-Term Matrix Heatmap")
        ax.set_xlabel("Terms")
        ax.set_ylabel("Documents")
        plt.tight_layout()

        # Pass figure object to session and close pyplot figure to prevent duplicate inline outputs
        session.pending_figure = fig
        plt.close(fig)

        args_dict = {"n_terms": n_terms, "n_documents": n_documents}

        summary = {
            "result_id": session.next_result_id("dtm_heatmap"),
            "n_documents": actual_n_docs,
            "n_terms": actual_n_terms,
            "terms": terms_slice,
            "document_labels": doc_labels,
            "matrix": matrix_slice.tolist(),
        }

        session.store_result("dtm_heatmap_tool", args_dict, summary)
        return summary

    return [build_dtm_tool, term_frequency_tool, dtm_heatmap_tool]
