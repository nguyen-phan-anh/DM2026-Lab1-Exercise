"""
Filter-style feature selection for bag-of-words / document-term matrices.

Three lenses on each term, none of which require training a classifier
model: variance_filter is an unsupervised sparsity check
(VarianceThreshold); pearson_filter is the linear correlation between a
term and a binary label; spearman_filter is the monotonic, or rank,
correlation between a term and a binary label.

Categorical targets with more than two classes are handled one-vs-rest.
Each class gets its own binary indicator, and per-feature correlation is
computed against each indicator separately, which is what makes Pearson
and Spearman meaningful for a nominal, non-ordinal label.

Performance note: on the real 20-newsgroups vocabulary, around 36,000
terms, a naive per-column scipy.stats.pearsonr loop takes about 40
seconds, too slow for an interactive chat turn. pearson_filter is
therefore computed with vectorized matrix algebra, a few matrix-vector
products over the whole vocabulary at once, instead of a Python loop.
Spearman needs per-column rank statistics that don't vectorize as
cleanly, so spearman_filter instead bounds its cost by narrowing to the
`max_terms` highest-variance columns (1500 by default) before computing
ranks. That narrowing also happens to match the intended pedagogical
order of running variance first, then correlation.
"""

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import spearmanr, t as t_dist
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import LabelBinarizer


def _as_dense_column(X, j):
    col = X[:, j]
    if sparse.issparse(col):
        col = col.toarray().ravel()
    else:
        col = np.asarray(col).ravel()
    return col


def _column_variances(X):
    # `variances_` holds every column's variance regardless of `threshold`
    # (threshold only affects get_support()/transform) -- 0.0 is just the
    # minimum valid value sklearn accepts here.
    selector = VarianceThreshold(threshold=0.0)
    selector.fit(X)
    return selector.variances_


def variance_filter(X, feature_names, threshold=0.0):
    """
    Score each feature by variance across all documents.

    Returns a DataFrame with columns [feature, variance, keep], sorted by
    variance descending. `keep` marks features whose variance is above
    `threshold` (VarianceThreshold semantics: strictly greater than).
    """
    selector = VarianceThreshold(threshold=threshold)
    selector.fit(X)
    result = pd.DataFrame({
        "feature": feature_names,
        "variance": selector.variances_,
        "keep": selector.get_support(),
    })
    return result.sort_values("variance", ascending=False).reset_index(drop=True)


def pearson_filter(X, y_binary, feature_names):
    """
    Pearson (linear) correlation of each feature with a binary target.

    Vectorized across the whole vocabulary at once (matrix-vector products,
    no per-column Python loop) so it stays fast even at tens of thousands
    of terms.

    Returns a DataFrame [feature, pearson_r, pearson_p], sorted by |r| descending.
    """
    y = np.asarray(y_binary, dtype=float)
    n = len(y)

    if sparse.issparse(X):
        X = X.tocsc()
        sum_x = np.asarray(X.sum(axis=0)).ravel()
        sum_x2 = np.asarray(X.multiply(X).sum(axis=0)).ravel()
        sum_xy = np.asarray(X.T.dot(y)).ravel()
    else:
        X = np.asarray(X, dtype=float)
        sum_x = X.sum(axis=0)
        sum_x2 = (X ** 2).sum(axis=0)
        sum_xy = X.T.dot(y)

    sum_y = y.sum()
    sum_y2 = (y ** 2).sum()

    numerator = n * sum_xy - sum_x * sum_y
    denom_x = n * sum_x2 - sum_x ** 2
    denom_y = n * sum_y2 - sum_y ** 2
    denom = np.sqrt(np.clip(denom_x, 0, None) * max(denom_y, 0))

    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(denom > 0, numerator / denom, 0.0)
        r = np.clip(r, -1.0, 1.0)

    df = max(n - 2, 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        t_stat = r * np.sqrt(df / np.clip(1 - r ** 2, 1e-12, None))
        p = 2 * t_dist.sf(np.abs(t_stat), df)
    p = np.where(denom > 0, p, 1.0)

    result = pd.DataFrame({"feature": feature_names, "pearson_r": r, "pearson_p": p})
    return result.reindex(result["pearson_r"].abs().sort_values(ascending=False).index).reset_index(drop=True)


def spearman_filter(X, y_binary, feature_names, max_terms=1500):
    """
    Spearman (monotonic/rank) correlation of each feature with a binary
    target.

    Rank statistics don't vectorize as cleanly as Pearson, so to stay fast
    on a large vocabulary this first narrows to the `max_terms`
    highest-variance columns (a cheap, vectorized pre-filter) before
    computing per-column rank correlation on that reduced set. Pass
    `max_terms=None` to disable narrowing and run on every column (slow on
    a full ~36k-term vocabulary).

    Returns a DataFrame [feature, spearman_r, spearman_p], sorted by |r|
    descending. Only includes the narrowed-to columns when narrowing
    applied.
    """
    n_features = X.shape[1]
    if max_terms is not None and n_features > max_terms:
        variances = _column_variances(X)
        top_idx = np.argsort(variances)[::-1][:max_terms]
        X = X[:, top_idx]
        feature_names = [feature_names[i] for i in top_idx]
        n_features = max_terms

    r_values = np.empty(n_features)
    p_values = np.empty(n_features)
    y_binary = np.asarray(y_binary)
    y_is_constant = np.std(y_binary) == 0

    for j in range(n_features):
        col = _as_dense_column(X, j)
        if y_is_constant or np.std(col) == 0:
            r_values[j], p_values[j] = 0.0, 1.0
        else:
            r, p = spearmanr(col, y_binary)
            r_values[j], p_values[j] = r, p

    result = pd.DataFrame({"feature": feature_names, "spearman_r": r_values, "spearman_p": p_values})
    return result.reindex(result["spearman_r"].abs().sort_values(ascending=False).index).reset_index(drop=True)


def correlation_filter_multiclass(X, y, feature_names, method="pearson", max_terms=1500):
    """
    One-vs-rest correlation of each feature against every class in `y`.

    `max_terms` is only used for method="spearman" (see spearman_filter);
    ignored for "pearson", which is vectorized and fast at full vocabulary
    size regardless.

    Returns a dict {class_label: DataFrame[feature, {method}_r, {method}_p]}
    (unsorted, in original feature order, so results across classes stay
    aligned -- except under Spearman narrowing, where the DataFrame is
    restricted to the narrowed column set for every class alike).
    """
    lb = LabelBinarizer()
    Y = lb.fit_transform(y)
    classes = lb.classes_
    if Y.shape[1] == 1:
        # binary label: LabelBinarizer collapses to a single column
        Y = np.hstack([1 - Y, Y])

    per_class = {}
    for i, cls in enumerate(classes):
        if method == "pearson":
            df = pearson_filter(X, Y[:, i], feature_names)
            df = df.set_index("feature").reindex(feature_names).reset_index()
        else:
            df = spearman_filter(X, Y[:, i], feature_names, max_terms=max_terms)
        per_class[cls] = df
    return per_class


def combined_filter_report(X, y, feature_names, variance_threshold=0.0, max_terms=1500):
    """
    Run variance + one-vs-rest Pearson + one-vs-rest Spearman and merge into
    a single report: one row per feature, with per-class correlation columns
    and a `max_abs_pearson` / `max_abs_spearman` summary column for ranking.

    Note: Spearman columns are narrowed to the `max_terms` highest-variance
    features (see spearman_filter); rows outside that set will have NaN in
    the spearman columns.
    """
    variance_df = variance_filter(X, feature_names, threshold=variance_threshold)
    variance_df = variance_df.set_index("feature")

    pearson_per_class = correlation_filter_multiclass(X, y, feature_names, method="pearson")
    spearman_per_class = correlation_filter_multiclass(X, y, feature_names, method="spearman", max_terms=max_terms)

    report = pd.DataFrame({"feature": feature_names}).set_index("feature")
    report["variance"] = variance_df["variance"]
    report["keep_by_variance"] = variance_df["keep"]

    pearson_r_cols = []
    for cls, df in pearson_per_class.items():
        col = f"pearson_r__{cls}"
        report[col] = df.set_index("feature")["pearson_r"].reindex(report.index).values
        pearson_r_cols.append(col)

    spearman_r_cols = []
    for cls, df in spearman_per_class.items():
        col = f"spearman_r__{cls}"
        report[col] = df.set_index("feature")["spearman_r"].reindex(report.index).values
        spearman_r_cols.append(col)

    report["max_abs_pearson"] = report[pearson_r_cols].abs().max(axis=1)
    report["max_abs_spearman"] = report[spearman_r_cols].abs().max(axis=1)

    return report.reset_index().sort_values("max_abs_pearson", ascending=False).reset_index(drop=True)


def select_top_k(report_df, score_column="max_abs_pearson", k=50):
    """Convenience helper: top-k feature names by a given score column."""
    return report_df.sort_values(score_column, ascending=False).head(k)["feature"].tolist()
