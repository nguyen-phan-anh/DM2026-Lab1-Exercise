# Test fixture: known-answer data for testing your own tools

If you ask the agent to "write a test," and it invents its own
example data on the spot, the test only checks that the code agrees
with itself. It can't catch a real bug, since whatever the
implementation happens to compute becomes the expected answer by
definition. A test like that will always pass, including on a broken
implementation, which makes it worthless as a check.

The fix is a fixture with answers that were computed independently of
whatever code you write: `agent_dev/sample_fixture.csv`, eight
rows small enough to reason about by hand, with the exact results this
page lists below computed directly with `pandas`, `scikit-learn`, and
`scipy`, not derived from any tool in this workspace. When you ask for a
test, ask for one that loads this file and asserts against the specific
numbers here, instead of letting the agent make up its own data and its
own expected values in the same breath.

## The data

```
text                              label
always alpha alpha alpha beta     catA
always alpha alpha beta           catA
always alpha beta                 catA
always gamma gamma gamma delta    catB
always gamma gamma delta          catB
always gamma delta                catB
always gamma delta                catB   <- exact duplicate of the row above
                                   catA   <- empty text
```

Load it with `load_dataset_tool(file_path="agent_dev/sample_fixture.csv", text_column="text", label_column="label")`.

## Known answers

Numbered so you can point the agent at one directly ("build a test
for known answer 6", say) instead of describing it from scratch each
time.

### 1. `load_dataset_tool`

Loading the fixture itself: exactly 8 documents, split `{"catA": 4,
"catB": 4}` in `counts_per_category`.

### 2. `check_missing_tool`

Exactly 1 row has missing/empty text (the last row).

### 3. `check_duplicates_tool`

Exactly 1 duplicate (row 6 is an exact copy of row 5, both
`"always gamma delta"`).

Dropping with `keep=False` (matching Master's own
`X.drop_duplicates(keep=False, inplace=True)`, which removes every copy
of a duplicated row, not just the extras) leaves exactly 6 rows: rows
5 and 6 are both gone, not just one of them. If a drop call instead
behaves like pandas' own default (`keep="first"`, which would leave
row 5 and only remove row 6, ending at 7 rows), that's the wrong
behavior for this tool, not a different valid choice.

### 4. `build_dtm_tool`

Vocabulary (`CountVectorizer` defaults, all 8 rows including the empty
one): `alpha`, `always`, `beta`, `delta`, `gamma`. Five terms.

Sparsity, on this same 8x5 matrix: 21 non-zero cells out of 40 total
(`8 rows * 5 columns`), so `sparsity_pct = 100 * (1 - 21/40) = 47.5`.

### 5. `variance_filter_tool`

Per-term counts and variance, one number per document in the order
above:

| Term | Counts per document | Variance |
|---|---|---|
| `alpha` | 3, 2, 1, 0, 0, 0, 0, 0 | 1.1875 |
| `always` | 1, 1, 1, 1, 1, 1, 1, 0 | 0.109375 |
| `beta` | 1, 1, 1, 0, 0, 0, 0, 0 | 0.234375 |
| `delta` | 0, 0, 0, 1, 1, 1, 1, 0 | 0.25 |
| `gamma` | 0, 0, 0, 3, 2, 1, 1, 0 | 1.109375 |

Nothing here has zero variance (even `always`, which appears in every
row except the one with no text at all, still isn't perfectly constant),
and that's deliberate. A variance filter with `threshold=0.15` should
keep `alpha`, `beta`, `delta`, and `gamma`, and remove only `always`.
That's a real, checkable claim about a real threshold, not "the filter
removes whatever it removes."

### 6. `pearson_filter_tool` / `spearman_filter_tool`

Correlation with the `catB` label, `target_class="catB"`, one-vs-rest:

| Term | Pearson r | Spearman r |
|---|---|---|
| `alpha` | -0.6882 | -0.7500 |
| `always` | 0.3780 | 0.3780 |
| `beta` | -0.7746 | -0.7746 |
| `delta` | 1.0000 | 1.0000 |
| `gamma` | 0.8307 | 0.9363 |

`delta` appears in every `catB` document and no `catA` document, so a
correct correlation filter must report it at essentially `r = 1.0`. If
a proposed implementation reports something far from that for `delta`,
the implementation is wrong, not the test.

### 7. `reduce_dimensions_tool`, method `"pca"`

Same document-term matrix, `n_components=2`. Needs no seed at all:
it's exactly deterministic on this input (confirmed across repeated
runs), since a matrix this small always uses the exact `full` solver
rather than a randomized one:

| Doc | PC1 | PC2 |
|---|---|---|
| 0 | 2.3669 | 0.8871 |
| 1 | 1.7039 | 0.2642 |
| 2 | 1.0408 | -0.3587 |
| 3 | -2.0759 | 1.0129 |
| 4 | -1.4554 | 0.3405 |
| 5 | -0.8349 | -0.3319 |
| 6 | -0.8349 | -0.3319 |
| 7 | 0.0894 | -1.4822 |

`explained_variance_ratio`: `[0.7532, 0.1965]`.

### 8. `reduce_dimensions_tool`, method `"tsne"`

Only reproducible with `random_state` fixed, and can't use its normal
default on a dataset this small: `tsne`'s default `perplexity=30`
requires more documents than exist here (must be less than
`n_samples`, 8), and that's exactly why `reduce_dimensions_tool` takes
`perplexity` as a parameter, so a test can override it. Call with
`method="tsne", perplexity=2, random_state=42`:

| Doc | Dim 1 | Dim 2 |
|---|---|---|
| 0 | 1330.9991 | 32.1039 |
| 1 | 951.2459 | 393.5588 |
| 2 | 530.8209 | 682.9791 |
| 3 | -1558.6448 | -314.1625 |
| 4 | -915.5153 | -346.3907 |
| 5 | -280.0313 | -1261.2115 |
| 6 | -280.0313 | -1261.2115 |
| 7 | -701.3674 | 1465.3973 |

### 9. `reduce_dimensions_tool`, method `"umap"`

Same idea as #8: `umap`'s default `n_neighbors=15` has the same
too-few-documents problem, which is why `n_neighbors` is a parameter
too. Call with `method="umap", n_neighbors=3, random_state=42`:

| Doc | Dim 1 | Dim 2 |
|---|---|---|
| 0 | 6.6811 | -6.8218 |
| 1 | 6.9907 | -7.2245 |
| 2 | 7.7022 | -7.3712 |
| 3 | 10.2228 | -6.2310 |
| 4 | 9.8759 | -5.6537 |
| 5 | 9.3831 | -6.5748 |
| 6 | 9.2153 | -5.8297 |
| 7 | 8.5860 | -7.2622 |

The `perplexity`/`n_neighbors` values in #8 and #9 only make sense for
testing on this tiny fixture; don't treat them as reasonable defaults
for a real dataset. A test against #7, #8, or #9 should assert with a
numerical tolerance (`np.allclose` or `pytest.approx`, not exact
equality), since floating-point results can differ in their last couple
of decimal places depending on the machine's BLAS backend even with
every input and seed identical.

### 10. `tokenize_tool`

Assumes the standard approach, `nltk.word_tokenize` on lowercased
text, consistent with `CountVectorizer`'s own default lowercasing
elsewhere in this pipeline. If your implementation doesn't lowercase,
adjust the expected tokens accordingly; the tokenizing library itself
is still `nltk.word_tokenize`. One token list per document, in the
8-row order above:

| Doc | Tokens |
|---|---|
| 0 | `always`, `alpha`, `alpha`, `alpha`, `beta` |
| 1 | `always`, `alpha`, `alpha`, `beta` |
| 2 | `always`, `alpha`, `beta` |
| 3 | `always`, `gamma`, `gamma`, `gamma`, `delta` |
| 4 | `always`, `gamma`, `gamma`, `delta` |
| 5 | `always`, `gamma`, `delta` |
| 6 | `always`, `gamma`, `delta` |
| 7 | (empty list, empty text) |

### 11. `list_files_tool`

Not about the CSV fixture at all; this tests against real repo state,
so use a directory unlikely to change: `list_files_tool(subdirectory=
"newdataset")` should return `directories: []`, `files:
["Reddit-stock-sentiment.csv"]`. If this repo's contents change later,
this is the one entry that would need updating to match, not because
the fixture changed but because the thing being listed did.

### 12. `inspect_data_tool`

No known answer here, on purpose. This is one of two tools
(`binarize_labels_tool` is the other, #15 below) deliberately left
without a defined contract or a fixture answer, for you to design and
test yourself using what you've learned from the other sixteen. See
`TOOLS_PIPELINE_BREAKDOWN.md`'s "Two tools with no contract or known
answer, on purpose" section for why.

### 13. `sample_data_tool`

Call with `n=4, random_state=42`: returns exactly rows `[1, 5, 0, 7]`
(pandas' `DataFrame.sample`, same seed convention as the rest of this
project). A different `random_state` or a different `n` will
legitimately give different rows; this exact answer only holds for
this exact call.

### 14. `term_frequency_tool`

Column sums of the same document-term matrix used in #4-#9. This is
just the row totals of #5's per-document counts, so it should already
agree with those numbers, not be computed separately:

| Term | Total frequency |
|---|---|
| `alpha` | 6 |
| `always` | 7 |
| `beta` | 3 |
| `delta` | 4 |
| `gamma` | 7 |

### 15. `binarize_labels_tool`

No known answer here either, same reason as #12 above: deliberately
left for you to design and test yourself.

### 16. `cosine_similarity_tool`

On the same document-term matrix, using raw counts (not normalized
frequencies):

| Doc pair | Cosine similarity |
|---|---|
| 0 vs 1 (both catA, near-identical text) | 0.9847 |
| 0 vs 3 (catA vs catB, share only "always") | 0.0909 |
| 3 vs 4 (both catB, near-identical text) | 0.9847 |
| 0 vs 7 (catA vs the empty-text row) | 0.0 |

The 0.9847 pairs and the 0.0909 pair are the useful checks: near-duplicate
documents in the same category should score close to 1, and documents
sharing only the near-universal `always` term should score low, not
close to 1. The empty-text row against anything should be exactly 0,
not `NaN` or an error: a real edge case this fixture was built to
cover.

### 17. `mine_patterns_tool`

Per-category document-term matrix uses its own fresh
`CountVectorizer(stop_words="english")`, fit only on that category's
text: a different, smaller vocabulary than #4's global DTM, and it
drops `"always"` as an English stopword even though every other known
answer here treats `"always"` as an ordinary term. catA's local
vocabulary is `["alpha", "beta"]`; catB's is `["delta", "gamma"]`.

`filtering_method="variance"` and `filtering_method="term_frequency"`
both keep **zero terms for both categories** on this fixture. Not a
bug, a real edge case: only 2 terms survive per category, and a
5th-95th percentile band computed across just 2 values excludes both of
them. `mine_patterns_tool` must return a clean "no terms survived
filtering" result here, not raise.

`filtering_method="tfidf"` is the one that leaves anything to mine:
catA keeps only `alpha` (mean TF-IDF 0.6376 vs `beta`'s 0.3676, cut at
the 20th percentile); catB keeps only `gamma` (0.8143 vs `delta`'s
0.5444). Mining each category's single surviving column (catA's
empty-text row drops out of the transactional database entirely, since
it has no items):

| Category | Kept term | Pattern | Support | `min_sup` range that still finds it |
|---|---|---|---|---|
| catA | `alpha` | `{alpha}` | 3 | 1-3 (0 patterns at `min_sup=4`) |
| catB | `gamma` | `{gamma}` | 4 | 1-4 |

### 18. `describe_data_tool`

`text_length` (character count) per document, in the 8-row order above:
`[29, 23, 17, 30, 24, 18, 18, 0]`.

Overall `.describe()`: count=8, mean=19.875, std=9.4330, min=0, 25%=17.75,
50%=20.5, 75%=25.25, max=30.

Per category: catA (rows 0,1,2,7), count=4, mean=17.25, std=12.5,
min=0, 25%=12.75, 50%=20.0, 75%=24.5, max=29. catB (rows 3,4,5,6),
count=4, mean=22.5, std=5.7446, min=18, 25%=18.0, 50%=21.0, 75%=25.5,
max=30.

### 19. `feature_correlation_matrix_tool`

All 5 terms survive the top-20 cutoff (there are only 5 to begin with).
By variance, descending: `alpha`, `gamma`, `delta`, `beta`, `always`.
Full correlation matrix, rounded to 4 decimals:

| | alpha | gamma | delta | beta | always |
|---|---|---|---|---|---|
| **alpha** | 1.0000 | -0.5718 | -0.6882 | 0.8885 | 0.2601 |
| **gamma** | -0.5718 | 1.0000 | 0.8307 | -0.6435 | 0.3140 |
| **delta** | -0.6882 | 0.8307 | 1.0000 | -0.7746 | 0.3780 |
| **beta** | 0.8885 | -0.6435 | -0.7746 | 1.0000 | 0.2928 |
| **always** | 0.2601 | 0.3140 | 0.3780 | 0.2928 | 1.0000 |

The diagonal is always exactly 1.0 (a term against itself); `alpha`/`beta`
(0.8885) and `gamma`/`delta` (0.8307) are the useful checks: both pairs
are strongly positively correlated, which makes sense, since `alpha`/`beta`
only ever appear in catA documents and `gamma`/`delta` only in catB ones.

### 20. `mine_patterns_tool`, `algorithm="topk"` / `algorithm="maxfpgrowth"`

Same `filtering_method="tfidf"`-filtered, per-category transactional
data as #17 (catA -> `alpha` only, catB -> `gamma` only).

`algorithm="topk"` (`PAMI.frequentPattern.topk.FAE`): at `k=1`, `k=2`,
and `k=3` alike, both categories return exactly one pattern,
catA -> `{alpha}` support 3, catB -> `{gamma}` support 4, since
there's only ever one possible pattern to return regardless of `k`.

`algorithm="maxfpgrowth"` (`PAMI.frequentPattern.maximal.MaxFPGrowth`):
**returns zero patterns for both categories**, at both `min_sup=1` and
`min_sup=4` (expected, not a bug in your implementation). This is
real PAMI behavior on a single-surviving-term transactional database,
even though plain `FPGrowth` (#17) finds a pattern on the exact same
data. Assert on the empty result, don't treat it as something to debug.

### 21. `dtm_heatmap_tool`

Same global DTM as #4/#5: no scoring, no filtering, just a raw
positional slice. `n_terms=20, n_documents=20` (the defaults) both
exceed what the fixture has, so this returns the *entire* matrix: all
5 terms in vocabulary order (`alpha`, `always`, `beta`, `delta`,
`gamma`), all 8 documents in the fixture's row order. This is the same
data as #5's per-term counts, just transposed to one row per document
instead of one row per term:

| Doc | alpha | always | beta | delta | gamma |
|---|---|---|---|---|---|
| 0 | 3 | 1 | 1 | 0 | 0 |
| 1 | 2 | 1 | 1 | 0 | 0 |
| 2 | 1 | 1 | 1 | 0 | 0 |
| 3 | 0 | 1 | 0 | 1 | 3 |
| 4 | 0 | 1 | 0 | 1 | 2 |
| 5 | 0 | 1 | 0 | 1 | 1 |
| 6 | 0 | 1 | 0 | 1 | 1 |
| 7 | 0 | 0 | 0 | 0 | 0 |

A call with `n_terms=2, n_documents=3` should return just the top-left
3x2 corner of that same table (rows 0-2, columns `alpha`/`always`):
`[[3, 1], [2, 1], [1, 1]]`, useful for checking the slicing itself,
separately from checking the underlying counts are right.

## How to actually use this

Ask the agent to write a test using this file and these numbers
directly, for example: `Build a test for variance_filter_tool using
agent_dev/sample_fixture.csv. It should assert that "delta"
has variance 0.25 and that a threshold of 0.15 keeps alpha, beta,
delta, and gamma but removes always.` A test written that way fails
if your implementation is wrong, which is the entire point of writing
one. A test the agent writes by inventing its own numbers can't do
that, because there's nothing independent for it to be wrong against.

Or point at a known answer by number instead of restating it:
`Build a test for known answer 6 in TEST_FIXTURE.md.` The agent is
instructed to fetch this exact page live with `read_spec_file_tool`
before proposing a test, so naming a number means you're both reading
the same current text, not a copy that could go stale.

Every test file goes in `tests/`, named after the tool file it tests:
a test for `tools_dtm.py` is `tests/test_tools_dtm.py`, a test for the
premade `load_dataset_tool` is `tests/test_load_dataset_tool.py`. One
test file per tool file, not one per tool, since several tools often
live in the same file (`tools_filtering.py` has four). Don't let the
agent scatter test files flatly into the workspace root or name them
some other way; ask for them at this path if it proposes something
else.
