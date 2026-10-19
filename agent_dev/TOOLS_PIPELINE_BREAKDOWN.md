# Tools pipeline breakdown: what your tool files need to match

Once you've built tools in this workspace, a separate notebook
(`DM2026-Lab1-AgenticPipeline.ipynb`) loads them and runs a real
analysis agent on top of your own code, instead of a hidden reference
implementation. For that to work, each file needs to follow the same
shape the loader expects. This page describes that shape: the object
your tools receive, and the function signature the loader looks for. It
does not show you how any tool is implemented; that part is yours to
figure out with the agent's help.

## What you actually need to build

The full pipeline this course covers is twenty tools worth of work.
Eighteen of them are yours to build:

| File | Tool | What it does |
|---|---|---|
| `tools_data.py` | `tokenize_tool` | Tokenizes each document's text into unigrams. |
| `tools_data.py` | `list_files_tool` | Lists files/folders in the repo, so a dataset path can be discovered before loading it. |
| `tools_data.py` | `inspect_data_tool` | Peeks at a few rows of the working DataFrame. **No contract or known answer; see below.** |
| `tools_data.py` | `check_missing_tool` | Checks for missing/empty text values. |
| `tools_data.py` | `check_duplicates_tool` | Checks for (and optionally drops) duplicate documents. When dropping, use `keep=False` (drop every copy of a duplicated row, not just the extras), matching Master's own `X.drop_duplicates(keep=False, inplace=True)` exactly. |
| `tools_data.py` | `sample_data_tool` | Subsamples the working DataFrame. |
| `tools_data.py` | `describe_data_tool` | `.describe()` stats on document text length, plus a box plot by category; see its own contract section below. |
| `tools_dtm.py` | `build_dtm_tool` | Builds the document-term matrix into `session.feature_matrix` / `feature_names`, and reports its sparsity; see its own contract section below. |
| `tools_dtm.py` | `term_frequency_tool` | Aggregates term frequencies across all documents. |
| `tools_dtm.py` | `dtm_heatmap_tool` | Heatmap of a slice of the raw document-term matrix; see its own contract section below. |
| `tools_filtering.py` | `variance_filter_tool` | Flags near-constant terms by variance for removal. |
| `tools_filtering.py` | `pearson_filter_tool` | Pearson correlation between a term's count and a target class. |
| `tools_filtering.py` | `spearman_filter_tool` | Spearman correlation, same idea, monotonic rather than linear. |
| `tools_patterns.py` | `mine_patterns_tool` | Frequent-pattern mining (FPGrowth, Top-K, or MaxFPGrowth), one category per call; see its own contract section below, not just this row. |
| `tools_reduction.py` | `reduce_dimensions_tool` | PCA/t-SNE/UMAP reduction. **Must plot its output every time it runs**: a scatter plot of the reduced coordinates colored by category, set on `session.pending_figure`, not just return coordinates; see the plotting requirement below. Accepts `random_state` (for reproducible output), `perplexity` (t-SNE) and `n_neighbors` (UMAP), so it can be tested deterministically. |
| `tools_reduction.py` | `binarize_labels_tool` | One-hot encodes the category labels. **No contract or known answer; see below.** |
| `tools_exploration.py` | `cosine_similarity_tool` | Cosine similarity between two documents. |
| `tools_exploration.py` | `feature_correlation_matrix_tool` | Feature-vs-feature correlation heatmap over the top-20 terms by variance; see its own contract section below. |

Two tools are the exception: `load_dataset_tool` (in `tools_data.py`)
and `visualize_result_tool`, both already built for you and living in
`premade_tools/` at the repo root. `load_dataset_tool` is the one tool
every other tool's input depends on (it's what populates
`session.dataframe` in the first place), so it doubles as a single
worked example of what a finished tool actually looks like before you
build the other eighteen yourself. `visualize_result_tool` renders a bar
chart of any previously stored result; it's provided rather than
student-built because it doesn't compute anything new, only visualizes
a result some other tool already produced. It does **not** cover a
heatmap or a scatter plot: `feature_correlation_matrix_tool`,
`dtm_heatmap_tool`, and `reduce_dimensions_tool` each build their own
plot directly and set `session.pending_figure` themselves, never
routing through `visualize_result_tool` at all. The agentic-pipeline
notebook loads your workspace and `premade_tools/` together into one
agent, so both provided tools are available automatically once you get
there. Read either for reference any time; they're the two files in
this whole exercise that aren't meant to be hidden from you.

## Two tools with no contract or known answer, on purpose

Sixteen of your eighteen tools have a defined shape somewhere on this
page (in the table above, or their own contract section below) and a
known answer in `TEST_FIXTURE.md` to test against. Two don't:
`inspect_data_tool` and `binarize_labels_tool`. That's deliberate, not
an oversight. Both are simple enough, and self-contained enough
(nothing else in this pipeline depends on their exact output shape),
that you're meant to design and test them yourself, using what you've
already learned building the other sixteen, rather than being handed
another spec to match. Ask the agent to build one the same way as any
other tool (propose, review, approve, write), but you'll need to
describe what it should return in your own words, and work out what a
reasonable test for it looks like together with the agent, since there
is no known answer for either one in `TEST_FIXTURE.md` to point it at.

One naming detail worth getting right, since another one of your own
tools reads it: `build_dtm_tool` must store the fitted vectorizer at
`session.artifacts["count_vectorizer"]`, since `cosine_similarity_tool`
needs it later to vectorize documents the same way the DTM did.

`build_dtm_tool` must also report the matrix's sparsity in its returned
summary: `non_zero` (the matrix's `.nnz`), `total_elements` (`n_rows *
n_cols`), and `sparsity_pct` (`100 * (1 - non_zero / total_elements)`),
matching Master's own sparsity calculation exactly. No plot is involved
here: Master's own version of this is three printed numbers, nothing
more, so neither is this.

Another one worth getting right, since the provided `visualize_result_tool`
depends on it: `variance_filter_tool`, `pearson_filter_tool`, and
`spearman_filter_tool` must each pass `full_report` to
`session.store_result(...)` as a DataFrame with the term/feature name as
its **first column** and the number to plot (variance, `pearson_r`,
`spearman_r`) as its **second column**, with rows already sorted in the
order they should be charted. `visualize_result_tool` just bar-charts
the head of whatever it's given, in whatever order it finds; it does
no sorting or column-guessing of its own.

## The plotting requirement for `reduce_dimensions_tool`

Unlike every other tool in this workspace, `reduce_dimensions_tool` is
required to produce a real plot, not just return numbers. Every time it
runs, regardless of `method`, it must build a scatter plot of the
reduced 2D coordinates, colored by `category` (or `category_name`), and
set it on `session.pending_figure` before returning, the same
mechanism any other plotting tool would use, described below under
`session.pending_figure`. Setting that attribute is the whole job:
saving the image to `plots/` and displaying it in the chat are handled
automatically by the notebook, not something your tool needs to do
itself. A version that only returns the coordinates
and `explained_variance_ratio` without ever touching
`session.pending_figure` does not satisfy the assignment, even though
`TEST_FIXTURE.md`'s known answers #7-9 (which only check the returned
coordinates) would not catch that omission; ask the agent for a
manual check of the figure itself if you want to confirm this before
moving on, since there's no automated test for it.

## The `mine_patterns_tool` contract

In plain terms: this tool takes one category's documents, narrows down
which words actually matter using one of three methods, then finds
groups of words that tend to show up together often (a "pattern").
It's the most involved tool in this course, so read through the steps
below carefully rather than guessing at the shape from the name alone.

Unlike the rest of the pipeline, `mine_patterns_tool` doesn't read from
`session.feature_matrix`: that's one global document-term matrix, and
this tool needs a *per-category* one, built fresh from
`session.dataframe` each call. One call mines one category, the same
"one explicit action per call" shape every other tool in this workspace
already follows; call it once per category to reproduce what the
Master notebook shows for all of them at once.

Arguments:

- `category_name` (str): which category to mine. Rows outside this
  category are excluded before anything else happens.
- `filtering_method` (`"variance"`, `"tfidf"`, or `"term_frequency"`):
  which of the three vocabulary filters to apply before mining, mirroring
  the Master notebook's "Three Filtering Approaches for Pattern Mining"
  section, not the global-DTM `variance_filter_tool` you already built.
- `algorithm` (`"fpgrowth"`, `"topk"`, or `"maxfpgrowth"`, default
  `"fpgrowth"`): which PAMI mining algorithm to run against the same
  transactional database built in steps 1-4 below. `"fpgrowth"` and
  `"maxfpgrowth"` both take `min_sup`; `"topk"` takes `k` instead (see
  below); they're different selection criteria, not interchangeable.
- `min_sup` (int, required for `"fpgrowth"`/`"maxfpgrowth"`): passed
  straight through as `minSup`. There is no auto-selection here (Master
  picks its own `minSup` per filtering method by eyeballing
  item-frequency/transaction-length histograms first; that judgment
  call stays in the Master notebook's own exploration; this tool just
  takes whatever value the caller supplies).
- `k` (int, required for `"topk"`): passed straight through to
  `PAMI.frequentPattern.topk.FAE` as `k`, the number of
  highest-support patterns to keep, a different selection rule than a
  support threshold.

What it needs to do, step by step, matching Master's own pipeline
exactly so the result is reproducible against it:

1. Filter `session.dataframe` to `category_name`, and build a **fresh,
   category-local** document-term matrix: `CountVectorizer(stop_words=
   "english")`, fit only on that category's text. This is a different
   vocabulary than `build_dtm_tool`'s global one, and it does drop
   stopwords, which `build_dtm_tool` doesn't. On
   `sample_fixture.csv`, `"always"` (present in every non-empty row of
   the fixture) is an English stopword and disappears from the
   per-category vocabulary entirely, leaving catA with just `["alpha",
   "beta"]` and catB with just `["delta", "gamma"]`.
2. Apply the chosen `filtering_method` to that matrix's columns:
   `variance`/`term_frequency` keep the middle 90% (5th-95th percentile)
   by variance or column-sum respectively; `tfidf` keeps the top 80% (at
   or above the 20th percentile) by mean TF-IDF score
   (`TfidfTransformer`). Same percentile bands as
   `helpers/text_filtering_helpers.py`'s `apply_variance_filtering` /
   `apply_tfidf_filtering` / `apply_term_frequency_filtering`.
3. **A filtering method can legitimately keep zero terms**: this isn't
   an error to guard against, it's a real outcome your tool must handle
   cleanly. On this fixture, both `variance` and
   `term_frequency` keep nothing at all for either category (only 2
   terms survive stopword removal per category, and a 5th-95th
   percentile band across just 2 values excludes both of them). When
   that happens, return a summary that says so plainly (e.g. `{"patterns":
   [], "note": "no terms survived filtering"}`), not an exception.
4. Convert whatever terms survived into a transactional database (one
   transaction per document, items are the terms present with count >=
   1) via `PAMI.extras.convert.DF2DB.DF2DB(...).convert2TransactionalDatabase(...)`,
   then mine it with whichever algorithm was requested:
   `PAMI.frequentPattern.basic.FPGrowth` (`minSup=min_sup`),
   `PAMI.frequentPattern.topk.FAE` (`k=k`), or
   `PAMI.frequentPattern.maximal.MaxFPGrowth` (`minSup=min_sup`). All
   three read the exact same transactional-database file from step 4;
   only the mining call itself changes.
5. **`"maxfpgrowth"` can legitimately return zero patterns even when
   `"fpgrowth"` finds one on the same data**: on this
   fixture's single-surviving-term-per-category case, `MaxFPGrowth`
   returns nothing at either `min_sup=1` or `min_sup=4`, while plain
   `FPGrowth` finds a pattern there. Don't treat an empty `MaxFPGrowth`
   result as your tool being broken; return it plainly, same as the
   zero-terms-survived case above.

Outputs: a summary with the mined patterns (term(s) and support), via
`session.store_result`, and the pattern list stored in
`session.artifacts` (e.g. keyed
`f"patterns_{category_name}_{filtering_method}_{algorithm}"`) so a
later tool call or discussion turn can reference it.

Known answer for this exact pipeline, on `sample_fixture.csv`, is
`TEST_FIXTURE.md` #17 (`"fpgrowth"`) and #20 (`"topk"`/`"maxfpgrowth"`).

Naming collision worth being deliberate about: `filtering_method=
"term_frequency"` here and `term_frequency_tool` (#14) are not the same
thing and don't share code. `term_frequency_tool` sums a term's count
across the *entire, already-built* global DTM (`session.feature_matrix`)
and returns those totals; it doesn't filter anything, it just reports.
`filtering_method="term_frequency"` inside `mine_patterns_tool` builds
its *own* fresh per-category matrix from scratch and uses column sums
only as the metric to filter by, discarding terms outside the middle
90%. Reusing `term_frequency_tool`'s output here would be wrong even if
the numbers happened to look similar, since it's scoped to the whole
dataset, not one category.

## The `describe_data_tool` contract

Two parts, both on `session.dataframe`, matching Master's data-quality
section exactly:

1. Compute `text_length` as `df["text"].apply(len)` (character count,
   not word count; confirmed against Master's own `X['text_length'] =
   X['text'].apply(len)`), then return `.describe()`'s standard stats
   (`count`, `mean`, `std`, `min`, `25%`, `50%`, `75%`, `max`) over
   `text_length`, both overall and broken out per `category_name`.
2. Build a box plot of `text_length` grouped by `category_name`
   (`seaborn.boxplot`, matching Master's own call) and set it on
   `session.pending_figure` before returning, same plotting mechanism
   as `reduce_dimensions_tool`, described above.

Known answer: `TEST_FIXTURE.md` #18.

## The `feature_correlation_matrix_tool` contract

Operates on the **global** DTM (`session.feature_matrix` /
`session.feature_names`), not a per-category one. Unlike
`mine_patterns_tool`, this mirrors Master's feature-correlation section
exactly, which runs on the whole dataset at once.

1. Compute each term's variance across all documents, and take the
   top 20 by variance (fewer than 20 terms in the vocabulary, as
   happens on `sample_fixture.csv`, which only has 5, just means
   "all of them," not an error).
2. Compute the Pearson correlation matrix (`np.corrcoef`) across just
   those top-variance terms' columns.
3. Build a heatmap of that correlation matrix (`seaborn.heatmap`,
   annotated with the correlation values, matching Master's own call)
   and set it on `session.pending_figure` before returning.

Return the correlation matrix and the corresponding term names in the
summary (small enough to be JSON-serializable at 20 terms), not just
the plot; the plot alone isn't testable against a known answer.

Known answer: `TEST_FIXTURE.md` #19.

## The `dtm_heatmap_tool` contract

Mirrors Master's own DTM heatmap exactly: no scoring, no filtering,
just a raw positional slice of the already-built global DTM
(`session.feature_matrix` / `session.feature_names`), rendered as a
heatmap. Unlike `feature_correlation_matrix_tool`, nothing here is
computed; it's the same matrix `build_dtm_tool` already produced,
sliced and colored in.

Arguments:

- `n_terms` (int, default 20): how many terms (columns) to include,
  taken in `session.feature_names`'s existing order, no sorting by
  variance or anything else, matching Master's own `[0:20]` slice.
- `n_documents` (int, default 20): how many documents (rows) to
  include, taken in `session.dataframe`'s existing row order.

If the DTM has fewer terms or documents than requested (as on
`sample_fixture.csv`, which has only 5 terms and 8 documents), that
just means "all of them," the same "fewer than requested isn't an
error" handling as `feature_correlation_matrix_tool`'s top-20 cutoff.

1. Slice `session.feature_matrix[0:n_documents, 0:n_terms]` and
   convert to a dense array (`.toarray()`).
2. Build a heatmap of that slice (`seaborn.heatmap`, annotated with the
   raw counts) and set it on `session.pending_figure` before returning.

Return the sliced matrix (as a small, JSON-serializable nested list,
one row per document) and the term/document labels included, in the
summary, not just the plot, for the same reason
`feature_correlation_matrix_tool` does.

Known answer: `TEST_FIXTURE.md` #21.

## Testing what you build

`agent_dev/TEST_FIXTURE.md` is the one reference for testing:
what fixture data exists (`agent_dev/sample_fixture.csv`),
why it matters that its expected answers were computed independently
rather than invented on the spot, and the exact numbers to test
against. Read it before asking for a test.

Test files go in `tests/`, named after the tool file they test:
`tools_dtm.py` gets `tests/test_tools_dtm.py`, not a flat file in the
workspace root.

## The `session` object

Every tool your file defines will receive one shared `session` object
(an instance of `agent_pipeline.session_state.SessionState`) as a
closure variable. Its attributes are:

| Attribute | Type | Meaning |
|---|---|---|
| `session.dataframe` | pandas DataFrame or `None` | The working dataset. Has at least `text`, `category`, `category_name` columns once loaded; may also have `unigrams` if tokenized. |
| `session.feature_matrix` | scipy sparse matrix or `None` | The document-term matrix, once built. |
| `session.feature_names` | list of str or `None` | Vocabulary, aligned with `feature_matrix`'s columns. |
| `session.labels` | numpy array or `None` | `category_name` per document, aligned with `feature_matrix`'s rows. |
| `session.categories` | list of str | Sorted unique labels. Kept in sync automatically when you call `session.set_labels(...)`. |
| `session.artifacts` | dict | A catch-all for anything else a stage produces (tokens, patterns, reduced coordinates, and so on), keyed by whatever name you choose. |
| `session.results_store` | dict | Populated by `session.store_result(...)`; don't write to it directly. |
| `session.pending_figure` | any or `None` | Set this to a matplotlib figure if your tool produces a plot; the chat widget displays it, saves it to `plots/` under a name tied to the result's id, and clears it automatically. Your tool should never call `savefig(...)` or `plt.show()` itself; just set this attribute and return, the same as any other result. |

Two helper methods you'll use in most tools:

- `session.next_result_id(prefix: str) -> str`: returns a fresh id like
  `variance_3`, incrementing a shared counter.
- `session.store_result(tool_name, args, summary, full_report=None) -> str`:
  records a tool's result under `summary["result_id"]`. `summary` must be
  a small, JSON-serializable dict; it's what the chat widget and the
  grounding check see. `full_report`, if given, can be a larger object
  (e.g. a full DataFrame) for tools like `visualize_result_tool` to read
  back later; it never gets sent to the LLM directly.

For tools that build on data loading (`build_dtm_tool` and everything
after it), call `session.set_labels(...)` once you know the per-document
category labels; this also populates `session.categories`.

## Why the schema and the docstring aren't optional polish

A regular Python function only needs to work; a tool needs to work *and*
be understandable to something that has never read its source. The
agent never sees your function body. What it actually gets, every time
it decides whether to call a tool, is a schema generated from the
`@tool` decorator: the tool's name, its parameter names and types, and
its docstring, packaged into the same kind of structured description
the model's function-calling API expects. If that schema is vague or
missing detail, the agent isn't being lazy when it calls a tool wrong or
can't explain what one does. It genuinely has nothing else to go on.

That's why parameter names should say what they mean
(`threshold`, not `t`; `target_class`, not `x`), and why a docstring
needs to state, in the first line, what the tool does and what it's
for, followed by an `Args:` block describing each parameter, the same
way the tools in `premade_tools/` are written. A vague docstring like
"filters stuff" gives the agent nothing to reason about when a student
asks what the tool does, whether it needs a document-term matrix first,
or what a given parameter controls. A precise one lets the agent answer
those questions correctly from the schema alone, without guessing.

Not every type is safe in that schema either. `agent_dev/BACKEND_SCHEMA_NOTES.md`
covers this in full: a fixed-size tuple like `Tuple[int, int]` (the
natural type for something like `build_dtm_tool`'s n-gram range) breaks
on Gemini specifically, even though Groq accepts it fine. Read that
page yourself, and know that the agent is instructed to check it too
before typing any parameter, so a correctly-behaving agent should avoid
this on its own.

## The function the loader looks for

Each `.py` file in this workspace should define a module-level function:

```python
def make_tools(session):
    ...
    return [some_tool, another_tool]
```

`make_tools` receives the shared `session` object and returns a list of
LangChain tool objects (built with the `@tool` decorator from
`langchain_core.tools`, or `tool("name")(func)` for a tool whose
docstring needs to be built dynamically). The loader calls `make_tools`
on every file that has it and merges the results into one agent. A file
without a `make_tools` function is simply skipped, with a note printed
by the loader; it won't block the other files from loading.

Each tool function should:
- return a small, JSON-serializable dict as its result (never a raw
  DataFrame or matrix)
- call `session.store_result(...)` before returning, so the result gets
  an id the student can reference later
- read and write `session`'s attributes directly (it's available via
  closure, since `make_tools` receives it as an argument)

## What this does and doesn't tell you

This page tells you the shape your code needs to fit into the loader,
the same way a function signature or an API contract would. It doesn't
tell you what any specific tool should compute, how it should be
structured internally, or what its docstring should say. That's the
actual exercise, and it's still yours to work out.
