"""Provided tool for charting a stored result: visualize_result_tool.

Like load_dataset_tool, this is given to you already built rather than
something you write yourself. It doesn't compute anything new -- it
only renders a bar chart of a result some other tool already stored via
session.store_result(...), which is why it's provided rather than part
of the eighteen tools you build.

It works generically against any stored full_report, not by knowing
about specific tool names: it expects full_report to be a DataFrame
whose first column is the label to put on the x-axis and whose second
column is the number to plot on the y-axis, with rows already sorted in
the order they should be charted (it takes the head, in whatever order
it finds, and does no sorting of its own). See
agent_dev/TOOLS_PIPELINE_BREAKDOWN.md for the exact contract your own
variance_filter_tool, pearson_filter_tool, and spearman_filter_tool
need to follow for this to work against their results.
"""

import matplotlib.pyplot as plt
from langchain_core.tools import tool


def _build_figure(record, full_report, top_n):
    df = full_report.head(top_n)
    label_col, value_col = df.columns[0], df.columns[1]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(df[label_col].astype(str), df[value_col])
    ax.set_title(f"{record.tool_name}: {record.result_id}")
    ax.set_ylabel(value_col)
    ax.tick_params(axis="x", rotation=75)
    fig.tight_layout()
    plt.close(fig)  # caller displays explicitly; don't double-render
    return fig


def make_tools(session):
    @tool
    def visualize_result_tool(result_id: str, top_n: int = 15) -> dict:
        """Render a bar chart of a previously stored result, referenced
        by its exact result_id (e.g. 'variance_1', 'pearson_2'). Does
        not compute anything new -- only visualizes an existing result.

        Args:
            result_id: the exact id of a previously stored result.
            top_n: how many top rows to plot.
        """
        if result_id not in session.results_store:
            return {
                "error": f"Unknown result_id '{result_id}'.",
                "known_result_ids": list(session.results_store.keys()),
            }
        entry = session.results_store[result_id]
        record, full_report = entry["record"], entry["full_report"]
        if full_report is None:
            return {"error": f"Result '{result_id}' has no tabular data to visualize."}
        if not hasattr(full_report, "columns") or len(full_report.columns) < 2:
            return {
                "error": (
                    f"Result '{result_id}''s full_report isn't a table with at "
                    "least two columns (a label column and a value column)."
                )
            }

        fig = _build_figure(record, full_report, top_n)
        session.pending_figure = fig
        rows_plotted = min(top_n, len(full_report))
        return {"visualized_result_id": result_id, "rows_plotted": rows_plotted}

    return [visualize_result_tool]
