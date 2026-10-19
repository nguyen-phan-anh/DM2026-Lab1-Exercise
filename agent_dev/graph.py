"""Builds the agent-dev (code-authoring) agent's graph. It binds the
codegen tools to the same shared one-tool-call-per-turn state machine
used by the pipeline-runner agent (agent_pipeline/graph_core.py): the
same structural guardrails, with different tools and a different system
prompt.
"""

from agent_pipeline.graph_core import build_agent_graph

from . import tools_codegen
from .prompts import SYSTEM_PROMPT


def build_graph(session, llm):
    tools = tools_codegen.make_tools(session)
    return build_agent_graph(tools, SYSTEM_PROMPT, llm, session)
