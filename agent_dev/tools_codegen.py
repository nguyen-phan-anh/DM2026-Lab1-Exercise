"""LangChain tools for the agent-dev (code-authoring) track: the agent
helps the student write the pipeline tool modules themselves, rather than
running the pipeline for them.

Write access is a two-step, two-turn process, mirroring the same
structural discipline as the pipeline-runner agent (one tool call per
turn, explicit approval before anything consequential happens):

1. propose_file_tool shows a diff or preview of a proposed file. It does
   not touch disk. It requires the student's next message to explicitly
   approve before anything is written, which the one-tool-call-per-turn
   cap enforces: propose and apply can never happen in the same turn.
2. apply_pending_write_tool writes only what was most recently proposed
   and not yet applied or discarded. This is the only tool in this
   module that touches disk, other than run_tests_tool's own test-run
   side effects (whatever the test itself does, e.g. building a DTM in
   memory -- nothing persists past the subprocess).

All file access (listing, reading, proposing, applying) is confined to
`workspace_dir` (default agent_dev_workspace/, a directory separate from
everything else in the repo). There's no reference implementation for
the agent to reach even if this confinement failed: no finished version
of these tools ships in the repo a student receives at all, only in the
maintainer's own copy (used to compute TEST_FIXTURE.md's known-answer
numbers).

run_tests_tool is different in kind from the others: it doesn't narrate
or write anything, it actually runs a test file the student already
wrote and approved, with real pytest, and reports the real result. The
point is that "the test passes" should mean the test process actually
exited 0, not that the model claimed it would.
"""

import difflib
import os
import subprocess
import sys

from langchain_core.tools import tool

_NO_PENDING_ERROR = {"error": "No pending write to apply. Propose a file first with propose_file_tool."}
_TEST_TIMEOUT_SECONDS = 60

# The only files outside the student's workspace this agent can ever read.
# Each is a spec, not a reference implementation -- the confinement in
# _resolve_within_workspace exists to keep a finished solution out of
# reach, and none of these three are that. Fixed set, fixed relative
# paths, no student- or model-supplied path involved, so there's no
# traversal risk in allowing this narrow exception.
_SPEC_FILES = {
    "TOOLS_PIPELINE_BREAKDOWN.md": os.path.join("agent_dev", "TOOLS_PIPELINE_BREAKDOWN.md"),
    "TEST_FIXTURE.md": os.path.join("agent_dev", "TEST_FIXTURE.md"),
    "BACKEND_SCHEMA_NOTES.md": os.path.join("agent_dev", "BACKEND_SCHEMA_NOTES.md"),
}
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _resolve_within_workspace(session, path):
    root = os.path.abspath(session.workspace_dir)
    os.makedirs(root, exist_ok=True)
    target = os.path.abspath(os.path.join(root, path))
    if target != root and not target.startswith(root + os.sep):
        return None
    return target


def make_tools(session):
    @tool
    def list_workspace_files_tool(subdirectory: str = ".") -> dict:
        """List files/folders in the student's workspace (their own
        in-progress files only -- this cannot see anything outside the
        workspace, including any finished reference implementation).

        Args:
            subdirectory: path relative to the workspace root.
        """
        target = _resolve_within_workspace(session, subdirectory)
        if target is None or not os.path.isdir(target):
            return {"error": f"'{subdirectory}' is not a valid directory within the workspace."}
        entries = sorted(os.listdir(target))
        dirs = [e for e in entries if os.path.isdir(os.path.join(target, e))]
        files = [e for e in entries if os.path.isfile(os.path.join(target, e))]
        return {"subdirectory": subdirectory, "directories": dirs, "files": files}

    @tool
    def read_workspace_file_tool(file_path: str) -> dict:
        """Read a file already in the student's workspace (e.g. to review
        code written in an earlier turn). Confined to the workspace --
        cannot read anything outside it.

        Args:
            file_path: path relative to the workspace root.
        """
        target = _resolve_within_workspace(session, file_path)
        if target is None or not os.path.isfile(target):
            return {"error": f"File not found in workspace: '{file_path}'."}
        with open(target, encoding="utf-8") as f:
            content = f.read()
        return {"file_path": file_path, "content": content, "n_lines": content.count("\n") + 1}

    @tool
    def propose_file_tool(file_path: str, content: str, explanation: str) -> dict:
        """Propose writing `content` to `file_path` in the student's
        workspace. Shows a diff against the current file (or a full
        preview if the file is new) -- does NOT write anything. The
        student must explicitly approve in a later message before
        apply_pending_write_tool can be called; this tool only stages the
        proposal.

        Args:
            file_path: path relative to the workspace root, e.g.
                'tools_filtering.py'.
            content: the full proposed file content.
            explanation: a short note on what this code does and why,
                for the student to review alongside the diff.
        """
        target = _resolve_within_workspace(session, file_path)
        if target is None:
            return {"error": f"'{file_path}' is not a valid path within the workspace."}

        if os.path.isfile(target):
            with open(target, encoding="utf-8") as f:
                old_content = f.read()
        else:
            old_content = ""

        diff = "".join(
            difflib.unified_diff(
                old_content.splitlines(keepends=True),
                content.splitlines(keepends=True),
                fromfile=f"current/{file_path}",
                tofile=f"proposed/{file_path}",
            )
        )
        if not diff:
            diff = "(no changes -- proposed content is identical to the current file)"

        from .dev_session_state import PendingWrite

        session.pending_write = PendingWrite(file_path=file_path, content=content, explanation=explanation)

        result_id = session.next_result_id("propose_file")
        summary = {
            "result_id": result_id,
            "file_path": file_path,
            "explanation": explanation,
            "is_new_file": old_content == "",
            "diff": diff,
        }
        session.store_result("propose_file", {"file_path": file_path}, summary)
        return summary

    @tool
    def apply_pending_write_tool() -> dict:
        """Write the most recently proposed (and not yet applied or
        discarded) file to disk. Only call this after the student has
        explicitly approved the proposal shown by propose_file_tool --
        never call it in the same turn as a proposal, and never guess
        that approval was implied."""
        if session.pending_write is None:
            return _NO_PENDING_ERROR

        target = _resolve_within_workspace(session, session.pending_write.file_path)
        if target is None:
            return {"error": f"'{session.pending_write.file_path}' is not a valid path within the workspace."}

        os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            f.write(session.pending_write.content)

        written_path = session.pending_write.file_path
        session.pending_write = None

        result_id = session.next_result_id("apply_write")
        summary = {"result_id": result_id, "file_path": written_path, "status": "written"}
        session.store_result("apply_write", {"file_path": written_path}, summary)
        return summary

    @tool
    def run_tests_tool(test_file: str) -> dict:
        """Actually runs a test file already written in the student's
        workspace, with pytest, and reports the real result. This does
        not narrate or guess at whether a test passes -- it runs the
        real process and reports its real exit code and output.

        The test file's own imports resolve relative to the workspace
        (so `from tools_filtering import make_tools` finds the
        student's own tools_filtering.py), while the process itself
        runs from the repo root, so a path like
        "agent_dev/sample_fixture.csv" inside the test
        resolves the same way it does everywhere else in this project.

        Args:
            test_file: path to the test file, relative to the
                workspace, e.g. "test_tools_filtering.py".
        """
        target = _resolve_within_workspace(session, test_file)
        if target is None or not os.path.isfile(target):
            return {"error": f"'{test_file}' is not a valid file within the workspace."}

        repo_root = os.path.abspath(os.getcwd())
        env = os.environ.copy()
        workspace_root = os.path.abspath(session.workspace_dir)
        existing_path = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = workspace_root + (os.pathsep + existing_path if existing_path else "")

        try:
            result = subprocess.run(
                [sys.executable, "-m", "pytest", target, "-v", "--tb=short"],
                capture_output=True,
                text=True,
                timeout=_TEST_TIMEOUT_SECONDS,
                cwd=repo_root,
                env=env,
            )
        except subprocess.TimeoutExpired:
            return {
                "test_file": test_file,
                "passed": False,
                "error": f"Test run did not finish within {_TEST_TIMEOUT_SECONDS} seconds and was stopped.",
            }

        output = (result.stdout + result.stderr)
        # Long pytest output (a stack trace on a real failure especially)
        # doesn't need to reach the model in full to explain what broke;
        # the tail has the actual failure, the head just has setup noise.
        if len(output) > 4000:
            output = "...(truncated)...\n" + output[-4000:]

        result_id = session.next_result_id("run_tests")
        summary = {
            "result_id": result_id,
            "test_file": test_file,
            "passed": result.returncode == 0,
            "exit_code": result.returncode,
            "output": output,
        }
        if "No module named pytest" in output:
            # Not a test failure -- pytest itself isn't installed in
            # whatever Python environment is running this kernel (it's
            # declared in pyproject.toml/requirements.txt, but that only
            # takes effect once someone actually installs from them into
            # *this* environment). Rewriting the test to avoid pytest
            # wouldn't help: this tool always shells out to `pytest`
            # specifically, regardless of how the test file is written.
            summary["environment_error"] = (
                f"pytest is not installed in this Python environment ({sys.executable}). "
                "This is not a problem with the test file or the tool under test -- "
                "install it with `pip install -r requirements.txt` (or `pip install pytest`) "
                "in that same environment, then ask to run the test again. Rewriting the "
                "test to avoid pytest will not fix this, since this tool always runs pytest "
                "specifically."
            )
        session.store_result("run_tests", {"test_file": test_file}, summary)
        return summary

    @tool
    def discard_pending_write_tool() -> dict:
        """Discard the current pending proposal without writing it (e.g.
        if the student asks for changes before approving)."""
        if session.pending_write is None:
            return _NO_PENDING_ERROR
        discarded_path = session.pending_write.file_path
        session.pending_write = None
        result_id = session.next_result_id("discard_write")
        summary = {"result_id": result_id, "file_path": discarded_path, "status": "discarded"}
        session.store_result("discard_write", {"file_path": discarded_path}, summary)
        return summary

    @tool
    def read_spec_file_tool(spec_name: str) -> dict:
        """Read the actual, current text of one of this course's three
        spec documents. This is the ONLY file access this agent has
        outside the student's own workspace, and only these three exact
        files -- nothing else, no reference implementation anywhere in
        reach.

        Always call this for the relevant spec before proposing a
        tool's code, a test's expected numbers, or advice about a
        parameter's type, if you have not already read that spec
        earlier in this same conversation. These files can't change
        mid-session, so reading one once per conversation is enough;
        don't re-read the same spec on every single turn.

        Args:
            spec_name: exactly one of "TOOLS_PIPELINE_BREAKDOWN.md" (the
                tool contract: the session object, make_tools shape,
                per-tool requirements like mine_patterns_tool's
                arguments), "TEST_FIXTURE.md" (the fixture data and all
                twenty numbered known answers to test against), or
                "BACKEND_SCHEMA_NOTES.md" (which parameter types break
                on Gemini).
        """
        relative_path = _SPEC_FILES.get(spec_name)
        if relative_path is None:
            return {"error": f"Unknown spec_name '{spec_name}'. Valid options: {sorted(_SPEC_FILES)}"}
        target = os.path.join(_REPO_ROOT, relative_path)
        if not os.path.isfile(target):
            return {"error": f"Spec file missing from disk: '{relative_path}'."}
        with open(target, encoding="utf-8") as f:
            content = f.read()
        return {"spec_name": spec_name, "content": content, "n_lines": content.count("\n") + 1}

    return [
        list_workspace_files_tool,
        read_workspace_file_tool,
        propose_file_tool,
        apply_pending_write_tool,
        run_tests_tool,
        discard_pending_write_tool,
        read_spec_file_tool,
    ]
