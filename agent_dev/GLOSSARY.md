# Glossary

New to building with AI agents, or to some of the programming words
around it? This page explains every term used in
`DM2026-Lab1-AgentDev.ipynb`, `DM2026-Lab1-AgenticPipeline.ipynb`, and
the reference pages in `agent_dev/` (like `BACKEND_SCHEMA_NOTES.md`)
that might be new to you. If you see a word in **bold** with a small
number after it, like this<sub>3</sub>, that number is this page.

This page covers agent and programming vocabulary only. Data mining
concepts (variance, correlation, and so on) are taught in
`DM2026-Lab1-Master.ipynb`.

---

## Part 1: AI agent terms

### 1. LLM
Short for Large Language Model. This is the AI itself (Gemini or Groq
in this course). Think of it like a very good autocomplete: you give
it text, and it predicts what should come next. It has no memory
between separate requests, can't read a file, and can't run code on
its own.

### 2. Token
The small chunk of text an LLM predicts at a time. Usually a word or
part of a word, not always a whole one.

### 3. Transformer
The type of neural network an LLM is built from. You don't need to
know how it works for this course, just that it's the reason the
notebook calls an LLM "a transformer that predicts the next token."

### 4. Agent
An **LLM**<sub>1</sub> plus **tools**<sub>5</sub> plus a
**harness**<sub>6</sub>. Together they let the model's predicted text
turn into a real action, like writing a file, instead of staying just
words on a screen.

### 5. Tool
A normal Python function, given a name, typed inputs, and a short
description, so the model can ask for it by name. The model never
runs the tool itself; it only predicts "call this tool with these
arguments," and something else actually runs it.

### 6. Harness
The code that turns an LLM and some tools into a working agent. It
resends the conversation each time, notices when the model asks for a
tool, runs it for real, and gives the result back so the model can
react. In this course, that's `graph_core.py` and the chat widget,
already built for you.

### 7. LangGraph
The library the harness is built with. Instead of hand-writing the
loop of "send a message, check for a tool request, run it, repeat,"
you describe it as a graph: a set of steps connected by paths that say
what happens next.

### 8. Node
One named step in a LangGraph graph. This course's graph has three:
`agent` (calls the model), `tool_node` (runs whatever tool was
requested), and `discuss` (lets the model comment on the result).

### 9. Edge
A connection between two **nodes**<sub>8</sub> that says what runs
next. This is what makes a rule like "one tool call, then a comment,
then stop" hold every time. It's built into the graph's shape, not
something the model has to remember.

### 10. Memory
What an agent "remembers." In this course, it's just the list of
messages so far, resent every time, and gone once the kernel restarts.

### 11. Vector store
A way of storing memory so it can be searched by meaning instead of
just replayed in order. Mentioned as an example of a more advanced
memory system, not something this course's harness uses.

### 12. Backend
Which company's model actually answers your request: Groq or Gemini,
in this course.

### 13. Rate limit
A cap a provider puts on how many requests you can make in a given
time. Hit it, and that provider stops answering until it resets (once
a day, on the free plans used here).

### 14. Multi-provider fallback
If your backend hits a **rate limit**<sub>13</sub> mid-session, the
system automatically tries a different one instead of leaving you
stuck.

### 15. Grounding check
Checking that a number the agent says is actually traceable back to a
real tool result, not something it guessed.

### 16. Hallucination
When an LLM states something false, confidently, without meaning to.
There's no intent to deceive, the way there is with a lie; it's just
wrong. This is why this course insists on actually running things,
like tests, instead of trusting the model's own claim that something
worked.

### 17. Stochastic
Can give a different result each time, even for the same input. An
LLM's answers are stochastic.

### 18. Deterministic
Always gives the same result for the same input. A test is
deterministic: run it twice against the same code and you get the
same pass or fail both times.

---

## Part 2: Programming terms

### 19. Decorator
A Python feature that wraps a function to change how it's used,
without changing what the function itself does. `@tool` is the
decorator every tool in this course starts with.

### 20. Schema
A description of what shape some data should have. Here, it's a
tool's parameters, built from its **type hints**<sub>21</sub> and
**docstring**<sub>22</sub>, and sent to the model so it knows what
it's allowed to pass in.

### 21. Type hint
A note on a function's parameters, like `categories: list[str]`,
saying what type of value is expected. In this course these matter
more than usual, since they become part of the
**schema**<sub>20</sub> the model reads.

### 22. Docstring
The description at the top of a function. The model reads this to
decide when to call a tool and what to pass it.

### 23. Interface
The agreed shape a piece of code has to match, so other code can rely
on it without needing to see how it works inside. Like a plug needing
to fit a socket, regardless of what's wired up behind the wall.

### 24. Contract
Close to **interface**<sub>23</sub>: the specific set of promises a
piece of code makes (what it takes in, what it gives back) that other
code is allowed to depend on.

### 25. Workspace
The folder your own code lives in (`agent_dev_workspace/` in this
course). The agent's file tools can't read or write outside it.

### 26. Diff
A view showing exactly what would change in a file: lines added,
lines removed, shown before anything is actually written, so you can
review it first.

### 27. Gate
A rule that blocks an action until some condition is met. Example: a
file can't be written until you've approved a proposal for it first.

### 28. State machine
A system that's always in exactly one of a fixed set of states, with
clear rules for what can happen next. A LangGraph<sub>7</sub> graph is
one.

### 29. Structural (rule)
Something true because of how the system is built, not because of an
instruction someone has to remember to follow. "One tool call per
turn" holds here because the graph is built that way, not because the
model was told to behave.

### 30. Subprocess
A genuinely separate program your computer runs. `run_tests_tool`
launches `pytest` as a real subprocess, so a test result comes from
Python actually running your code, not from the AI describing what it
thinks would happen.

### 31. Exit code
The number a program hands back when it finishes, saying whether it
worked. This is the actual signal `run_tests_tool` checks to know if
your test passed.

### 32. JSON-serializable
Data simple enough to convert cleanly into plain text: numbers,
strings, lists, and dictionaries. Every tool in this course has to
return something like this, never a raw table or matrix object.

### 33. Reference implementation
A finished, correct version of the code you're being asked to write.
Deliberately kept from you in this course, so you build your own
solution instead of copying an answer.

### 34. Spec-driven development
Agreeing on one written specification first (here,
`TOOLS_PIPELINE_BREAKDOWN.md` and `TEST_FIXTURE.md`), then building
and testing against that one shared reference, instead of describing
what you want fresh, slightly differently, in every new message.

---

## Part 3: Testing terms

### 35. Test
Code that runs your function on a known input and checks the output
with `assert`. In this course, a test is a real file, written the
same way any tool file is: proposed, approved, then written to disk.

### 36. Fixture
A small, fixed set of input data with a correct answer that was
worked out some other way, not by running your own code. A fixture
gives a test something real to check against, unlike the full
dataset, which is too big and whose "correct" answers usually aren't
known ahead of time.

### 37. Function
A named, reusable block of code that takes some input and does
something with it, usually returning a result. Every **tool**<sub>5</sub>
in this course is a function first; `@tool` is what turns a plain one
into something the model can call.

### 38. pytest
The Python library `run_tests_tool` actually uses to run a test file.
Not something this course's code reimplements: a real, independent
tool, the same one used outside this course, so "the test passed"
means pytest itself said so, not the agent.

### 39. Assert
A Python statement that checks a value is what you expect, e.g.
`assert n_documents == 8`. If the value doesn't match, Python raises an
error right there, which is what makes a **test**<sub>35</sub> "fail"
instead of just quietly running to the end.
