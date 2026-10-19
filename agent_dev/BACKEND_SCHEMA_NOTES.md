# Backend schema notes: parameter types that don't travel well

(New terms? `agent_dev/GLOSSARY.md` has the full list; a number like
this<sub>1</sub> points back to it.)

When you write a **tool**<sub>5</sub>, the **type hint**<sub>21</sub>
you give each parameter (`list[str]`, `int`, and so on) doesn't just
describe your code to a human. It also gets turned into a
**schema**<sub>20</sub>, a description of what shape the model is
allowed to send for that parameter, and sent to whichever
**backend**<sub>12</sub> you're using (Groq or Gemini). This happens
every single turn, for every tool the agent has, not just the one
actually being called that turn.

Most Python types turn into a schema the same way no matter which
backend reads it. A couple of types don't, and this page exists so you
don't have to work that out yourself if you hit one.

The agent has one tool, `read_spec_file_tool`, that can read this exact
page live (along with `TOOLS_PIPELINE_BREAKDOWN.md` and
`TEST_FIXTURE.md`), the one narrow exception to its file tools
otherwise being confined to your own workspace. It's instructed to
call it before deciding any parameter's type, so it should already
avoid whatever's documented here without you having to say anything.
Still worth reading yourself, for two reasons: you'll recognize the
problem immediately if you ever see it in a raw error message instead
of taking the agent's word for it, and the agent can only ever avoid
what's *already written here*: it has no way to discover a new case
on its own, so if you hit one that isn't listed, you're the one who
has to notice and flag it. The agent still can't write to this page
itself either way; if you find a new case, you add it.

## Why a schema can make Gemini fail

Gemini checks a tool's schema more strictly than Groq does. A schema
Gemini doesn't like gets rejected before the agent ever gets to run
anything, every single turn, with an error naming the parameter it
didn't like. The tool's actual logic is never even reached; nothing is
wrong with your code, only with the type you gave one parameter.

This is a real trap for testing: if you build and test a tool on Groq
only, it can look completely correct, then fail the moment you (or a
classmate) switch to Gemini, for a reason that has nothing to do with
what the tool computes.

## Known problem: a tuple with a fixed length

```python
ngram_range: Tuple[int, int] = (1, 1)
```

This looks like the natural type for something like `build_dtm_tool`'s
n-gram range. It's exactly the kind of type that trips Gemini up:
**Gemini rejects it outright**, while **Groq accepts it without
complaint** and calls the tool correctly.

## What to tell the agent

You shouldn't need to tell it anything for this specific case; it's
instructed to read this page before typing any parameter, so a
correctly-behaving agent avoids the fixed-length tuple on its own. If
you ever see it slip through anyway, or a call still fails on Gemini
naming a specific parameter, say so plainly: "check
BACKEND_SCHEMA_NOTES.md again, this parameter's type may not work on
Gemini." Once it says it's fixed, ask it to run the tool again on the
same backend so you can both actually confirm the error is gone,
rather than taking its word for it.

## Verified fine on both backends

These types were each actually tried on a real tool, on both Gemini
and Groq, and work fine, so there's no need to avoid them:

- `List[int]`, and lists of other simple types
- `List[dict]` (a list of arbitrary records)
- `Dict[str, int]` (and other simple key/value mappings)
- `Union[int, float]`
- `Optional[int]`, and `Optional[Union[int, str]]` (nested)
- `Literal["a", "b", "c"]` (e.g. `reduce_dimensions_tool`'s `method`
  parameter)
- plain `dict`

## If you hit something not on this list

If a tool call fails with an error from Gemini blaming a specific
parameter, that's real signal, not something to guess your way around.
Try the same tool on Groq first: if it works there, the type is the
problem, not the tool's logic. Give the agent that context and ask it
to fix the parameter's type; it doesn't need a specific replacement
type from you, just to know Gemini rejected this one.

The agent can read this page, but it still can't write to it: its
file tools other than `read_spec_file_tool` are confined to your own
workspace, and `read_spec_file_tool` is read-only. Once the type is
fixed, add the new case here yourself: the type that failed and the
type that fixed it, so the next student (and the agent, next time it
reads this page) doesn't have to work it out from scratch either.
