*This project has been created as part of the 42 curriculum by papilaz.*

# Call Me Maybe

## Description

This project implements a function-calling tool that translates natural
language prompts into structured tool calls (`name` + `parameters`) using a
small local language model (`Qwen/Qwen3-0.6B` by default). Instead of asking
the model to freely produce JSON and hoping it gets the syntax right, the
generation is driven by **constrained decoding**: at every generation step,
the logits produced by the model are masked so that only tokens compatible
with a valid tool name, a valid parameter type, or the JSON structure can be
selected. This guarantees syntactically valid, schema-compliant JSON output
even though the underlying model is very small (500M parameters) and would
be unreliable if prompted naively.

## Instructions

### Requirements

- Python 3.10+
- [`uv`](https://docs.astral.sh/uv/) for dependency management

### Install

```bash
make install
# or: uv sync
```

### Run

```bash
make run
# or: uv run python -m src
```

By default, the program reads `data/input/functions_definition.json` and
`data/input/function_calling_tests.json`, and writes the results to
`data/output/function_calling_results.json`.

All paths, and the model used, can be overridden:

```bash
uv run python -m src \
    --functions_definition data/input/functions_definition.json \
    --input data/input/function_calling_tests.json \
    --output data/output/function_calling_results.json \
    --model Qwen/Qwen3-0.6B
```

### Using another model

The `--model` flag lets you replace `Qwen/Qwen3-0.6B` with any other causal
model loadable by `llm_sdk`, for example:

```bash
uv run python -m src --model Qwen/Qwen3-1.7B
```

### Other Makefile targets

```bash
make debug        # run the program under pdb
make test         # run the pytest suite
make lint         # flake8 + mypy (required flags)
make lint-strict  # flake8 + mypy --strict
make clean        # remove caches and generated output
```

## Example Usage

Input (`data/input/function_calling_tests.json`):

```json
[
  { "prompt": "Is -4 an even number?" },
  { "prompt": "Format template: Say \"hello\" to {name}" }
]
```

Output (`data/output/function_calling_results.json`):

```json
[
  {
    "prompt": "Is -4 an even number?",
    "name": "fn_is_even",
    "parameters": { "n": -4 }
  },
  {
    "prompt": "Format template: Say \"hello\" to {name}",
    "name": "fn_format_template",
    "parameters": { "template": "Say \"hello\" to {name}" }
  }
]
```

## Algorithm Explanation

The pipeline (see `src/generator.py`) has three stages, all driven by
constrained decoding rather than free-form prompting:

1. **Tool name selection** (`select_tool_name`): the prompt and the list of
   available tools are encoded once. Then, token by token, the logits are
   masked (`src/masking.py::mask_logits`) so only tokens that extend a valid
   prefix of an allowed tool name (or the `fn_none` fallback) can be chosen
   (`get_allowed_next_tokens`). The loop stops as soon as no candidate name
   remains extendable, which naturally bounds it to the longest tool name.

2. **Parameter generation** (`generate_parameters`): for each parameter of
   the chosen tool:
   - **Numbers** are generated with the logits restricted to digits, a
     decimal point, and the minus sign (in both its tokenized forms, with
     and without a leading space, since the tokenizer merges a preceding
     space into the sign for negative numbers).
   - **Strings** are generated freely (the model usually copies the
     relevant part of the prompt verbatim), then truncated at the first
     unescaped closing quote. Anything the model generated beyond that
     point (extra JSON, hallucinated continuation) is discarded and the
     rest of the JSON structure is added by the program itself, not the
     model — this is what prevents malformed JSON.
   - **Booleans** are generated with the logits restricted to the single
     tokens for `true` and `false`, so the value is always one of the two.

3. **Assembly and cleanup** (`generate_tool_call` + `clean_json_output`):
   the tool name and parameters are assembled into a JSON string, trailing
   commas are removed, the first balanced `{...}` object is extracted
   (dropping anything printed after it), and path-like string values are
   stripped of stray whitespace.

Both the maximum number of generation steps for numbers and strings are
computed from real bounds instead of arbitrary constants: the number bound
is the length of the longest possible JSON float representation, and the
string bound scales with the length of the user prompt (since string values
are copied from it), so the loop can never be cut short on a legitimate
value.

## Design Decisions

- **Constrained decoding lives entirely in `src/generator.py` and
  `src/masking.py`**, separate from prompting (`src/prompting.py`) and I/O
  (`src/parsing.py`, `src/utils.py`), so each concern can be tested and
  reasoned about independently.
- **`Config` is a `pydantic.BaseModel`** (`src/parsing.py`) rather than a
  plain dict, so the run configuration (file paths, model name) is
  validated and self-documenting.
- **Display/coloring code lives in its own module** (`src/display.py`),
  separate from the actual function-calling logic in `src/generator.py` and
  `src/utils.py`.
- **Strings are generated freely and truncated afterward**, rather than
  trying to constrain every character, because the target value (copied
  from the user prompt) is not known in advance and can contain arbitrary
  characters, including quotes that must be escaped.

## Challenges Faced

- **Negative numbers were silently dropped.** The model almost never chose
  the plain `-` token before a digit, because the tokenizer merges a
  preceding space with the minus sign into a single token (`" -"`) that
  was not in the allowed set. Fixed by adding that merged token to the
  allowed set and no longer hard-coding a space before the number.
- **Truncating strings without corrupting them.** Early on, the code kept
  everything the model generated up to and including its own closing
  quote, then appended a separator on top — producing malformed JSON like
  `"query": "...", "..."` when the model tried to continue the JSON itself.
  The fix rebuilds the value from only the text before the first
  unescaped quote, discarding anything generated after it.
- **A prompt containing an internal double quote**
  (`Say "hello" to {name}`) was not copied correctly by the very small
  model when generation was under-constrained. Making the few-shot prompt
  explicit about escaping internal quotes (with a worked example) was
  enough to fix it once the JSON-corruption bug above was also fixed.

## Performance Analysis

- **Structural validity**: the braces, separators, and delimiting quotes of
  the JSON structure are always written by the program itself, never
  sampled from the model — which eliminates most JSON syntax errors that
  free-form prompting would produce. A string's value, however, is still
  freely generated text before being truncated: in the edge case where the
  model never produces a closing quote within the allotted token budget,
  the corresponding entry falls back to the error mechanism (`error` +
  `raw_output`) instead of silently producing an invalid output.
- **Speed**: each prompt is processed in a few seconds on CPU, well under
  the 5-minute budget set by the subject for the full test set.
- **Reliability**: generation is deterministic (greedy decoding via
  `argmax`, no sampling), so results are stable across repeated runs for a
  given prompt and function set.

## Testing Strategy

Unit tests (`tests/`, run with `make test`) cover every pure function that
does not require loading the model: parameter cleanup and JSON extraction
(`test_utils.py`), the logits-masking primitives (`test_masking.py`),
prompt construction (`test_prompting.py`), and the `Config` class,
including CLI argument parsing and error handling on malformed input files
(`test_parsing.py`).

End-to-end validation is done by running the full pipeline against the
provided `data/input/` files and manually inspecting the generated output
file (`data/output/function_calling_results.json`) to check that each
entry contains the right function name and parameters.

## Resources

- Project subject: *"call me maybe — Introduction to function calling in
  LLMs"* (provided separately, not included in this repository)
- [Hugging Face `transformers` documentation](https://huggingface.co/docs/transformers)
- [pydantic documentation](https://docs.pydantic.dev/)

### AI usage

- **Understanding tokenization** at the very start of the project: how the
  `llm_sdk` tokenizer encodes/decodes text, and how a preceding space can
  merge with the next character into a single token — knowledge that later
  explained why negative numbers were being dropped.
- **Writing part of the unit test suite** (`tests/`) for the pure,
  model-free functions.
- **Writing this documentation.**