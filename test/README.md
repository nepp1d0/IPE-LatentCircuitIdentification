# Tests

The tests cover IPE's core Python functionality, component readouts and interactive visualization using `pytest` and Node.js's built-in `node:test` frameworks. The Python tests are located directly in the `test/` directory, while the JavaScript visualization tests are organized under `test/visualization/`.

## Directory structure

```text
test/
├── __init__.py
├── README.md
├── test_component_logit_lens.py
├── test_experiment.py
├── test_joint_scoring.py
├── test_nodes.py
├── test_path_provenance.py
└── visualization/
    ├── component_decoding.test.cjs
    ├── path_comparison.test.cjs
    ├── path_diagram.test.cjs
    ├── path_exploration.test.cjs
    ├── path_inspector.test.cjs
    ├── path_inspector_layout.test.cjs
    ├── path_presentation.test.cjs
    └── path_fixtures.cjs
```

### JavaScript visualization tests

| File | Coverage |
| --- | --- |
| `component_decoding.test.cjs` | Component readout validation, token positions and fallback, Top-9 probabilities, numerical formatting, axis scaling, and responsive chart geometry. |
| `path_comparison.test.cjs` | Component alignment, structural differences, Q/K/V interventions, and missing metadata. |
| `path_diagram.test.cjs` | Component metadata, patch variants, token labels, and SVG text wrapping. |
| `path_exploration.test.cjs` | Path provenance, identities, selection, sorting, and stable graph lanes. |
| `path_inspector.test.cjs` | Path Inspector integration: navigation, selection, comparisons, scores, filtering, state management, component decoding, and information popovers. |
| `path_inspector_layout.test.cjs` | Browser-based layout validation: responsive sizing, scrolling, navigation, histogram readability, probability labels, and popover behavior. |
| `path_presentation.test.cjs` | Score and probability formatting, metric descriptions, tooltips, and path difference labels. |
| `path_fixtures.cjs` | Shared graph and occurrence factories for tests (not a test file). |

### Python tests

| File | Coverage |
| --- | --- |
| `test_component_logit_lens.py` | MLP/head decoding, token positions, full-vocabulary probabilities, activation validation, caching, hook safety, error handling, and Top-9 readouts. |
| `test_experiment.py` | Experiment validation, setup, saving, and algorithm × strategy × metric execution. Loads GPT-2. |
| `test_joint_scoring.py` | Cached tree messages, incremental evaluation, isolated/joint scoring, and simultaneous admission. Loads GPT-2. |
| `test_nodes.py` | Node messages, patching, expansion candidates, and gradients. Loads GPT-2 and Qwen. |
| `test_path_provenance.py` | Original path identities, Q/K/V occurrences, tree merging, scoring modes, search results, and serialization, including Top-9 configuration. |

The `benchmark/Automatic-Circuit-Discovery` submodule contains its own upstream tests, which are not included in this suite.

## Running tests

Run the following commands from the repository root, using the appropriate Python environment.

**Requirements:** Python 3.12+ with the project dependencies and Node.js 22+.

### JavaScript tests

Run the entire JavaScript test suite:
```bash
node --test "test/visualization/*.test.cjs"
```
Run a specific test file:
```bash
node --test test/visualization/path_exploration.test.cjs
```
*Replace `path_exploration.test.cjs` with the desired filename to run the other visualization tests individually.*

**Note:** The browser-based layout test requires Chromium, Chrome, or Edge. It may be skipped if a compatible browser is unavailable.

### Python tests

Run the entire Python test suite:
```bash
python -m pytest ./test
```
Run a specific test file:
```bash
python -m pytest test/test_component_logit_lens.py -v
```
*Replace `test_component_logit_lens.py` with the desired filename to run the other Python tests individually.*

The `-v` option displays the results of individual test cases.