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
| `path_comparison.test.cjs` | Component alignment, structural differences, Q/K/V interventions, and missing metadata. |
| `path_diagram.test.cjs` | Diagram metadata, patch variants, clean-token labels, and Unicode/text wrapping. |
| `path_exploration.test.cjs` | Provenance indexing and validation, path identities, selection, sorting, and stable lane geometry. |
| `path_inspector.test.cjs` | Application integration: headers, help, resizing, graph/list selection, comparisons, scores, diagrams, filtering, and result/live-run state. |
| `path_inspector_layout.test.cjs` | Real Chromium layout, responsive sizing, overflow, keyboard/pointer controls, and circuit-navigation stability. |
| `path_presentation.test.cjs` | Score formatting, saved metrics, units, direction, tooltips, concise difference labels, and component probability formatting (including unavailable values). |
| `path_fixtures.cjs` | Shared graph/occurrence factories, not tests. Each call returns fresh mutable data. |

### Python tests

| File | Coverage |
| --- | --- |
| `test_component_logit_lens.py` | MLP/head decoding, clean-row/position and softmax contracts, activation validation, hook safety, live caching/retries and job isolation, diagnostics, and JSON scrubbing of invalid readout probabilities. |
| `test_experiment.py` | Experiment validation, root/metric/algorithm setup, saving, and algorithm × strategy × metric execution. Loads GPT-2. |
| `test_joint_scoring.py` | Cached tree messages, incremental evaluation, isolated versus joint candidate scores, and simultaneous admission. Loads GPT-2. |
| `test_nodes.py` | Node forward messages, patching, expansion candidates, and gradients. Loads GPT-2 and Qwen. |
| `test_path_provenance.py` | Original path/occurrence identity, duplicate Q/K/V display routes, merged tree diamonds, root-only/incomplete trees, signed scores and search filtering, scoring modes, occurrence deduplication/shared references, and completed search payloads. |

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
*Replace `test_component_logit_lens.py` with the desired filename to run the other visualization tests individually.*  

The `-v` option displays the results of individual test cases.