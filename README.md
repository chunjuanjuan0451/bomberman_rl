# Bomberman RL final project

This repository contains the source code, training tools, evaluation scripts and
selected checkpoints for our Bomberman RL project. The official game engine is
kept in the repository root and the project agents are under `agent_code/`.

## Final agents

The final submission agent is `agent_code/niulai/`. It is a full-board dueling
CNN with a collision-consistent safety mask and D4 symmetry-averaged inference.
The directory is self-contained: it contains the model definition, feature
construction, safety logic, inference callback and the exported weights.

The final reference agent is the frozen v4 Dueling Double DQN in
`agent_code/model_a_dqn/`. The two source checkpoints are also kept under
`models/final/` for reproducibility:

| Model | File | SHA-256 |
| --- | --- | --- |
| D4 (`niulai`) | `models/final/d4/round-0150.pt` | `3fe04112793ea8fd8fde7f72e7da40f4bb5e54f7ebe0a45f5f1b92f378ac3dee` |
| Frozen v4 | `models/final/v4/model_a.pt` | `d5fe215b1f909149e1eefa2ee6ac00d7365e61343b333ba91b900e11e911d239` |

`agent_code/niulai/model.pt` is the inference-only export used by the 21
September submission. The separate upload archive contains only that agent
directory and its required parameter file.

## Running the agent

The local Python environment needs NumPy and PyTorch. To run ten evaluation-only
rounds against the built-in rule-based agents:

```bash
PYTHONPATH=. python main.py play \
  --my-agent niulai \
  --scenario classic --n-rounds 10 --train 0 \
  --continue-without-training --no-gui
```

The agent uses one CPU thread and resolves its model path relative to its own
directory. It does not use multiprocessing or external agent code.

## Tests and reproducibility

Run the project test entry point from the repository root:

```bash
PYTHONPATH=. python tools/run_unit_tests.py
```

The `experiments/` and `tools/` directories contain the registered training,
evaluation and diagnostic scripts used during the project. The preregistered
JSON configurations are kept under `experiments/configs/`; large rollout logs
and transient checkpoints are intentionally not committed. The `docs/`
directory contains the report handoff notes and the experiment brief. Runtime
outputs, local caches and temporary replay files are ignored by Git.

The historical unit-test collection refers to some of those archived rollout
artifacts. The final agent does not need them: use the `main.py play` command
above for a clean inference smoke test.

The Dockerfile pins the Python and PyTorch versions used for compatibility
checks. A normal container smoke test is:

```bash
docker build -t bomberman-rl-local .
docker run --rm bomberman-rl-local \
  python main.py play --my-agent niulai \
  --train 0 --continue-without-training --no-gui --n-rounds 10
```
