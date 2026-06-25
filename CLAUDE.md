# Contributor Guide

Orientation for contributors and AI coding assistants working on this repository — the code release for the paper *"Hallucination in World Models is Predictable and Preventable"* (the **MMBench2** world model, dataset tooling, and hallucination detection/mitigation method). For installation and usage, see `README.md`.

## What this repository contains
A Dreamer-4-style generative **world model** for large-scale multi-task continuous control (~350M parameters: ≈100M tokenizer + ≈250M dynamics), the **MMBench2** dataset tooling (210 tasks across 10 domains, 224×224 RGB observations, 1–16 action dimensions), and the paper's method: hallucination predictors, coverage-aware training, and curiosity-driven targeted data collection.

## Code layout
All code lives under `src/`:
- `model.py` — shared model components (tokenizer encoder/decoder, dynamics transformer, reward/BC heads); imported by both training scripts.
- `train_tokenizer.py` / `train_dynamics.py` — the two training stages (a causal video tokenizer, then a block-causal flow-matching dynamics model with reward and behavior-cloning heads).
- `uncertainty.py` — the hallucination predictors `u_r` / `u_f` / `u_s` (motion-normalized).
- `collect_data.py` / `curiosity.py` — curiosity-driven targeted data collection.
- `plan_cem.py` — evaluation via CEM planning.
- `interactive.py` (+ `interactive.html`) — the browser interface for open-ended interaction with the model.
- `wm_dataset.py`, `sharded_frame_dataset.py`, `preprocess_dataset.py`, `task_set.py` — dataset loading and preprocessing.
- `envs/` — Gymnasium-compatible simulators for every task.
- `download_dataset.py` / `download_checkpoints.py` — fetch the dataset and pretrained checkpoints from the Hugging Face Hub.

## Dataset / observation convention (load-bearing)
For a trajectory of length `T+1` there are `T+1` observations but only `T` actions and `T` rewards. The first tuple is `(obs_0, nan, nan)` (no action or reward precedes the initial observation); thereafter `(obs_t, action_{t-1}, reward_{t-1})`, where `action_{t-1}` and `reward_{t-1}` describe the transition `obs_{t-1} → obs_t`. Keep this convention in mind for any change to data loading or preprocessing.

## Conventions
- The training scripts use flat imports and must be run from inside `src/`.
- Training is expensive — prefer minimal, well-justified changes, and verify with smoke tests when a change could affect convergence or correctness.
- Add comments only where they clarify non-obvious logic; use American English in docs and comments.

See `CONTRIBUTING.md` for how to propose changes, and `README.md` for installation, training, evaluation, and the interactive interface.
