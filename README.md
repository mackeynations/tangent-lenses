# Tangent-projected Jacobian lenses

Reference code for the tangent/normal decomposition experiments in our paper,
[How to track your tangent space](tangent_paper.pdf).
The repository isolates the three reported experiment families and the two
tangent-space estimators used:

| experiment | pushforward tangent | direct-PCA tangent |
| --- | --- | --- |
| local oracle | `J_x P_push(x)`, `J_x (I - P_push(x))` | `J_x P_dir(x)`, `J_x (I - P_dir(x))` |
| fitted lens | `E[J_x P_push(x)]`, `E[J_x (I - P_push(x))]` | `E[J_x P_dir(x)]`, `E[J_x (I - P_dir(x))]` |
| evaluation of projected components | `E[J] P_push(h)`, `E[J] (I - P_push(h))` | `E[J] P_dir(h)`, `E[J] (I - P_dir(h))` |

The fitted direct-PCA lens is a samplewise average of products. It is not the
different, state-dependent construction `E[J] P_dir(x)`. Tangent and normal
terms are accumulated on identical prompt-position samples, and every output
contains a numerical audit of

```text
E[J] = E[J P_T] + E[J P_N].
```

The code therefore keeps three distinct operations explicit:

```text
local oracle:              J_h P_T(h)
average projected lens:    E_h[J_h P_T(h)]
local projection evaluation: E_h[J_h] P_T(h).
```

In general, the last two are not equal because the local Jacobian and local
projector are correlated. Indeed, we find a strong noncommutativity result, with
projected lenses being normal-dominant, and local projections being tangent-dominant.

## Methods

At layer `l` and position `p`, `J_x` is the exact downstream Jacobian from the
source residual to the final residual. By default it uses the same causal
broadcast convention as the Jacobian lens: its output cotangent is summed over
valid current-and-future target positions. Pass `--strict-local` to instead use
the diagonal block `d h_L[p] / d h_l[p]`.

The pushforward estimator starts from a local k-nearest-neighbor PCA basis in
token-embedding space. For positions `q` in a causal lookback window it pushes
those low-rank directions through the network, concatenates

```text
[D h_l[p] / D h_0[q] B_0(q)]_q,
```

and retains the leading left singular vectors at the requested energy
threshold. Forward-mode AD computes only these low-rank pushforwards.

The direct-PCA estimator finds neighboring realized residuals in a fixed
activation bank at layer `l`, performs PCA on their differences from the query
residual, and applies the same energy rule. Use a bank corpus independent of
the evaluation examples.

For the state-dependent mean split, the mean Jacobian is loaded from a fitted
checkpoint and only `P_T(h)`/`P_N(h)` are recomputed on each evaluation state.
This path needs no prompt-local downstream Jacobian and is therefore much
cheaper than the local oracle, especially for the direct-PCA estimator.

## Install

Python 3.10+, PyTorch 2.2+, and a CUDA-capable device are recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

For development:

```bash
pip install -e '.[dev]'
pytest
ruff check .
```

No model weights or pretraining corpus are included. Hugging Face models and
datasets remain subject to their own licenses.

## Input data

A fitting/bank corpus is JSON in either supported form:

```json
{"prompts": ["first pretraining-like passage", "second passage"]}
```

or a top-level string list. Sequences should contain more tokens than
`--skip-first` (16 by default). `data/corpus.example.json` is only a format
example, not a paper-scale training set.

The six synthetic evaluation suites used by the commands below are included
in `data/evaluations/`. Each item contains a prompt and one or more accepted
target strings. Evaluation uses the first token of both the space-prefixed and
unprefixed target spellings.

## Reproduce the fitted projected lenses

The following fits all five matched matrices at each listed layer:
`E[J]`, pushforward `E[J P_T]`/`E[J P_N]`, and direct-PCA
`E[J P_T]`/`E[J P_N]`.

```bash
tangent-lenses fit \
  --model Qwen/Qwen2.5-7B \
  --layers 8,12,16,20,24 \
  --corpus /path/to/pretraining_corpus.json \
  --bank-corpus /path/to/independent_bank_corpus.json \
  --n-train 400 --n-bank 400 \
  --samples-per-prompt 4 --bank-samples-per-prompt 8 \
  --prefix-lookback 4 --k-neighbors 64 --energy-threshold 0.95 \
  --activation-bank results/qwen_activation_bank.pt \
  --output results/qwen_projected_lenses.pt
```

Evaluate the saved family:

```bash
tangent-lenses evaluate \
  --model Qwen/Qwen2.5-7B \
  --layers 8,12,16,20,24 \
  --checkpoint results/qwen_projected_lenses.pt \
  --eval data/evaluations/lens-eval-*.json \
  --limit 30 \
  --output results/qwen_projected_lenses_eval.json
```

Add `--start 30 --limit 30` to evaluate the second 30-example slice rather
than the default first slice. The `mean-split` and `local-oracle` commands
support the same arguments and record original item indices, so resumed runs
remain stable.

The checkpoint has one matrix dictionary per estimator and component. Its
adjacent `.json` file records configuration, sample counts, and decomposition
error without duplicating the matrices.

## Reproduce the state-dependent mean split

This evaluates `E[J] P_T(h)` and `E[J] P_N(h)` for both tangent estimators,
using `E[J]` from the fitted checkpoint and the local projector at each held-out
state:

```bash
tangent-lenses mean-split \
  --model Qwen/Qwen2.5-7B \
  --layers 8,12,16,20,24 \
  --checkpoint results/qwen_projected_lenses.pt \
  --bank-corpus /path/to/independent_bank_corpus.json \
  --n-bank 400 --bank-samples-per-prompt 8 \
  --prefix-lookback 4 --k-neighbors 64 --energy-threshold 0.95 \
  --activation-bank results/qwen_activation_bank.pt \
  --eval data/evaluations/lens-eval-*.json \
  --limit 30 \
  --output results/qwen_mean_jacobian_split.json
```

The result labels are deliberately explicit: `mean_jacobian`,
`pushforward_mean_split_tangent`, `pushforward_mean_split_normal`,
`direct_pca_mean_split_tangent`, and `direct_pca_mean_split_normal`. Each
example includes an audit of `E[J] = E[J]P_T(h) + E[J]P_N(h)`.

## Reproduce the local oracle

```bash
tangent-lenses local-oracle \
  --model Qwen/Qwen2.5-7B \
  --layers 8,12,16,20,24 \
  --bank-corpus /path/to/independent_bank_corpus.json \
  --n-bank 400 --bank-samples-per-prompt 8 \
  --prefix-lookback 4 --k-neighbors 64 --energy-threshold 0.95 \
  --activation-bank results/qwen_activation_bank.pt \
  --eval data/evaluations/lens-eval-*.json \
  --limit 30 \
  --output results/qwen_local_oracle.json
```

The local command writes after every example and resumes by `(suite,
item_index)`. Result JSON contains best target rank across the requested
workspace layers, mean workspace KL, residual cosine, estimated ranks, and the
per-example decomposition audit. It intentionally does not serialize the
large prompt-local matrices.

`scripts/run_qwen2.5_7b.sh` and `scripts/run_olmo2_7b.sh` contain the complete
paper-scale command sequences. Set `CORPUS` and `BANK_CORPUS` before running.

## Reproducibility notes

- Position selection is deterministic and evenly spaced over valid tokens.
- Direct-PCA neighbors with zero distance from the query are excluded.
- `P_N` is always the exact ambient complement `I - P_T`; rank-matched normal
  controls are deliberately outside the scope of the reported experiments.
- The activation bank can be cached with `--activation-bank` and is reused by
  all three experiment families.
- Fitting float32 matrices avoids accumulation error. Checkpoints default to
  float16 storage and load into float32.
- Fused attention kernels generally do not support forward-mode AD. The
  default `--forward-ad-attention math` selects PyTorch's math SDPA backend for
  the pushforward computation.

## Repository layout

```text
src/tangent_lenses/
  geometry.py       tangent estimators and projectors
  jacobians.py      exact reverse-mode J and low-rank forward-mode pushforwards
  study.py          local oracle, matched averages, and state-dependent mean split
  lens.py           checkpoint and transport API
  evaluation.py     target rank, KL, and cosine metrics
  hf.py             Hugging Face decoder adapter
  cli.py            fit, evaluate, mean-split, and local-oracle commands
data/evaluations/   synthetic evaluation prompts
scripts/            paper-scale command lines
tests/              geometry invariants and tiny-model end-to-end tests
```

## License

Apache 2.0. See `LICENSE`.


## Citation

If you use this repository, please consider citing us as
```
@article{mackey2026tangent,
	title={How to track your tangent space: the geometry of the linear representation hypothesis},
	author={Mackey, Wyatt and Rinderspacher, Berend and Franaszczuk, Piotr and Boothe, David L.},
	year={2026}
}
```

## Acknowledgements
Our implementation was based on [https://github.com/anthropics/jacobian-lens](https://github.com/anthropics/jacobian-lens).
