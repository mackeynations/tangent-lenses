#!/usr/bin/env bash
set -euo pipefail

: "${CORPUS:?Set CORPUS to the fitting-corpus JSON path}"
: "${BANK_CORPUS:?Set BANK_CORPUS to the independent activation-bank corpus JSON path}"

MODEL="allenai/OLMo-2-1124-7B"
LAYERS="8,12,16,20,24,28"
OUTDIR="${OUTDIR:-results/olmo2-7b}"
mkdir -p "$OUTDIR"

tangent-lenses fit \
  --model "$MODEL" --layers "$LAYERS" \
  --corpus "$CORPUS" --bank-corpus "$BANK_CORPUS" \
  --n-train 400 --n-bank 400 \
  --samples-per-prompt 4 --bank-samples-per-prompt 8 \
  --prefix-lookback 4 --k-neighbors 64 --energy-threshold 0.95 \
  --activation-bank "$OUTDIR/activation_bank.pt" \
  --output "$OUTDIR/projected_lenses.pt"

tangent-lenses evaluate \
  --model "$MODEL" --layers "$LAYERS" \
  --checkpoint "$OUTDIR/projected_lenses.pt" \
  --eval data/evaluations/lens-eval-*.json --limit 30 \
  --output "$OUTDIR/projected_lenses_eval.json"

tangent-lenses mean-split \
  --model "$MODEL" --layers "$LAYERS" \
  --checkpoint "$OUTDIR/projected_lenses.pt" \
  --bank-corpus "$BANK_CORPUS" --n-bank 400 --bank-samples-per-prompt 8 \
  --prefix-lookback 4 --k-neighbors 64 --energy-threshold 0.95 \
  --activation-bank "$OUTDIR/activation_bank.pt" \
  --eval data/evaluations/lens-eval-*.json --limit 30 \
  --output "$OUTDIR/mean_jacobian_split.json"

tangent-lenses local-oracle \
  --model "$MODEL" --layers "$LAYERS" \
  --bank-corpus "$BANK_CORPUS" --n-bank 400 --bank-samples-per-prompt 8 \
  --prefix-lookback 4 --k-neighbors 64 --energy-threshold 0.95 \
  --activation-bank "$OUTDIR/activation_bank.pt" \
  --eval data/evaluations/lens-eval-*.json --limit 30 \
  --output "$OUTDIR/local_oracle.json"
