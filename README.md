# ModernBERT VA Regression

This project fine-tunes `answerdotai/ModernBERT-base` for valence-arousal regression.

## Task

- Input: natural language text
- Output: `[V, A]`
- Range: `V in [0, 1]`, `A in [0, 1]`
- Model: `ModernBertForSequenceClassification`
- Config: `num_labels=2`, `problem_type="regression"`

## Data sources

The training script supports two input modes:

- Local CSV files with `--train-file` and `--validation-file`
- Hugging Face datasets with `--dataset-name`

For Hugging Face datasets that only provide a `train` split, the script automatically creates a validation split with `--validation-size`.

## CSV format

CSV files must contain:

- `text`
- `v_score`
- `a_score`

By default, labels are assumed to be on a `1-9` scale and are linearly normalized to `[0, 1]`.
If your labels are already in `[0, 1]`, use:

```bash
--label-min 0 --label-max 1
```

## Install

```bash
pip install -e .
```

## Train from CSV

```bash
python main.py ^
  --train-file data/train.csv ^
  --validation-file data/valid.csv ^
  --model-id answerdotai/ModernBERT-base ^
  --output-dir output/modernbert-va-regression ^
  --max-length 256 ^
  --per-device-train-batch-size 16 ^
  --learning-rate 2e-5 ^
  --num-train-epochs 5
```

## Train from Hugging Face dataset

Example for `Mavdol/NPC-Valence-Arousal`:

```bash
python main.py ^
  --dataset-name Mavdol/NPC-Valence-Arousal ^
  --text-column text ^
  --valence-column valence ^
  --arousal-column arousal ^
  --label-min -1 ^
  --label-max 1 ^
  --validation-size 0.1 ^
  --model-id answerdotai/ModernBERT-base
```

## Metrics

- MAE
- Pearson r
- CCC

## Notes

- This project uses an encoder-style model instead of a causal LM.
- The default setup is better suited to English sentence-level regression tasks.
- Model and dataset downloads are cached under `data/hf_cache` by default.
