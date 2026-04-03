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

## Install

This project does not pin `torch` in `pyproject.toml`.
That is intentional for cloud images that already ship with CUDA-enabled PyTorch.

Recommended flow on prebuilt GPU images:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
uv sync
```

If your environment does not already have a working PyTorch install, install the correct PyTorch build first, then run:

```bash
uv sync
```

## Train from Hugging Face dataset

```bash
python main.py --dataset-name Mavdol/NPC-Valence-Arousal --text-column text --valence-column valence --arousal-column arousal --label-min -1 --label-max 1 --validation-size 0.1 --model-id answerdotai/ModernBERT-base --output-dir output/modernbert-va-regression --cache-dir data/hf_cache --max-length 256 --per-device-train-batch-size 16 --per-device-eval-batch-size 32 --learning-rate 2e-5 --num-train-epochs 3
```

## Inference

Use the trained model directory directly:

```bash
python -m src.predict_cli --model-dir output/modernbert-va-regression --text "I feel calm and content today."
```

Example output:

```json
{
  "valence": 0.73,
  "arousal": 0.41
}
```

## Python usage

```python
from src.predict import predict_va

result = predict_va(
    text="I feel calm and content today.",
    model_dir="output/modernbert-va-regression",
)
print(result)
```

## Metrics

- MAE
- Pearson r
- CCC

## Notes

- This project uses an encoder-style model instead of a causal LM.
- The default setup is better suited to English sentence-level regression tasks.
- Model and dataset downloads are cached under `data/hf_cache` by default.
