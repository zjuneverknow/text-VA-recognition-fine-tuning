from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from transformers import (
    AutoTokenizer,
    EvalPrediction,
    ModernBertForSequenceClassification,
    Trainer,
    TrainingArguments,
)

from src.dataset_loader import DataConfig, load_datasets, tokenize_dataset
from src.eval_metrics import compute_regression_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune ModernBERT for valence-arousal regression."
    )
    parser.add_argument("--model-id", default="answerdotai/ModernBERT-base")
    parser.add_argument("--train-file")
    parser.add_argument("--validation-file")
    parser.add_argument("--dataset-name")
    parser.add_argument("--dataset-config-name")
    parser.add_argument("--dataset-train-split", default="train")
    parser.add_argument("--dataset-validation-split")
    parser.add_argument("--validation-size", type=float, default=0.1)
    parser.add_argument("--output-dir", default="output/modernbert-va-regression")
    parser.add_argument("--cache-dir", default="data/hf_cache")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--valence-column", default="v_score")
    parser.add_argument("--arousal-column", default="a_score")
    parser.add_argument("--label-min", type=float, default=1.0)
    parser.add_argument("--label-max", type=float, default=9.0)
    parser.add_argument(
        "--prompt-template",
        default="Analyze the valence and arousal of the following text: {text}",
    )
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--num-train-epochs", type=float, default=5.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--per-device-train-batch-size", type=int, default=16)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=32)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--disable-bf16", action="store_true")
    args = parser.parse_args()

    has_files = bool(args.train_file and args.validation_file)
    has_dataset = bool(args.dataset_name)
    if has_files == has_dataset:
        parser.error(
            "Provide either --dataset-name or both --train-file and --validation-file."
        )
    if not 0.0 < args.validation_size < 1.0:
        parser.error("--validation-size must be between 0 and 1.")
    return args


def load_tokenizer(model_id: str, cache_dir: str):
    return AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)


def load_model(args: argparse.Namespace):
    return ModernBertForSequenceClassification.from_pretrained(
        args.model_id,
        num_labels=2,
        problem_type="regression",
        cache_dir=args.cache_dir,
    )


def compute_metrics(eval_prediction: EvalPrediction) -> dict[str, float]:
    predictions = eval_prediction.predictions
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    labels = eval_prediction.label_ids
    predictions = np.asarray(predictions, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.float32)
    return compute_regression_metrics(predictions, labels)


def build_training_arguments(args: argparse.Namespace) -> TrainingArguments:
    return TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_train_epochs=args.num_train_epochs,
        warmup_ratio=args.warmup_ratio,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        bf16=not args.disable_bf16,
        fp16=False,
        eval_strategy="steps",
        save_strategy="steps",
        save_total_limit=2,
        save_only_model=True,
        logging_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="ccc",
        greater_is_better=True,
        seed=args.seed,
        report_to="none",
    )


def main() -> None:
    args = parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.cache_dir).mkdir(parents=True, exist_ok=True)

    data_config = DataConfig(
        train_file=args.train_file,
        validation_file=args.validation_file,
        dataset_name=args.dataset_name,
        dataset_config_name=args.dataset_config_name,
        dataset_train_split=args.dataset_train_split,
        dataset_validation_split=args.dataset_validation_split,
        validation_size=args.validation_size,
        dataset_cache_dir=args.cache_dir,
        text_column=args.text_column,
        valence_column=args.valence_column,
        arousal_column=args.arousal_column,
        label_min=args.label_min,
        label_max=args.label_max,
        prompt_template=args.prompt_template,
        seed=args.seed,
    )

    tokenizer = load_tokenizer(args.model_id, args.cache_dir)
    train_dataset, validation_dataset = load_datasets(data_config)
    train_dataset = tokenize_dataset(train_dataset, tokenizer, args.max_length)
    validation_dataset = tokenize_dataset(validation_dataset, tokenizer, args.max_length)

    model = load_model(args)

    trainer = Trainer(
        model=model,
        args=build_training_arguments(args),
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    metrics = trainer.evaluate()
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)
    trainer.save_state()


if __name__ == "__main__":
    main()
