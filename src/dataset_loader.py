from __future__ import annotations

from dataclasses import dataclass

from datasets import Dataset, DatasetDict, load_dataset
import pandas as pd


DEFAULT_PROMPT_TEMPLATE = "Analyze the valence and arousal of the following text: {text}"


@dataclass(slots=True)
class DataConfig:
    train_file: str | None = None
    validation_file: str | None = None
    dataset_name: str | None = None
    dataset_config_name: str | None = None
    dataset_train_split: str = "train"
    dataset_validation_split: str | None = None
    validation_size: float = 0.1
    dataset_cache_dir: str | None = None
    text_column: str = "text"
    valence_column: str = "v_score"
    arousal_column: str = "a_score"
    label_min: float = 1.0
    label_max: float = 9.0
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE
    seed: int = 42


def normalize_label(value: float, label_min: float, label_max: float) -> float:
    if label_max <= label_min:
        raise ValueError("label_max must be greater than label_min.")
    normalized = (float(value) - label_min) / (label_max - label_min)
    return float(min(1.0, max(0.0, normalized)))


def build_prompt(text: str, prompt_template: str) -> str:
    return prompt_template.format(text=str(text).strip())


def _read_csv(path: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"Dataset is empty: {path}")
    return frame


def prepare_dataframe(frame: pd.DataFrame, config: DataConfig) -> pd.DataFrame:
    required_columns = {
        config.text_column,
        config.valence_column,
        config.arousal_column,
    }
    missing = required_columns - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    processed = frame.copy()
    processed[config.text_column] = processed[config.text_column].fillna("").astype(str)
    processed["model_text"] = processed[config.text_column].map(
        lambda text: build_prompt(text, config.prompt_template)
    )
    processed["labels"] = processed.apply(
        lambda row: [
            normalize_label(row[config.valence_column], config.label_min, config.label_max),
            normalize_label(row[config.arousal_column], config.label_min, config.label_max),
        ],
        axis=1,
    )
    return processed[["model_text", "labels"]]


def _prepare_hf_dataset_split(dataset: Dataset, config: DataConfig) -> Dataset:
    frame = dataset.to_pandas()
    prepared = prepare_dataframe(frame, config)
    return Dataset.from_pandas(prepared, preserve_index=False)


def _load_from_hf(config: DataConfig) -> tuple[Dataset, Dataset]:
    dataset_dict = load_dataset(
        path=config.dataset_name,
        name=config.dataset_config_name,
        cache_dir=config.dataset_cache_dir,
    )
    if not isinstance(dataset_dict, DatasetDict):
        raise ValueError("Expected a dataset with named splits.")

    if config.dataset_train_split not in dataset_dict:
        raise ValueError(
            f"Train split '{config.dataset_train_split}' not found. Available: {list(dataset_dict.keys())}"
        )

    if config.dataset_validation_split:
        if config.dataset_validation_split not in dataset_dict:
            raise ValueError(
                f"Validation split '{config.dataset_validation_split}' not found. Available: {list(dataset_dict.keys())}"
            )
        train_dataset = dataset_dict[config.dataset_train_split]
        validation_dataset = dataset_dict[config.dataset_validation_split]
    else:
        split_dataset = dataset_dict[config.dataset_train_split].train_test_split(
            test_size=config.validation_size,
            seed=config.seed,
        )
        train_dataset = split_dataset["train"]
        validation_dataset = split_dataset["test"]

    return (
        _prepare_hf_dataset_split(train_dataset, config),
        _prepare_hf_dataset_split(validation_dataset, config),
    )


def load_datasets(config: DataConfig) -> tuple[Dataset, Dataset]:
    if config.dataset_name:
        return _load_from_hf(config)
    if not config.train_file or not config.validation_file:
        raise ValueError(
            "Provide either dataset_name or both train_file and validation_file."
        )

    train_df = prepare_dataframe(_read_csv(config.train_file), config)
    valid_df = prepare_dataframe(_read_csv(config.validation_file), config)
    return Dataset.from_pandas(train_df, preserve_index=False), Dataset.from_pandas(
        valid_df, preserve_index=False
    )


def tokenize_dataset(
    dataset: Dataset,
    tokenizer,
    max_length: int,
) -> Dataset:
    def _tokenize(batch: dict) -> dict:
        tokenized = tokenizer(
            batch["model_text"],
            max_length=max_length,
            padding="max_length",
            truncation=True,
        )
        tokenized["labels"] = batch["labels"]
        return tokenized

    return dataset.map(
        _tokenize,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing dataset",
    )
