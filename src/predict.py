from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
from transformers import AutoTokenizer, ModernBertForSequenceClassification

DEFAULT_PROMPT_TEMPLATE = "Analyze the valence and arousal of the following text: {text}"


@dataclass(frozen=True, slots=True)
class VAResult:
    valence: float
    arousal: float

    def as_dict(self) -> dict[str, float]:
        return {
            "valence": self.valence,
            "arousal": self.arousal,
        }


class VAPredictor:
    def __init__(
        self,
        model_dir: str,
        max_length: int = 256,
        prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
        device: str | None = None,
        clamp_output: bool = True,
    ) -> None:
        self.model_dir = str(Path(model_dir).resolve())
        self.max_length = max_length
        self.prompt_template = prompt_template
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.clamp_output = clamp_output

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self.model = ModernBertForSequenceClassification.from_pretrained(self.model_dir)
        self.model.to(self.device)
        self.model.eval()

    def _build_prompt(self, text: str) -> str:
        return self.prompt_template.format(text=text.strip())

    def _normalize_scores(self, scores: Sequence[float]) -> VAResult:
        valence = float(scores[0])
        arousal = float(scores[1])
        if self.clamp_output:
            valence = min(1.0, max(0.0, valence))
            arousal = min(1.0, max(0.0, arousal))
        return VAResult(valence=valence, arousal=arousal)

    @torch.inference_mode()
    def predict(self, text: str) -> VAResult:
        return self.predict_batch([text])[0]

    @torch.inference_mode()
    def predict_batch(self, texts: Sequence[str]) -> list[VAResult]:
        if not texts:
            return []

        prompts = [self._build_prompt(text) for text in texts]
        inputs = self.tokenizer(
            prompts,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        outputs = self.model(**inputs)
        logits = outputs.logits.detach().float().cpu().tolist()
        return [self._normalize_scores(scores) for scores in logits]


_predictor_cache: dict[tuple[str, int, str, str | None, bool], VAPredictor] = {}


def load_predictor(
    model_dir: str,
    max_length: int = 256,
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
    device: str | None = None,
    clamp_output: bool = True,
) -> VAPredictor:
    resolved_model_dir = str(Path(model_dir).resolve())
    cache_key = (resolved_model_dir, max_length, prompt_template, device, clamp_output)
    predictor = _predictor_cache.get(cache_key)
    if predictor is None:
        predictor = VAPredictor(
            model_dir=resolved_model_dir,
            max_length=max_length,
            prompt_template=prompt_template,
            device=device,
            clamp_output=clamp_output,
        )
        _predictor_cache[cache_key] = predictor
    return predictor


def predict_va(
    text: str,
    model_dir: str,
    max_length: int = 256,
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
    device: str | None = None,
    clamp_output: bool = True,
) -> dict[str, float]:
    predictor = load_predictor(
        model_dir=model_dir,
        max_length=max_length,
        prompt_template=prompt_template,
        device=device,
        clamp_output=clamp_output,
    )
    return predictor.predict(text).as_dict()


def predict_va_batch(
    texts: Sequence[str],
    model_dir: str,
    max_length: int = 256,
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
    device: str | None = None,
    clamp_output: bool = True,
) -> list[dict[str, float]]:
    predictor = load_predictor(
        model_dir=model_dir,
        max_length=max_length,
        prompt_template=prompt_template,
        device=device,
        clamp_output=clamp_output,
    )
    return [result.as_dict() for result in predictor.predict_batch(texts)]
