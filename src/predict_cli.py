from __future__ import annotations

import argparse
import json

from src.predict import predict_va


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict valence and arousal for text.")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument(
        "--prompt-template",
        default="Analyze the valence and arousal of the following text: {text}",
    )
    parser.add_argument("--device")
    parser.add_argument("--no-clamp", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = predict_va(
        text=args.text,
        model_dir=args.model_dir,
        max_length=args.max_length,
        prompt_template=args.prompt_template,
        device=args.device,
        clamp_output=not args.no_clamp,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
