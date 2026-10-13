
import argparse
import csv
import json
from pathlib import Path

import evaluate
import torch
import yaml

from datasets import load_from_disk
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from meqsum.preprocess import SOURCE_PREFIX


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default="outputs/baseline",
    )
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    config = yaml.safe_load(
        Path("configs/baseline.yaml").read_text()
    )

    assert torch.cuda.is_available(), "GPU required"
    device = torch.device("cuda")

    dataset = load_from_disk(
        config["data_path"]
    )["validation"]

    if args.limit is not None:
        assert args.limit > 0
        dataset = dataset.select(
            range(min(args.limit, len(dataset)))
        )

    tokenizer = AutoTokenizer.from_pretrained(
        config["model_name"]
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        config["model_name"]
    ).to(device)
    model.eval()

    predictions = []
    batch_size = config["eval_batch_size"]

    for start in range(0, len(dataset), batch_size):
        batch = dataset[
            start:min(start + batch_size, len(dataset))
        ]

        texts = [
            SOURCE_PREFIX + source
            for source in batch["source"]
        ]

        inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=config["max_source_length"],
        ).to(device)

        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                num_beams=config["num_beams"],
                max_new_tokens=config["max_new_tokens"],
                do_sample=False,
            )

        predictions.extend(
            tokenizer.batch_decode(
                output_ids,
                skip_special_tokens=True,
            )
        )

        print(
            f"Processed {len(predictions)}/{len(dataset)}"
        )

    references = list(dataset["target"])

    rouge = evaluate.load("rouge")
    scores = rouge.compute(
        predictions=predictions,
        references=references,
        rouge_types=["rouge1", "rouge2", "rougeL"],
        use_stemmer=True,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "predictions.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "id", "source", "reference", "prediction"
            ],
        )
        writer.writeheader()

        for i, prediction in enumerate(predictions):
            writer.writerow({
                "id": dataset[i]["id"],
                "source": dataset[i]["source"],
                "reference": references[i],
                "prediction": prediction,
            })

    results = {
        "model": config["model_name"],
        "split": "validation",
        "num_examples": len(dataset),
        "generation": {
            "num_beams": config["num_beams"],
            "max_new_tokens": config["max_new_tokens"],
            "max_source_length": config["max_source_length"],
        },
        "rouge": scores,
    }

    (output_dir / "metrics.json").write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
