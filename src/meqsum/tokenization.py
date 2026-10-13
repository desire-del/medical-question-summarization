
from pathlib import Path

import yaml
from datasets import load_from_disk
from transformers import AutoTokenizer

from meqsum.preprocess import SOURCE_PREFIX


def tokenize_batch(
    examples, tokenizer, max_source_length, max_target_length
):
    sources = [
        SOURCE_PREFIX + text
        for text in examples["source"]
    ]

    inputs = tokenizer(
        sources,
        max_length=max_source_length,
        truncation=True,
    )

    targets = tokenizer(
        text_target=list(examples["target"]),
        max_length=max_target_length,
        truncation=True,
    )

    inputs["labels"] = targets["input_ids"]
    return inputs


def prepare_tokenized_dataset(config):
    tokenizer = AutoTokenizer.from_pretrained(
        config["model_name"]
    )

    dataset = load_from_disk(config["data_path"])

    tokenized = dataset.map(
        lambda batch: tokenize_batch(
            batch,
            tokenizer,
            config["max_source_length"],
            config["max_target_length"],
        ),
        batched=True,
        remove_columns=dataset["train"].column_names,
    )

    return tokenized, tokenizer


def main():
    config = yaml.safe_load(
        Path("configs/baseline.yaml").read_text()
    )

    dataset, tokenizer = prepare_tokenized_dataset(config)

    print(dataset)
    sample = dataset["train"][0]

    print("\nInput:")
    print(tokenizer.decode(
        sample["input_ids"], skip_special_tokens=True
    ))

    print("\nTarget:")
    print(tokenizer.decode(
        sample["labels"], skip_special_tokens=True
    ))

    assert len(sample["input_ids"]) <= 384
    assert len(sample["labels"]) <= 64


if __name__ == "__main__":
    main()
