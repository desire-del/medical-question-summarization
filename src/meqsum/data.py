
import json
from pathlib import Path

from datasets import DatasetDict, load_dataset

from meqsum.preprocess import prepare_example

SEED = 42
DATASET_ID = "albertvillanova/meqsum"
OUTPUT_DIR = Path("data/processed/meqsum")


def load_and_prepare():
    dataset = load_dataset(DATASET_ID, split="train")

    dataset = dataset.map(
        prepare_example,
        remove_columns=dataset.column_names,
    )

    dataset = dataset.filter(
        lambda x: bool(x["source"] and x["target"])
    )

    # Remove duplicate questions before splitting
    seen = set()
    indices = []

    for i, example in enumerate(dataset):
        key = example["source"].casefold()
        if key not in seen:
            seen.add(key)
            indices.append(i)

    return dataset.select(indices)


def create_splits(dataset):
    split = dataset.train_test_split(
        test_size=0.2, seed=SEED
    )

    holdout = split["test"].train_test_split(
        test_size=0.5, seed=SEED
    )

    return DatasetDict({
        "train": split["train"],
        "validation": holdout["train"],
        "test": holdout["test"],
    })


def main():
    dataset = load_and_prepare()
    splits = create_splits(dataset)

    if OUTPUT_DIR.exists():
        raise FileExistsError(
            f"{OUTPUT_DIR} already exists. "
            "Remove it explicitly before regenerating."
        )

    splits.save_to_disk(str(OUTPUT_DIR))

    # Keep exact split membership for reproducibility
    manifest = {
        "dataset": DATASET_ID,
        "seed": SEED,
        "splits": {
            name: list(data["id"])
            for name, data in splits.items()
        },
    }

    Path("reports").mkdir(exist_ok=True)
    Path("reports/split_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    for name, data in splits.items():
        print(f"{name}: {len(data)} examples")


if __name__ == "__main__":
    main()
