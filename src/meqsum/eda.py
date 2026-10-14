
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from datasets import load_from_disk
from transformers import AutoTokenizer

from meqsum.preprocess import SOURCE_PREFIX



MODEL_ID = "google/flan-t5-large"
DATA_PATH = "data/processed/meqsum"
REPORT_DIR = Path("reports")


def summarize_lengths(lengths):
    return {
        "min": int(np.min(lengths)),
        "mean": round(float(np.mean(lengths)), 2),
        "median": float(np.median(lengths)),
        "p90": float(np.percentile(lengths, 90)),
        "p95": float(np.percentile(lengths, 95)),
        "p99": float(np.percentile(lengths, 99)),
        "max": int(np.max(lengths)),
    }


def print_coverage(lengths, candidates):
    print("\nMax length | Coverage | Truncated")
    for limit in candidates:
        coverage = np.mean(lengths <= limit)
        print(
            f"{limit:>10} | "
            f"{coverage:>7.1%} | "
            f"{1 - coverage:>9.1%}"
        )


def main():
    REPORT_DIR.mkdir(exist_ok=True)
    (REPORT_DIR / "figures").mkdir(exist_ok=True)

    # Use train only to avoid test-set leakage
    dataset = load_from_disk(DATA_PATH)["train"]
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    sources = [
        SOURCE_PREFIX + text
        for text in dataset["source"]
    ]
    targets = list(dataset["target"])

    source_ids = tokenizer(
        sources, truncation=False
    )["input_ids"]

    target_ids = tokenizer(
        text_target=targets, truncation=False
    )["input_ids"]

    source_lengths = np.array(
        [len(ids) for ids in source_ids]
    )
    target_lengths = np.array(
        [len(ids) for ids in target_ids]
    )

    stats = {
        "model": MODEL_ID,
        "prompt": SOURCE_PREFIX,
        "num_examples": len(dataset),
        "source": summarize_lengths(source_lengths),
        "target": summarize_lengths(target_lengths),
    }

    print(json.dumps(stats, indent=2))

    print("\nSOURCE LENGTH COVERAGE")
    print_coverage(
        source_lengths, [128, 256, 384, 512]
    )

    print("\nTARGET LENGTH COVERAGE")
    print_coverage(
        target_lengths, [32, 64, 96, 128]
    )

    # Plot token length distributions
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax, lengths, label in zip(
        axes,
        [source_lengths, target_lengths],
        ["Source", "Target"],
    ):
        ax.hist(lengths, bins=30)
        ax.axvline(
            np.percentile(lengths, 95),
            linestyle="--",
            label="95th percentile",
        )
        ax.set_title(f"{label} token lengths")
        ax.set_xlabel("Number of tokens")
        ax.set_ylabel("Frequency")
        ax.legend()

    fig.tight_layout()
    fig.savefig(
        REPORT_DIR / "figures/token_lengths.png",
        dpi=150,
    )
    plt.close(fig)

    with open(REPORT_DIR / "eda.json", "w") as f:
        json.dump(stats, f, indent=2)


if __name__ == "__main__":
    main()
