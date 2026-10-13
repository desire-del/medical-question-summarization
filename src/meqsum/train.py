
import csv
import json
import platform
import time
from pathlib import Path

import evaluate
import numpy as np
import peft
import torch
import transformers
import yaml

from datasets import load_from_disk
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)

from meqsum.tokenization import prepare_tokenized_dataset


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, default=str),
        encoding="utf-8",
    )


def build_compute_metrics(tokenizer):
    rouge = evaluate.load("rouge")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred

        if isinstance(predictions, tuple):
            predictions = predictions[0]

        predictions = np.where(
            predictions != -100,
            predictions,
            tokenizer.pad_token_id,
        )
        labels = np.where(
            labels != -100,
            labels,
            tokenizer.pad_token_id,
        )

        decoded_preds = tokenizer.batch_decode(
            predictions, skip_special_tokens=True
        )
        decoded_labels = tokenizer.batch_decode(
            labels, skip_special_tokens=True
        )

        scores = rouge.compute(
            predictions=decoded_preds,
            references=decoded_labels,
            rouge_types=["rouge1", "rouge2", "rougeL"],
            use_stemmer=True,
        )

        return {key: round(value, 4)
                for key, value in scores.items()}

    return compute_metrics


def main():
    config = yaml.safe_load(
        Path("configs/lora.yaml").read_text()
    )

    assert torch.cuda.is_available(), "GPU required"

    train_cfg = config["training"]
    gen_cfg = config["generation"]
    lora_cfg = config["lora"]

    set_seed(train_cfg["seed"])

    output_dir = Path(config["output_dir"])
    experiment_dir = output_dir.parent
    experiment_dir.mkdir(parents=True, exist_ok=True)


    #  Dataset
    dataset, tokenizer = prepare_tokenized_dataset(config)

    raw_validation = load_from_disk(
        config["data_path"]
    )["validation"]

    #  Model and LoRA
    model = AutoModelForSeq2SeqLM.from_pretrained(
        config["model_name"]
    )

    # Match zero-shot baseline generation settings
    model.generation_config.num_beams = gen_cfg["num_beams"]
    model.generation_config.max_new_tokens = gen_cfg["max_new_tokens"]
    model.generation_config.do_sample = False

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        target_modules=lora_cfg["target_modules"],
    )

    model = get_peft_model(model, peft_config)

    trainable_params = sum(
        p.numel() for p in model.parameters()
        if p.requires_grad
    )
    total_params = sum(
        p.numel() for p in model.parameters()
    )

    model.print_trainable_parameters()

    #  Training setup
    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=-100,
    )

    # Prefer numerical stability for FLAN-T5
    bf16 = torch.cuda.is_bf16_supported()

    args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=train_cfg["epochs"],
        learning_rate=train_cfg["learning_rate"],
        per_device_train_batch_size=train_cfg["batch_size"],
        per_device_eval_batch_size=train_cfg["eval_batch_size"],
        gradient_accumulation_steps=train_cfg[
            "gradient_accumulation_steps"
        ],

        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=10,

        predict_with_generate=True,
        generation_num_beams=gen_cfg["num_beams"],

        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=2,

        bf16=bf16,
        fp16=False,
        seed=train_cfg["seed"],
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
        data_collator=collator,
        compute_metrics=build_compute_metrics(tokenizer),
    )


    # Training
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()

    train_result = trainer.train()

    torch.cuda.synchronize()
    training_time = time.perf_counter() - start

    peak_allocated_gb = (
        torch.cuda.max_memory_allocated() / 1024**3
    )
    peak_reserved_gb = (
        torch.cuda.max_memory_reserved() / 1024**3
    )

    # Save best adapter
    best_dir = experiment_dir / "best_adapter"

    trainer.save_model(str(best_dir))
    tokenizer.save_pretrained(str(best_dir))

    # Save training history
    save_json(
        experiment_dir / "training_history.json",
        trainer.state.log_history,
    )

    
    #  Best-model evaluation
    prediction_output = trainer.predict(
        dataset["validation"],
        max_new_tokens=gen_cfg["max_new_tokens"],
        num_beams=gen_cfg["num_beams"],
    )

    prediction_ids = np.where(
        prediction_output.predictions != -100,
        prediction_output.predictions,
        tokenizer.pad_token_id,
    )

    predictions = tokenizer.batch_decode(
        prediction_ids,
        skip_special_tokens=True,
    )

    with (experiment_dir / "predictions.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "id", "source", "reference", "prediction"
            ],
        )
        writer.writeheader()

        for example, prediction in zip(
            raw_validation, predictions
        ):
            writer.writerow({
                "id": example["id"],
                "source": example["source"],
                "reference": example["target"],
                "prediction": prediction,
            })

    save_json(
        experiment_dir / "evaluation.json",
        prediction_output.metrics,
    )


    # Experiment metadata
    metadata = {
        "experiment": "E1_LoRA",
        "config": config,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "peft": peft.__version__,
            "gpu": torch.cuda.get_device_name(0),
            "cuda": torch.version.cuda,
            "precision": "bf16" if bf16 else "fp32",
        },
        "dataset": {
            "train_samples": len(dataset["train"]),
            "validation_samples": len(dataset["validation"]),
        },
        "parameters": {
            "total": total_params,
            "trainable": trainable_params,
            "trainable_percentage": round(
                100 * trainable_params / total_params, 4
            ),
        },
        "performance": {
            "training_time_seconds": round(
                training_time, 2
            ),
            "peak_gpu_allocated_gb": round(
                peak_allocated_gb, 3
            ),
            "peak_gpu_reserved_gb": round(
                peak_reserved_gb, 3
            ),
            "global_steps": trainer.state.global_step,
        },
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "best_eval_loss": trainer.state.best_metric,
        "training_metrics": train_result.metrics,
        "evaluation_metrics": prediction_output.metrics,
    }

    save_json(
        experiment_dir / "metadata.json",
        metadata,
    )

    print("\n=== EXPERIMENT SUMMARY ===")
    print(json.dumps({
        "best_eval_loss": trainer.state.best_metric,
        "rouge": prediction_output.metrics,
        "training_time_seconds": round(training_time, 2),
        "peak_gpu_allocated_gb": round(
            peak_allocated_gb, 3
        ),
        "trainable_percentage": metadata[
            "parameters"
        ]["trainable_percentage"],
    }, indent=2, default=str))

    print(f"\nResults saved to: {experiment_dir}")


if __name__ == "__main__":
    main()
