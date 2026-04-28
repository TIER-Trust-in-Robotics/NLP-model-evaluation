import argparse
import argcomplete
import json
import h5py
import os
import numpy as np
from pathlib import Path
from datasets import load_dataset, Dataset
from dataclasses import dataclass, asdict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from to_ds import load_iemocap, load_msp_podcast
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    precision_score,
    recall_score,
)


# -------------------------
# NLP models (checkpoints)
# -------------------------


MODEL_REGISTRY = {
    "distilbert": "distilbert/distilbert-base-uncased",
    "tinybert": "huawei-noah/TinyBERT_General_4L_312D",
    "bertmini": "rajjwal1/bert-mini",
    "minilm": "microsoft/MiniLM-L12-H384-uncased",  # alt: microsoft/Multilingual-MiniLM-L12-H384
    "minilm_l6": "nreimers/MiniLM-L6-H384-uncased",  # optimized, faster, but slightly less accurate
}

DATASET_REGISTRY = {
    "dailydialog": "roskoN/dailydialog",
    "goemotion": "google-research-datasets/go_emotions",
    "sst2": "stanfordnlp/sst2",
    "iemocap": "IEMOCAP",
    "msp_podcast": "MSP-PODCAST",
}

OUTPUT_DIR = Path("./benchmark_results")

# -------------------------
# Dataset loaders
# -------------------------


def load_DailyDialog():
    """
    Labels:
    --------------
    0 | no_emotion
    1 | anger
    2 | disgust
    3 | fear
    4 | happiness
    5 | sadness
    6 | surprise

    Datapoints in this datasets are conversations

    of the form ["sentence_1", "sentence_2", ..., "sentence_n"] with n labels [0, 4, ..., 5]

    this function formats so that each sentence or utterance has its own label:
    ["sentence_1"] -> [0]
    ["senteice_2"] -> [4]
    ...
    ["sentence_n"] -> [5]
    """

    def flatten_dialogue(split):
        texts, labels = [], []

        for example in split:
            dialog_col = "dialog" if "dialog" in example else "utterances"
            emotion_col = "emotion" if "emotion" in example else "emotions"

            for utt, emo in zip(example[dialog_col], example[emotion_col]):
                texts.append(utt.strip())
                labels.append(emo)

        return Dataset.from_dict({"text": texts, "label": labels})

    ds = load_dataset("roskoN/dailydialog")
    label_names = [
        "no_emotion",
        "anger",
        "disgust",
        "fear",
        "happiness",
        "sadness",
        "surprise",
    ]

    train = flatten_dialogue(ds["train"])
    test = flatten_dialogue(ds["test"])

    return train, test, 7, label_names


def load_GoEmotion():
    """
    Dataset with 28 emotion labels in four categories:

    -------------------------------------------------------------
    Positive  | Admiration, Amusement, Approval, Caring, Excitement,
              | Gratitude, Joy, Love, Optimism, Pride, Relief
    -------------------------------------------------------------
    Negative  | Anger, Annoyance, Disappointment, Disapproval, Disgust,
              | Embarrassment, Fear, Grief, Nervousness, Remorse, Sadness
    -------------------------------------------------------------
    Ambiguous | Confusion, Curiosity, Desire, Realization, Surprise
    -------------------------------------------------------------
    Neutral   | Neutral
    -------------------------------------------------------------
    """
    ds = load_dataset(DATASET_REGISTRY["goemotion"])

    def flatten(example):
        # onlt taking the first label (the one with coresponding smal)
        example["label"] = example["label"][0]
        return example

    ds = ds.map(flatten, remove_columns=["labels"])

    label_names = [
        "admiration",
        "amusement",
        "anger",
        "annoyance",
        "approval",
        "caring",
        "confusion",
        "curiosity",
        "desire",
        "disappointment",
        "disapproval",
        "disgust",
        "embarrassment",
        "excitement",
        "fear",
        "gratitude",
        "grief",
        "joy",
        "love",
        "nervousness",
        "optimism",
        "pride",
        "realization",
        "relief",
        "remorse",
        "sadness",
        "surprise",
        "neutral",
    ]


def load_sst2():
    """
    Binary Dataset (positive, negative) sentiment
    """

    ds = load_dataset(DATASET_REGISTRY["sst2"])

    train = ds["train"].rename_column("sentence", "text")
    test = ds["validation"].rename_column("sentence", "text")

    return train, test, 2, ["negative", "positive"]


LOADER_REGISTRY = {
    "dailydialog": load_DailyDialog,
    "sst2": load_sst2,
    "iemocap": load_iemocap,
    "msp_podcast": load_msp_podcast,
}

# -------------------------
#  Metrics
# -------------------------


def compute_metrics(eval_pred):
    """
    To be used by finetuning trainer.
    """
    logits, labels = eval_pred
    pred = np.argmax(logits, axis=-1)

    return {
        "accuracy": accuracy_score(labels, pred),
        "weighted_f1": f1_score(
            labels,
            pred,
            average="weighted",
            zero_division=0,
        ),
        "macro_f1": f1_score(
            labels,
            pred,
            average="macro",
            zero_division=0,
        ),
        "macro_precision": precision_score(
            labels,
            pred,
            average="macro",
            zero_division=0,
        ),
        "macro_recall": recall_score(
            labels,
            pred,
            average="macro",
            zero_division=0,
        ),
    }


@dataclass
class BenchmarkResults:
    model_name: str
    dataset_name: str
    device: str
    num_labels: int
    accuracy: float
    weighted_f1: float
    macro_f1: float
    macro_precision: float
    macro_recall: float
    num_train_samples: int
    num_test_samples: int
    per_class_report: str


def tokenize(ds, tokenizer, max_len: int):
    def _tok(batch):
        return tokenizer(
            batch["text"], truncation=True, padding="max_length", max_length=max_len
        )

    ds = ds.map(_tok, batched=True)
    ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    return ds


# -------------------------
# Benchmark
# -------------------------


def run_benchmark(
    model_name: str,
    dataset_name: str,
    device: str,
    finetune: bool = True,  # only turn too False when importing a saved fine tune model
    max_length: int = 128,  # the max length of a text input (empty space is padded)
    epochs: int = 3,
    batch_size: int = 32,
    learning_rate: float = 2e-5,
    warmup_ratio: float = 0.1,  # deprecated, used in warmup_steps
    weight_decay: float = 0.01,
    patience: int = 2,
) -> BenchmarkResults:
    print(f"{'=' * 50}")
    print(f"Model: {model_name} \nDataset: {dataset_name} \nDevice: {device}")
    print(f"{'=' * 50}")

    # --- Dataset ---

    print("Getting dataset...")
    train_ds, eval_ds, num_labels, label_names = LOADER_REGISTRY[dataset_name]()
    print(f"Done. \n Labels: {num_labels} \n Label names: {label_names}")
    print(f"{'=' * 50}")

    # --- Model + Tokenizer ---
    hf_model_id = MODEL_REGISTRY[model_name]

    tokenizer = AutoTokenizer.from_pretrained(hf_model_id)
    model = AutoModelForSequenceClassification.from_pretrained(
        hf_model_id, num_labels=num_labels
    )

    train_ds = tokenize(train_ds, tokenizer, max_length)
    eval_ds = tokenize(eval_ds, tokenizer, max_length)

    # --- Fine Tuning ---

    if finetune:
        run_dir = OUTPUT_DIR / f"{model_name}_{dataset_name}"
        run_dir.mkdir(parents=True, exist_ok=True)

        training_args = TrainingArguments(
            output_dir=str(run_dir),
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size * 2,
            learning_rate=learning_rate,
            # warmup_ratio=warmup_ratio,
            warmup_steps=int(len(train_ds) * warmup_ratio),
            weight_decay=weight_decay,
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="macro_f1",
            greater_is_better=True,
            save_total_limit=1,
            report_to="none",
            # fp16=(device == "cuda"),  # cuda is the only on that supports fp16
            use_cpu=(device == "cpu"),
            dataloader_num_workers=0,
            disable_tqdm=False,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=patience)],
        )

        print("Training...")
        trainer.train()
        print("Finished Training")
    else:
        exit

    # --- Eval ----

    metrics = trainer.evaluate()

    # --- Per-class ---
    preds_out = trainer.predict(eval_ds)
    preds = np.argmax(preds_out.predictions, axis=-1)
    labels = preds_out.label_ids
    report = classification_report(
        labels, preds, zero_division=0, target_names=label_names
    )

    # --- Outfile ---
    result = BenchmarkResults(
        model_name,
        dataset_name=dataset_name,
        device=device,
        num_labels=num_labels,
        accuracy=metrics["eval_accuracy"],
        weighted_f1=metrics["eval_weighted_f1"],
        macro_f1=metrics["eval_macro_f1"],
        macro_precision=metrics["eval_macro_precision"],
        macro_recall=metrics["eval_macro_recall"],
        num_train_samples=len(train_ds),
        num_test_samples=len(eval_ds),
        per_class_report=report,
    )

    _print_result(result)
    save_result(result, run_dir / "results.json")
    return result


def save_result(result: BenchmarkResults, path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    existing = []
    if path.exists():
        with open(path) as f:
            data = json.load(f)
            existing = data if isinstance(data, list) else [data]

    existing.append(asdict(result))

    with open(path, "w") as f:
        json.dump(existing, f, indent=2)

    print(f"Results saved to {path}")


def _print_result(r: BenchmarkResults):
    print(f"\n --- Results: {r.model_name} × {r.dataset_name} ---")
    print(f"  Accuracy         : {r.accuracy:.4f}")
    print(f"  Weighted F1      : {r.weighted_f1:.4f}")
    print(f"  Macro F1         : {r.macro_f1:.4f}")
    print(f"  Macro Precision  : {r.macro_precision:.4f}")
    print(f"  Macro Recall     : {r.macro_recall:.4f}")
    print(f"  Per-class report: {r.per_class_report}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--datasets", nargs="+", choices=list(DATASET_REGISTRY.keys()))
    parser.add_argument("--finetune", default=True)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--output", default=os.path.join(OUTPUT_DIR, "results.json"))
    # argcomplete.autocomplete(parser)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # results = [BenchmarkResults]
    for ds in args.datasets:
        print(f"  Running {args.model} on {ds}...")

        try:
            result = run_benchmark(
                model_name=args.model, dataset_name=ds, device=args.device
            )

        except Exception as e:
            print(f"Error running {args.model} on {args.dataset} with error: {e}")

        save_result(result, args.output)


if __name__ == "__main__":
    main()
