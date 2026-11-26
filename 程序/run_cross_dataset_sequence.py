import argparse
import copy
import csv
import json
import random
import time
from itertools import islice
from pathlib import Path

import numpy as np
import torch
from scipy.io import loadmat
from scipy.signal import resample_poly
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from wdcnn import WDCNN


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "程序/模型与日志/current_ABC_ABCD"
WINDOW = 2048
STRIDE = 512
TARGET_RATE = 12_000
CLASS_NAMES = ["normal", "inner_race", "outer_race"]
TASKS = ["A_CWRU", "B_JNU", "C_UESTC", "D_Ottawa"]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def source_specs() -> dict[str, list[dict]]:
    cwru = ROOT / "数据集/cwru/raw/12k_DE"
    jnu = ROOT / "数据集/external/jnu/raw"
    uestc = ROOT / "数据集/external/uestc/raw"
    ottawa = ROOT / "数据集/ottawa/raw/extracted"
    sources = {
        "A_CWRU": [
            {"path": cwru / "97.mat", "variable": "X097_DE_time", "label": "normal", "rate": 12_000},
            {"path": cwru / "105.mat", "variable": "X105_DE_time", "label": "inner_race", "rate": 12_000},
            {"path": cwru / "130.mat", "variable": "X130_DE_time", "label": "outer_race", "rate": 12_000},
        ],
        "B_JNU": [
            {"path": jnu / "n600_3_2.csv", "variable": None, "label": "normal", "rate": 50_000},
            {"path": jnu / "ib600_2.csv", "variable": None, "label": "inner_race", "rate": 50_000},
            {"path": jnu / "ob600_2.csv", "variable": None, "label": "outer_race", "rate": 50_000},
        ],
        "C_UESTC": [
            {"path": uestc / "N_800_1.mat", "variable": "Data", "label": "normal", "rate": 20_000},
            {"path": uestc / "I_800_1.mat", "variable": "Data", "label": "inner_race", "rate": 20_000},
            {"path": uestc / "O_800_1.mat", "variable": "Data", "label": "outer_race", "rate": 20_000},
        ],
    }
    ottawa_classes = {
        "normal": ("1 Data collected from a healthy bearing", "H"),
        "inner_race": ("2 Data collected from a bearing with inner race fault", "I"),
        "outer_race": ("3 Data collected from a bearing with outer race fault", "O"),
    }
    sources["D_Ottawa"] = [
        {
            "path": ottawa / folder / f"{prefix}-{speed}-{trial}.mat",
            "variable": "Channel_1",
            "label": label,
            "rate": 200_000,
            "speed_profile": speed,
            "trial": int(trial),
        }
        for label, (folder, prefix) in ottawa_classes.items()
        for speed in "ABCD"
        for trial in "123"
    ]
    return sources


def read_signal(spec: dict) -> np.ndarray:
    if spec["path"].suffix == ".csv":
        signal = np.loadtxt(spec["path"], delimiter=",", dtype=np.float64)
    else:
        signal = np.asarray(loadmat(spec["path"], variable_names=[spec["variable"]])[spec["variable"]])
    return np.asarray(signal, dtype=np.float64).reshape(-1)


def resample_signal(signal: np.ndarray, spec: dict) -> np.ndarray:
    if spec["rate"] != TARGET_RATE:
        divisor = np.gcd(spec["rate"], TARGET_RATE)
        signal = resample_poly(signal, TARGET_RATE // divisor, spec["rate"] // divisor)
    return signal.astype(np.float32)


def make_windows(signal: np.ndarray) -> np.ndarray:
    if signal.size < WINDOW:
        raise ValueError(f"Signal has only {signal.size} samples")
    windows = np.stack(
        [signal[start : start + WINDOW] for start in range(0, signal.size - WINDOW + 1, STRIDE)]
    ).astype(np.float32)
    mean = windows.mean(axis=1, keepdims=True)
    std = windows.std(axis=1, keepdims=True)
    return (windows - mean) / np.maximum(std, 1e-8)


def load_task(task: str) -> tuple[dict[str, tuple[torch.Tensor, torch.Tensor]], dict]:
    parts = {split: [] for split in ("train", "validation", "test")}
    labels = {split: [] for split in parts}
    records = []
    specs = source_specs()[task]
    for spec in specs:
        if not spec["path"].is_file():
            raise FileNotFoundError(f"Missing dataset file: {spec['path']}")
        signal = read_signal(spec)
        if task == "D_Ottawa":
            split = {1: "train", 2: "validation", 3: "test"}[spec["trial"]]
            ranges = {split: (0, signal.size)}
        else:
            train_end = int(signal.size * 0.70)
            validation_end = int(signal.size * 0.85)
            ranges = {
                "train": (0, train_end),
                "validation": (train_end, validation_end),
                "test": (validation_end, signal.size),
            }
        window_counts = {}
        for split, (start, end) in ranges.items():
            windows = make_windows(resample_signal(signal[start:end], spec))
            parts[split].append(windows)
            labels[split].extend([CLASS_NAMES.index(spec["label"])] * len(windows))
            window_counts[split] = len(windows)
        record = {
            "path": str(spec["path"].relative_to(ROOT)),
            "label": spec["label"],
            "native_rate_hz": spec["rate"],
            "target_rate_hz": TARGET_RATE,
            "native_samples": int(signal.size),
            "native_boundaries": {name: list(bounds) for name, bounds in ranges.items()},
            "windows": window_counts,
        }
        for optional_key in ("speed_profile", "trial"):
            if optional_key in spec:
                record[optional_key] = spec[optional_key]
        records.append(record)

    data = {}
    counts = {}
    for split in parts:
        x = np.concatenate(parts[split])
        y = np.asarray(labels[split], dtype=np.int64)
        data[split] = (torch.from_numpy(x), torch.from_numpy(y))
        counts[split] = {name: int(np.sum(y == index)) for index, name in enumerate(CLASS_NAMES)}
    return data, {"task": task, "records": records, "window": WINDOW, "stride": STRIDE, "counts": counts}


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    confusion = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=np.int64)
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total = 0
    with torch.inference_mode():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs)
            total_loss += criterion(logits, targets).item() * targets.size(0)
            predictions = logits.argmax(dim=1)
            total += targets.size(0)
            for actual, predicted in zip(targets.cpu().tolist(), predictions.cpu().tolist()):
                confusion[actual, predicted] += 1
    recalls = np.divide(
        confusion.diagonal(), confusion.sum(axis=1),
        out=np.zeros(len(CLASS_NAMES), dtype=np.float64), where=confusion.sum(axis=1) != 0,
    )
    return {
        "loss": total_loss / total,
        "accuracy": float(confusion.diagonal().sum() / total),
        "macro_recall": float(recalls.mean()),
        "per_class_recall": dict(zip(CLASS_NAMES, recalls.tolist())),
        "confusion_matrix": confusion.tolist(),
        "n": total,
    }


def loader(data, split: str, batch_size: int, device: torch.device, seed: int | None = None) -> DataLoader:
    return DataLoader(
        TensorDataset(*data[split]), batch_size=batch_size, shuffle=seed is not None,
        drop_last=seed is not None, pin_memory=device.type == "cuda",
        generator=torch.Generator().manual_seed(seed) if seed is not None else None,
    )


def batchnorm_state(model: nn.Module) -> dict[str, torch.Tensor]:
    state = dict(model.named_buffers())
    return {
        name: value.detach().clone() for name, value in state.items()
        if name.rsplit(".", 1)[-1] in ("running_mean", "running_var", "num_batches_tracked")
    }


def set_batchnorm_state(model: nn.Module, state: dict[str, torch.Tensor]) -> None:
    buffers = dict(model.named_buffers())
    with torch.no_grad():
        for name, value in state.items():
            buffers[name].copy_(value)


def evaluate_configs(model, tests, device, profiles):
    shared = batchnorm_state(model)
    result = {}
    a_shared = evaluate(model, tests[TASKS[0]], device)
    parameters = {name: value.detach().clone() for name, value in model.named_parameters()}
    try:
        for task, test in tests.items():
            active_profile = profiles.get(task, shared)
            set_batchnorm_state(model, active_profile)
            result[task] = evaluate(model, test, device)
            result[task]["normalization_profile"] = task if task in profiles else "current_shared"
            if not all(torch.equal(value, dict(model.named_buffers())[name])
                       for name, value in active_profile.items()):
                raise RuntimeError(f"BatchNorm buffers changed during evaluation of {task}")
    finally:
        set_batchnorm_state(model, shared)
    if not all(torch.equal(value, parameters[name]) for name, value in model.named_parameters()):
        raise RuntimeError("Model parameters changed during configuration-only evaluation")
    return result, a_shared


def train_stage(model, train, validation, device, args, seed, stage, task, writer, stream):
    features, labels = train["train"]
    batches = DataLoader(
        TensorDataset(features, labels, torch.arange(len(labels))),
        batch_size=args.batch_size, shuffle=True, drop_last=True,
        generator=torch.Generator().manual_seed(seed), pin_memory=device.type == "cuda",
    )
    if not len(batches):
        raise ValueError(f"No complete batches for {task}")
    budget = args.return_epochs * len(batches) if stage.startswith("relearn_") else args.steps
    remaining = budget
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    seen = torch.zeros(len(labels), dtype=torch.bool)
    dropped = 0
    skipped = 0
    training_start = time.monotonic()
    for pass_number in range(1, (budget + len(batches) - 1) // len(batches) + 1):
        model.train()
        used_steps = min(remaining, len(batches))
        loss_sum = 0.0
        correct = 0
        count = 0
        for inputs, targets, indices in islice(batches, used_steps):
            seen[indices] = True
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * targets.size(0)
            correct += int((logits.argmax(1) == targets).sum())
            count += targets.size(0)
        remaining -= used_steps
        tail = len(labels) - count if used_steps == len(batches) else 0
        budget_skipped = len(labels) - count - tail
        dropped += tail
        skipped += budget_skipped
        metrics = evaluate(model, validation, device)
        writer.writerow({
            "seed": seed, "stage": stage, "task": task, "pass": pass_number,
            "update_steps": used_steps, "cumulative_steps": budget - remaining,
            "windows_seen": count, "tail_dropped": tail, "budget_skipped": budget_skipped,
            "train_loss": loss_sum / count, "train_accuracy": correct / count,
            "validation_loss": metrics["loss"], "validation_accuracy": metrics["accuracy"],
            "validation_macro_recall": metrics["macro_recall"],
            "elapsed_seconds": round(time.monotonic() - training_start, 2),
        })
        stream.flush()
        print(f"seed={seed} {stage} {budget - remaining}/{budget} steps "
              f"train_acc={correct / count:.3f} val_acc={metrics['accuracy']:.3f}", flush=True)
    return {
        "task": task, "train_windows": len(labels), "unique_windows_used": int(seen.sum()),
        "update_steps": budget, "windows_seen": budget * args.batch_size,
        "tail_dropped": dropped, "budget_skipped": skipped,
    }


def write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--return-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seeds", type=int, nargs="+", default=[2026, 2027, 2028])
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.steps < 1 or args.return_epochs < 1 or args.batch_size < 2:
        parser.error("steps and return epochs must be positive, batch size must be at least 2")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("seeds must be distinct")

    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading data on {device}", flush=True)
    datasets, manifests = {}, {}
    for task in TASKS:
        datasets[task], manifests[task] = load_task(task)
        print(f"{task}: {manifests[task]['counts']}", flush=True)
    tests = {task: loader(datasets[task], "test", args.batch_size, device) for task in TASKS}
    validations = {task: loader(datasets[task], "validation", args.batch_size, device) for task in TASKS}
    stages = ["initial_A", "after_B", "after_C", "after_D"]
    write_json(args.output / "split_manifests.json", manifests)
    write_json(args.output / "run_config.json", {
        "experiments": {"ABC": TASKS[:3], "ABCD": TASKS},
        "domain_input_configs": {
            task: {"native_rates_hz": sorted({r["native_rate_hz"] for r in manifest["records"]}),
                   "target_rate_hz": TARGET_RATE, "window": WINDOW, "stride": STRIDE,
                   "standardization": "per-window z-score", "label_mapping": dict(zip(CLASS_NAMES, range(len(CLASS_NAMES)))),
                   "inference_bn": "frozen running buffers captured at own initial training",
                   "adapter": None}
            for task, manifest in manifests.items()
        },
        "shared_model_state": "WDCNN convolution, BatchNorm affine parameters and the three-class classifier weights",
        "training_bn": "shared running buffers evolve with training; domain snapshots only switched at evaluation and before A relearning",
        "relearning_bn": "A starts from its initial frozen running buffers and is evaluated with its relearned running buffers",
        "unseen_domain_bn": "live shared buffers",
        "split": {"ABC": "raw chronological 70/15/15 before resampling/windowing",
                  "D": "whole recordings; trial indices 1/2/3 for train/validation/test"},
        "steps_per_domain": args.steps, "a_relearning_epochs": args.return_epochs,
        "batch_size": args.batch_size, "seeds": args.seeds, "lr": args.lr,
        "optimizer": "Adam reset at each training stage", "checkpoint_selection": "last optimizer step",
        "replay": False, "device": str(device), "torch": str(torch.__version__),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
    })
    fields = ["seed", "stage", "task", "pass", "update_steps", "cumulative_steps",
              "windows_seen", "tail_dropped", "budget_skipped", "train_loss", "train_accuracy",
              "validation_loss", "validation_accuracy", "validation_macro_recall", "elapsed_seconds"]
    all_results = {}
    for seed in args.seeds:
        seed_everything(seed)
        seed_output = args.output / str(seed)
        seed_output.mkdir(exist_ok=True)
        model = WDCNN(num_classes=len(CLASS_NAMES)).to(device)
        scores, a_shared, profiles, budgets = {}, {}, {}, {}
        with (seed_output / "training_log.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()

            def train_and_evaluate(stage, task):
                budgets[stage] = train_stage(model, datasets[task], validations[task], device,
                                             args, seed, stage, task, writer, stream)
                relearning = stage.startswith("relearn_")
                if not relearning:
                    profiles[task] = batchnorm_state(model)
                active_profiles = profiles | {task: batchnorm_state(model)} if relearning else profiles
                scores[stage], a_shared[stage] = evaluate_configs(model, tests, device, active_profiles)
                if relearning:
                    scores[stage][task]["normalization_profile"] = "A_relearned"
                torch.save({"model": model.state_dict(), "domain_bn": active_profiles, "classes": CLASS_NAMES,
                            "stage": stage}, seed_output / f"wdcnn_{stage}.pt")
                accuracies = {name: round(record["accuracy"], 4) for name, record in scores[stage].items()}
                print(f"seed={seed} {stage} tests={accuracies}", flush=True)

            for task, stage in zip(TASKS, stages):
                train_and_evaluate(stage, task)
                if stage in ("after_C", "after_D"):
                    continuation = copy.deepcopy(model.state_dict())
                    set_batchnorm_state(model, profiles[TASKS[0]])
                    try:
                        train_and_evaluate(f"relearn_{'ABC' if stage == 'after_C' else 'ABCD'}", TASKS[0])
                    finally:
                        model.load_state_dict(continuation)
        result = {"scores": scores, "a_with_shared_bn": a_shared, "training_budgets": budgets}
        write_json(seed_output / "stage_metrics.json", result)
        all_results[seed] = result
    write_json(args.output / "stage_metrics.json", all_results)
    print(f"Saved cross-dataset results to {args.output}", flush=True)


if __name__ == "__main__":
    main()
