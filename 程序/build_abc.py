import csv
import json
from pathlib import Path

from run_cross_dataset_sequence import CLASS_NAMES, STRIDE, TARGET_RATE, WINDOW, source_specs


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "数据集/abc"
OUTPUT = OUTPUT_DIR / "manifest.csv"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    selected = []
    sources = source_specs()
    task_letters = {"A_CWRU": "A", "B_JNU": "B", "C_UESTC": "C", "D_Ottawa": "D"}
    for task in ("A_CWRU", "B_JNU", "C_UESTC", "D_Ottawa"):
        for spec in sources[task]:
            selected.append({
                "abc_task": task_letters[task],
                "dataset": task.split("_", 1)[1],
                "recording": spec["path"].name,
                "source_path": str(spec["path"].relative_to(ROOT)),
                "label": spec["label"],
                "class_id": CLASS_NAMES.index(spec["label"]),
                "native_sample_rate_hz": spec["rate"],
                "target_sample_rate_hz": TARGET_RATE,
                "speed_profile": spec.get("speed_profile", ""),
                "trial": spec.get("trial", ""),
            })

    fields = list(selected[0])
    with OUTPUT.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(selected)

    summary = {
        "sequences": {"ABC": ["A=CWRU", "B=JNU", "C=UESTC"],
                      "ABCD": ["A=CWRU", "B=JNU", "C=UESTC", "D=Ottawa"]},
        "domains": {
            "A": {"dataset": "CWRU", "condition": "12 kHz drive-end, 0 HP"},
            "B": {"dataset": "JNU", "condition": "600 rpm"},
            "C": {"dataset": "UESTC", "condition": "800 rpm"},
            "D": {
                "dataset": "Ottawa v2",
                "condition": "time-varying speed profiles A-D; 3 trials per selected class and profile",
            },
        },
        "shared_labels": CLASS_NAMES,
        "target_sample_rate_hz": TARGET_RATE,
        "window": WINDOW,
        "stride": STRIDE,
        "split": {"ABC": "raw chronological 70/15/15 before resampling and windowing",
                  "D": "whole recordings by trial index 1/2/3 for train/validation/test"},
        "normalization": "per-window z-score",
        "ottawa": {
            "doi": "10.17632/v43hmbwxpm.2",
            "raw_zip": "数据集/ottawa/raw/ottawa_bearing_v2.zip",
            "raw_zip_sha256": "4e5538e71692c83e9f5a4338808983f2a9bb00bfdc4a4e4a9627b616f5794eb1",
            "selected_classes": ["normal", "inner_race", "outer_race"],
            "excluded_classes": ["ball_fault", "combined_fault"],
            "selected_records": 36,
            "signal_channel": "Channel_1 vibration; Channel_2 encoder excluded from WDCNN input",
            "speed_profiles": {"A": "increasing", "B": "decreasing", "C": "increasing then decreasing", "D": "decreasing then increasing"},
        },
        "note": "ABC and ABCD use configuration-only endpoint tests and separate 10-epoch A relearning branches.",
    }
    (OUTPUT_DIR / "dataset_config.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"Wrote {OUTPUT} with {len(selected)} records")


if __name__ == "__main__":
    main()
