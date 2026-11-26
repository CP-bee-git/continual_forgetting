import csv
import hashlib
from pathlib import Path

from scipy.io import loadmat, whosmat


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "数据集/cwru/manifest.csv"
RAW_DIR = ROOT / "数据集/cwru/raw/12k_DE"
INVENTORY = ROOT / "数据集/cwru/raw_inventory.csv"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    with MANIFEST.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))

    inventory = []
    errors = []
    for row in rows:
        path = RAW_DIR / row["filename"]
        if not path.is_file():
            errors.append(f"missing file: {path}")
            continue

        expected_variable = f"X{int(row['file_id']):03d}_DE_time"
        variables = [name for name, _, _ in whosmat(path) if name == expected_variable]
        if len(variables) != 1:
            errors.append(f"{row['filename']}: expected one DE_time variable, found {variables}")
            continue

        signal = loadmat(path, variable_names=variables)[variables[0]]
        inventory.append(
            {
                "filename": row["filename"],
                "task": row["task"],
                "load_hp": row["load_hp"],
                "class_id": row["class_id"],
                "label": row["label"],
                "variable": variables[0],
                "samples": signal.size,
                "sample_rate_hz": row["sample_rate_hz"],
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )

    if inventory:
        with INVENTORY.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=inventory[0].keys())
            writer.writeheader()
            writer.writerows(inventory)

    for row in inventory:
        print(
            f"{row['filename']:>7} {row['task']:>6} {row['label']:<12} "
            f"{row['variable']:<20} {row['samples']:>8} samples"
        )
    if errors:
        raise SystemExit("\n".join(errors))
    if len(inventory) != len(rows):
        raise SystemExit(f"expected {len(rows)} files, inspected {len(inventory)}")
    print(f"\nInspected {len(inventory)} files; wrote {INVENTORY}")


if __name__ == "__main__":
    main()
