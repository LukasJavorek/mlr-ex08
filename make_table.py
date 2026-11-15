"""Turn ablation_results.csv into the Markdown table for the report.

    .venv/Scripts/python.exe make_table.py            # prints it and writes ablation_report.md

No model training here, only formatting: one line per run plus, for every variant that was
repeated with seeds 0/1/2, a mean +/- std row for val_acc and camera_val_acc.
"""

import csv
import statistics
from pathlib import Path

HERE = Path(__file__).parent
CSV = HERE / "ablation_results.csv"
OUT = HERE / "ablation_report.md"


def base_name(name):
    return name.split("-s")[0]          # "V9-s2" -> "V9"


def fmt(rows, key):
    values = [r[key] for r in rows]
    if len(values) == 1:
        return f"{values[0]:.4f}"
    mean = statistics.mean(values)
    std = statistics.stdev(values)      # sample std over the seeds
    return f"{mean:.3f} +/- {std:.3f}"


def main():
    with open(CSV, newline="", encoding="utf-8") as f:
        rows = [{**r, "train_acc": float(r["train_acc"]), "val_acc": float(r["val_acc"]),
                 "camera_val_acc": float(r["camera_val_acc"])} for r in csv.DictReader(f)]

    order, groups = [], {}
    for r in rows:                                   # keep the V0, V1, ... V9 order
        base = base_name(r["variant_name"])
        if base not in groups:
            groups[base] = []
            order.append(base)
        groups[base].append(r)

    lines = ["| variant | what changed | epochs | seed | train_acc | val_acc | camera_val_acc |",
             "|---|---|---|---|---|---|---|"]
    for base in order:
        for r in sorted(groups[base], key=lambda r: int(r["seed"])):
            lines.append(f"| {r['variant_name']} | {r['what_changed']} | {r['epochs']} | "
                         f"{r['seed']} | {r['train_acc']:.3f} | {r['val_acc']:.3f} | "
                         f"{r['camera_val_acc']:.3f} |")
        if len(groups[base]) > 1:
            lines.append(f"| **{base} mean±std** | seeds "
                         f"{', '.join(sorted(r['seed'] for r in groups[base]))} |  | 0,1,2 |  | "
                         f"**{fmt(groups[base], 'val_acc')}** | **{fmt(groups[base], 'camera_val_acc')}** |")

    hist = [r for r in rows if r.get("hist_val_acc")]
    note = ("\n`hist_val_acc` and variant V10 are not available: no `archive.zip` (last year's "
            "train/val split) exists in this workspace, so no model was trained or evaluated on "
            "combined data."
            if not hist else "")
    text = "\n".join(lines) + ("\n" + note + "\n" if note else "\n")
    print(text)
    OUT.write_text("# Ablation results\n\n" + text, encoding="utf-8")
    print("written to", OUT)


if __name__ == "__main__":
    main()
