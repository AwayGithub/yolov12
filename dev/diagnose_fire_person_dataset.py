# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Diagnose RGBT-3M fire/person binary labels before training.

This script intentionally avoids starting a YOLO training loop. It verifies the
actual files used by `RGBT-3M-dual-fire-person.yaml`: RGB/IR image pairs,
`labels_fire_person`, label caches, class remapping, empty-label images, and
box geometry. It is meant to answer whether NaN training is caused by corrupt
binary labels or stale cache content.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np

from ultralytics.data.utils import load_dataset_cache_file


ROOT = Path("/data/xwh/dataset/RGBT-3M/RGBT-3M")
ORIG_LABEL_ROOT = ROOT / "labels"
BINARY_LABEL_ROOT = ROOT / "labels_fire_person"
RGB_ROOT = ROOT / "RGB"
IR_ROOT = ROOT / "IR"


def read_lines(path: Path) -> list[str]:
    """Read non-empty lines from a text file."""
    return [line.strip() for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def parse_label_line(line: str) -> tuple[int, list[float]]:
    """Parse one YOLO label line."""
    parts = line.split()
    return int(parts[0]), [float(x) for x in parts[1:5]]


def scan_split(split: str) -> dict:
    """Scan one split and return label/image/cache diagnostics."""
    stems = read_lines(ROOT / f"{split}.txt")
    stats = Counter()
    class_counts = Counter()
    combo_counts = Counter()
    bad: list[tuple] = []
    areas = []
    boxes_per_image = []
    remap_mismatches = []

    for stem in stems:
        rgb = RGB_ROOT / split / f"{stem}.jpg"
        ir = IR_ROOT / split / f"{stem}.jpg"
        original_label = ORIG_LABEL_ROOT / split / f"{stem}.txt"
        binary_label = BINARY_LABEL_ROOT / split / f"{stem}.txt"

        if not rgb.exists():
            bad.append(("missing_rgb", stem, str(rgb)))
        if not ir.exists():
            bad.append(("missing_ir", stem, str(ir)))
        if not binary_label.exists():
            bad.append(("missing_binary_label", stem, str(binary_label)))

        original_lines = read_lines(original_label)
        binary_lines = read_lines(binary_label)
        boxes_per_image.append(len(binary_lines))
        stats["empty_images"] += int(len(binary_lines) == 0)
        stats["nonempty_images"] += int(len(binary_lines) > 0)

        expected = []
        original_classes = set()
        binary_classes = set()
        seen = set()

        for line in original_lines:
            cls, vals = parse_label_line(line)
            original_classes.add(cls)
            if cls == 0:
                continue
            expected.append(" ".join([str(cls - 1)] + line.split()[1:]))

        if expected != binary_lines:
            remap_mismatches.append((stem, expected[:5], binary_lines[:5], original_lines[:5]))

        for line_no, line in enumerate(binary_lines, 1):
            parts = line.split()
            if len(parts) != 5:
                bad.append(("bad_columns", stem, line_no, line))
                continue
            try:
                cls, vals = parse_label_line(line)
            except Exception as e:  # noqa: BLE001
                bad.append(("parse_error", stem, line_no, line, repr(e)))
                continue

            binary_classes.add(cls)
            class_counts[cls] += 1
            x, y, w, h = vals
            areas.append(w * h)
            key = (cls, *vals)
            if key in seen:
                bad.append(("duplicate_box", stem, line_no, line))
            seen.add(key)
            if cls not in {0, 1}:
                bad.append(("class_out_of_range", stem, line_no, cls, line))
            if not all(np.isfinite(vals)):
                bad.append(("nonfinite_xywh", stem, line_no, vals))
            if not (0 <= x <= 1 and 0 <= y <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
                bad.append(("xywh_out_of_bounds", stem, line_no, vals))
            if w <= 0 or h <= 0:
                bad.append(("nonpositive_wh", stem, line_no, vals, line))

        combo_key = "+".join(str(c) for c in sorted(original_classes)) if original_classes else "empty"
        combo_counts[combo_key] += 1

    return {
        "split": split,
        "images": len(stems),
        "stats": stats,
        "class_counts": class_counts,
        "combo_counts": combo_counts,
        "boxes_per_image": np.array(boxes_per_image, dtype=float),
        "areas": np.array(areas, dtype=float),
        "bad": bad,
        "remap_mismatches": remap_mismatches,
    }


def inspect_cache(split: str) -> dict:
    """Inspect binary label cache content."""
    cache_path = BINARY_LABEL_ROOT / f"{split}.cache"
    if not cache_path.exists():
        return {"exists": False, "path": str(cache_path)}
    cache = load_dataset_cache_file(cache_path)
    class_counts = Counter()
    empty = 0
    nonpositive = []
    for label in cache["labels"]:
        cls = label["cls"]
        boxes = label["bboxes"]
        empty += int(len(cls) == 0)
        for c in cls.reshape(-1).tolist():
            class_counts[int(c)] += 1
        if len(boxes) and (boxes[:, 2:4] <= 0).any():
            nonpositive.append((label["im_file"], cls.tolist(), boxes.tolist()))
    return {
        "exists": True,
        "path": str(cache_path),
        "results": cache.get("results"),
        "labels": len(cache["labels"]),
        "empty": empty,
        "class_counts": class_counts,
        "nonpositive_wh": nonpositive[:5],
    }


def print_scan(scan: dict) -> None:
    """Print a readable scan summary."""
    split = scan["split"]
    bpi = scan["boxes_per_image"]
    areas = scan["areas"]
    print(f"=== {split} direct scan ===")
    print(f"images: {scan['images']}")
    print(f"empty_images: {scan['stats']['empty_images']}")
    print(f"nonempty_images: {scan['stats']['nonempty_images']}")
    print(f"class_counts: {dict(scan['class_counts'])}")
    print(f"original_class_combos: {dict(scan['combo_counts'])}")
    print(
        "boxes/image p0,p25,p50,mean,p75,p100:",
        [round(float(x), 6) for x in [*np.percentile(bpi, [0, 25, 50]), bpi.mean(), *np.percentile(bpi, [75, 100])]],
    )
    if len(areas):
        print("area p0,p1,p5,p50,p95,p99,p100:", [round(float(x), 8) for x in np.percentile(areas, [0, 1, 5, 50, 95, 99, 100])])
    print(f"bad_count: {len(scan['bad'])}")
    print(f"bad_examples: {scan['bad'][:10]}")
    print(f"remap_mismatch_count: {len(scan['remap_mismatches'])}")
    print(f"remap_mismatch_examples: {scan['remap_mismatches'][:3]}")
    print()


def main() -> None:
    """Run diagnostics."""
    print(f"dataset_root: {ROOT}")
    print(f"binary_label_root: {BINARY_LABEL_ROOT}")
    print()
    for split in ("train", "val"):
        print_scan(scan_split(split))
        print(f"=== {split} cache ===")
        print(inspect_cache(split))
        print()


if __name__ == "__main__":
    main()
