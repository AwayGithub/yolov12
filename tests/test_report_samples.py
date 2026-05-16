from pathlib import Path

from dev.make_ir_report_samples import find_fire_person_samples, load_requested_samples


def _touch_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"not-used-by-selection")


def _write_label(root: Path, split: str, stem: str, lines: list[str]) -> None:
    label_path = root / "labels_fire_person" / split / f"{stem}.txt"
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text("\n".join(lines), encoding="utf-8")


def test_find_fire_person_samples_requires_both_classes(tmp_path: Path) -> None:
    split = "val"
    stems = ["fire_only", "person_only", "both_a", "both_b"]
    (tmp_path / f"{split}.txt").write_text("\n".join(stems), encoding="utf-8")

    for stem in stems:
        _touch_image(tmp_path / "IR" / split / f"{stem}.jpg")

    _write_label(tmp_path, split, "fire_only", ["0 0.5 0.5 0.2 0.2"])
    _write_label(tmp_path, split, "person_only", ["1 0.5 0.5 0.2 0.2"])
    _write_label(tmp_path, split, "both_a", ["0 0.3 0.3 0.2 0.2", "1 0.7 0.7 0.2 0.2"])
    _write_label(tmp_path, split, "both_b", ["1 0.4 0.4 0.2 0.2", "0 0.6 0.6 0.2 0.2"])

    samples = find_fire_person_samples(tmp_path, split=split, count=2)

    assert [sample.stem for sample in samples] == ["both_a", "both_b"]
    assert all({0, 1}.issubset({box.class_id for box in sample.boxes}) for sample in samples)


def test_load_requested_samples_preserves_order_and_requires_both_classes(tmp_path: Path) -> None:
    split = "val"
    stems = ["both_b", "fire_only", "both_a"]
    (tmp_path / f"{split}.txt").write_text("\n".join(stems), encoding="utf-8")

    for stem in stems:
        _touch_image(tmp_path / "IR" / split / f"{stem}.jpg")

    _write_label(tmp_path, split, "both_b", ["1 0.4 0.4 0.2 0.2", "0 0.6 0.6 0.2 0.2"])
    _write_label(tmp_path, split, "fire_only", ["0 0.5 0.5 0.2 0.2"])
    _write_label(tmp_path, split, "both_a", ["0 0.3 0.3 0.2 0.2", "1 0.7 0.7 0.2 0.2"])

    samples = load_requested_samples(tmp_path, split=split, stems=["both_a", "both_b"])

    assert [sample.stem for sample in samples] == ["both_a", "both_b"]
    assert all({0, 1}.issubset({box.class_id for box in sample.boxes}) for sample in samples)
