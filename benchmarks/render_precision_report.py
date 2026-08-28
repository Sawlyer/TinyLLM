from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "docs" / "benchmarks" / "precision-report.json"
ASSETS = ROOT / "docs" / "assets"
COLORS = {"fp32": "#64748B", "fp16": "#2563EB", "bf16": "#F59E0B", "int8": "#0F766E"}
LABELS = {"fp32": "FP32", "fp16": "FP16", "bf16": "BF16", "int8": "INT8"}
INK, MUTED, GRID, WHITE = "#0F172A", "#475569", "#E2E8F0", "#FFFFFF"


def main() -> None:
    payload = json.loads(REPORT.read_text(encoding="utf-8"))
    ASSETS.mkdir(parents=True, exist_ok=True)
    _training_chart(payload["training"])
    _inference_chart(payload["inference"])


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "arialbd.ttf" if bold else "arial.ttf"
    return ImageFont.truetype(str(Path("C:/Windows/Fonts") / name), size)


def _canvas(width: int, height: int, title: str, subtitle: str):
    image = Image.new("RGB", (width, height), WHITE)
    draw = ImageDraw.Draw(image)
    draw.text((width // 2, 38), title, font=_font(34, bold=True), fill=INK, anchor="ma")
    draw.text((width // 2, 86), subtitle, font=_font(19), fill=MUTED, anchor="ma")
    return image, draw


def _training_chart(rows: list[dict[str, object]]) -> None:
    rows = sorted(rows, key=lambda row: ("fp32", "fp16", "bf16").index(row["mode"]))
    image, draw = _canvas(
        1600,
        720,
        "RTX 5080 — training precision benchmark",
        "3 matched runs · batch 16 × 512 · 20 warmup + 100 measured steps",
    )
    _bar_panel(
        draw,
        (80, 150, 760, 630),
        "Training throughput",
        "thousand tokens / second",
        [(row["mode"], row["tokens_per_second_mean"] / 1000) for row in rows],
        lambda value: f"{value:,.1f}k",
    )
    _bar_panel(
        draw,
        (840, 150, 1520, 630),
        "Peak allocated VRAM",
        "GiB",
        [(row["mode"], row["peak_allocated_bytes"] / 1024**3) for row in rows],
        lambda value: f"{value:.2f}",
    )
    draw.text(
        (800, 682),
        "Source: docs/benchmarks/precision-report.json · measured 2026-08-21",
        font=_font(16),
        fill=MUTED,
        anchor="mm",
    )
    image.save(ASSETS / "precision-training-benchmark.png", quality=95)


def _inference_chart(rows: list[dict[str, object]]) -> None:
    baseline = float(rows[0]["perplexity"])
    image, draw = _canvas(
        1600,
        1120,
        "Checkpoint step 210 — inference precision and quality",
        "8 fixed validation batches · median of 5 measured passes · FP32 reference logits",
    )
    panels = (
        (
            "Median inference throughput",
            "thousand tokens / second",
            [(r["mode"], r["tokens_per_second_median"] / 1000) for r in rows],
            lambda v: f"{v:,.1f}k",
        ),
        (
            "Peak allocated VRAM",
            "GiB",
            [(r["mode"], r["peak_allocated_bytes"] / 1024**3) for r in rows],
            lambda v: f"{v:.2f}",
        ),
        (
            "Perplexity change vs FP32",
            "%",
            [(r["mode"], 100 * (r["perplexity"] / baseline - 1)) for r in rows],
            lambda v: f"{v:+.3f}%",
        ),
        (
            "Mean absolute logit error vs FP32",
            "absolute error",
            [(r["mode"], r["mean_absolute_logit_error_vs_fp32"]) for r in rows],
            lambda v: f"{v:.5f}",
        ),
    )
    boxes = (
        (80, 150, 760, 580),
        (840, 150, 1520, 580),
        (80, 630, 760, 1060),
        (840, 630, 1520, 1060),
    )
    for box, (title, unit, values, formatter) in zip(boxes, panels, strict=True):
        _bar_panel(draw, box, title, unit, values, formatter)
    draw.text(
        (800, 1090),
        "INT8 uses TorchAO 0.18.0 · Source: docs/benchmarks/precision-report.json",
        font=_font(16),
        fill=MUTED,
        anchor="mm",
    )
    image.save(ASSETS / "precision-inference-quality.png", quality=95)


def _bar_panel(draw: ImageDraw.ImageDraw, box, title: str, unit: str, values, formatter) -> None:
    left, top, right, bottom = box
    draw.rounded_rectangle(box, radius=18, outline=GRID, width=2, fill="#F8FAFC")
    draw.text((left + 28, top + 25), title, font=_font(24, bold=True), fill=INK)
    draw.text((left + 28, top + 60), unit, font=_font(16), fill=MUTED)
    chart_left, chart_top, chart_right, chart_bottom = left + 55, top + 105, right - 30, bottom - 65
    minimum = min(0.0, min(value for _, value in values))
    maximum = max(value for _, value in values)
    span = max(maximum - minimum, 0.01)
    ceiling = maximum + span * 0.24
    floor = minimum - span * 0.18 if minimum < 0 else 0.0
    zero_y = chart_bottom - (0 - floor) / (ceiling - floor) * (chart_bottom - chart_top)
    for fraction in (0, 0.5, 1):
        y = chart_top + fraction * (chart_bottom - chart_top)
        draw.line((chart_left, y, chart_right, y), fill=GRID, width=2)
    draw.line((chart_left, zero_y, chart_right, zero_y), fill=MUTED, width=2)
    slot = (chart_right - chart_left) / len(values)
    width = slot * 0.58
    for index, (mode, value) in enumerate(values):
        center = chart_left + slot * (index + 0.5)
        value_y = chart_bottom - (value - floor) / (ceiling - floor) * (chart_bottom - chart_top)
        y1, y2 = sorted((zero_y, value_y))
        draw.rounded_rectangle(
            (center - width / 2, y1, center + width / 2, y2),
            radius=7,
            fill=COLORS[mode],
            outline=INK,
            width=1,
        )
        label_y = value_y - 15 if value >= 0 else value_y + 15
        draw.text(
            (center, label_y),
            formatter(value),
            font=_font(17, bold=True),
            fill=INK,
            anchor="ms" if value >= 0 else "ma",
        )
        draw.text(
            (center, bottom - 32), LABELS[mode], font=_font(18, bold=True), fill=INK, anchor="mm"
        )


if __name__ == "__main__":
    main()
