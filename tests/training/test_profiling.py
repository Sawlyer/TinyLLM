from pathlib import Path
from types import SimpleNamespace

import torch

from tinyllm.training import profiling


class FakeEvent:
    def __init__(self, key: str, cpu_time: float, device_time: float) -> None:
        self.key = key
        self.count = 1
        self.self_cpu_time_total = cpu_time
        self.self_device_time_total = device_time


class FakeProfiler:
    def __init__(self, **options) -> None:
        self.options = options
        self.steps = 0

    def __enter__(self):
        return self

    def __exit__(self, *_args) -> None:
        return None

    def step(self) -> None:
        self.steps += 1

    def key_averages(self):
        return [
            FakeEvent("tinyllm/model_forward", 12.0, 6.0),
            FakeEvent("tinyllm/optimizer", 4.0, 2.0),
            FakeEvent("aten::linear", 8.0, 4.0),
        ]

    def export_chrome_trace(self, path: str) -> None:
        Path(path).write_text("{}", encoding="utf-8")


class FakeTrainer:
    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.config = SimpleNamespace(
            training=SimpleNamespace(micro_batch_size=2, gradient_accumulation_steps=1),
            model=SimpleNamespace(max_seq_len=4),
        )
        self.train_dataset = SimpleNamespace(sequence_length=4)
        self.metric_logger = None

    def train(self, *, max_steps: int, on_step_end):
        for step in range(1, max_steps + 1):
            self.metric_logger.log(
                step,
                {
                    "tokens_per_second": 40.0 / step,
                    "data_loading_seconds": step / 10,
                    "evaluation_seconds": 0.0 if step < max_steps else 0.2,
                    "peak_vram_bytes": 0,
                },
            )
            on_step_end(step)
        return SimpleNamespace(elapsed_seconds=1.0)


def test_profile_separates_warmup_and_collects_step_operator_and_memory_data(
    monkeypatch,
) -> None:
    created: list[FakeProfiler] = []
    schedules: list[dict[str, int]] = []

    def make_profiler(**options):
        profiler = FakeProfiler(**options)
        created.append(profiler)
        return profiler

    monkeypatch.setattr(profiling.torch.profiler, "profile", make_profiler)
    monkeypatch.setattr(
        profiling.torch.profiler,
        "schedule",
        lambda **options: schedules.append(options) or options,
    )

    report = profiling.run_training_profile(
        FakeTrainer(),
        warmup_steps=1,
        measured_steps=2,
    )

    assert created[0].steps == 4
    assert created[0].options["record_shapes"] is True
    assert created[0].options["profile_memory"] is True
    assert schedules == [{"wait": 1, "warmup": 1, "active": 2, "repeat": 1}]
    assert report["unmeasured_steps"] == 1
    assert report["warmup_steps"] == 1
    assert report["measured_steps"] == 2
    assert report["measurement"]["tokens_per_step"] == 8
    assert report["measurement"]["mean_data_loading_seconds"] == 0.35
    assert report["measurement"]["validation_seconds"] == 0.2
    assert report["phase_events"][0]["name"] == "tinyllm/model_forward"
    assert report["memory"]["peak_allocated_bytes"] == 0
    assert report["memory"]["peak_reserved_bytes"] is None


def test_profile_exports_trace_only_when_requested(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        profiling.torch.profiler,
        "profile",
        lambda **options: FakeProfiler(**options),
    )
    trace = tmp_path / "profile.json"

    report = profiling.run_training_profile(
        FakeTrainer(),
        warmup_steps=0,
        measured_steps=1,
        trace_path=trace,
    )

    assert trace.read_text(encoding="utf-8") == "{}"
    assert report["trace_file"] == str(trace)


def test_profile_rejects_invalid_step_counts_before_starting_profiler(monkeypatch) -> None:
    def unexpected_profile(**_options):
        raise AssertionError("profiler must not start for invalid step counts")

    monkeypatch.setattr(profiling.torch.profiler, "profile", unexpected_profile)

    try:
        profiling.run_training_profile(FakeTrainer(), warmup_steps=-1, measured_steps=2)
    except ValueError as error:
        assert "warmup_steps" in str(error)
    else:
        raise AssertionError("negative warmup should be rejected")


def test_profile_training_command_has_distinct_warmup_steps_and_output_options() -> None:
    from tinyllm.cli import _build_parser

    args = _build_parser().parse_args(
        [
            "profile-training",
            "--warmup-steps",
            "0",
            "--steps",
            "2",
            "--output",
            "profile.json",
            "--trace",
            "trace.json",
        ]
    )

    assert args.command == "profile-training"
    assert args.warmup_steps == 0
    assert args.measured_steps == 2
    assert args.output == Path("profile.json")
    assert args.trace == Path("trace.json")

