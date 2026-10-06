"""Memory-mapped causal language-model batches."""

from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from tinyllm.data.artifacts import CorpusManifest, open_token_artifact


class TokenDataset(Dataset[tuple[Tensor, Tensor]]):
    """Read-only token windows backed by a verified corpus artifact."""

    def __init__(
        self,
        manifest: CorpusManifest,
        split: str,
        sequence_length: int,
        seed: int,
        *,
        artifact_dir: Path | None = None,
    ) -> None:
        if not isinstance(manifest, CorpusManifest):
            raise TypeError("manifest must be a CorpusManifest")
        if split not in manifest.artifacts:
            raise ValueError(f"unknown corpus split: {split}")
        if type(sequence_length) is not int or sequence_length <= 0:
            raise ValueError("sequence_length must be a positive integer")
        if type(seed) is not int:
            raise ValueError("seed must be an integer")

        artifact = manifest.artifacts[split]
        if artifact.token_count < sequence_length + 1:
            raise ValueError(
                "corpus must contain at least sequence_length + 1 tokens "
                f"(got {artifact.token_count}, need {sequence_length + 1})"
            )

        self.manifest = manifest
        self.split = split
        self.sequence_length = sequence_length
        self.seed = seed
        self._tokens = _open_manifest_artifact(manifest, artifact, artifact_dir=artifact_dir)
        self.token_count = artifact.token_count
        self._position_count = self.token_count - sequence_length
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(seed)

    @property
    def tokens(self) -> np.memmap | np.ndarray:
        """Underlying read-only memory map."""
        return self._tokens

    def __len__(self) -> int:
        return self._position_count

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        if type(index) is not int:
            raise TypeError("dataset index must be an integer")
        inputs, targets = self.batch(torch.tensor([index], dtype=torch.long))
        return inputs[0], targets[0]

    def batch(self, indices: Tensor) -> tuple[Tensor, Tensor]:
        """Return input windows and one-token-shifted target windows."""
        if not isinstance(indices, Tensor):
            raise TypeError("indices must be a torch.Tensor")
        if indices.ndim != 1:
            raise ValueError("indices must be a one-dimensional tensor")
        if indices.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
            raise TypeError("indices must contain integers")

        cpu_indices = indices.detach().to(device="cpu", dtype=torch.long)
        if cpu_indices.numel():
            if int(cpu_indices.min()) < 0 or int(cpu_indices.max()) >= self._position_count:
                raise IndexError(f"sample positions must be in [0, {self._position_count - 1}]")

        starts = cpu_indices.numpy()
        offsets = np.arange(self.sequence_length + 1, dtype=np.int64)
        windows = np.asarray(self._tokens)[starts[:, None] + offsets]
        window_tensor = torch.from_numpy(np.array(windows, dtype=np.uint16, copy=True)).to(
            dtype=torch.long
        )
        return window_tensor[:, :-1], window_tensor[:, 1:]

    def sample_positions(self, batch_size: int) -> Tensor:
        """Draw valid start positions from this dataset's serializable RNG."""
        if type(batch_size) is not int or batch_size < 0:
            raise ValueError("batch_size must be a non-negative integer")
        return torch.randint(
            low=0,
            high=self._position_count,
            size=(batch_size,),
            generator=self.generator,
            dtype=torch.long,
        )

    def get_rng_state(self) -> Tensor:
        """Return a clone suitable for checkpoint serialization."""
        return self.generator.get_state().clone()

    def set_rng_state(self, state: Tensor) -> None:
        """Restore a previously serialized generator state."""
        if not isinstance(state, Tensor):
            raise TypeError("RNG state must be a torch.Tensor")
        self.generator.set_state(state.detach().to(device="cpu"))

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "split": self.split,
            "sequence_length": self.sequence_length,
            "token_count": self.token_count,
            "position_count": self._position_count,
            "generator_device": "cpu",
            "generator_state": self.get_rng_state(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        generator_state = self.validate_state_dict(state)
        self.generator.set_state(generator_state)

    def validate_state_dict(self, state: Mapping[str, Any]) -> Tensor:
        """Validate serialized sampling domain and RNG without mutating this dataset."""
        required = {
            "version",
            "split",
            "sequence_length",
            "token_count",
            "position_count",
            "generator_device",
            "generator_state",
        }
        if not isinstance(state, Mapping) or set(state) != required:
            raise ValueError("dataset state has invalid fields")
        expected = {
            "version": 1,
            "split": self.split,
            "sequence_length": self.sequence_length,
            "token_count": self.token_count,
            "position_count": self._position_count,
            "generator_device": "cpu",
        }
        for field, expected_value in expected.items():
            if state[field] != expected_value:
                raise ValueError(f"loader {field} is incompatible")
        generator_state = state["generator_state"]
        if not isinstance(generator_state, Tensor):
            raise ValueError("loader generator_state must be a torch.Tensor")
        cpu_state = generator_state.detach().to(device="cpu")
        if cpu_state.dtype != torch.uint8:
            raise ValueError("loader generator_state must use torch.uint8")
        probe = torch.Generator(device="cpu")
        try:
            probe.set_state(cpu_state)
        except RuntimeError as error:
            raise ValueError("loader generator_state is invalid") from error
        return cpu_state


@dataclass(slots=True)
class _PrefetchedBatch:
    values: tuple[Tensor, Tensor]
    generator_state_after: Tensor


class DeterministicBatchPrefetcher:
    """Pin and look ahead without advancing checkpoint RNG past consumption."""

    def __init__(
        self,
        dataset: TokenDataset,
        batch_size: int,
        *,
        prefetch_batches: int = 2,
        pin_memory: bool = True,
    ) -> None:
        if not isinstance(dataset, TokenDataset):
            raise TypeError("dataset must be a TokenDataset")
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if type(prefetch_batches) is not int or prefetch_batches <= 0:
            raise ValueError("prefetch_batches must be a positive integer")
        self.dataset = dataset
        self.batch_size = batch_size
        self.prefetch_batches = prefetch_batches
        self.pin_memory = pin_memory and _pin_memory_supported()
        self._preview_generator = torch.Generator(device="cpu")
        self._queue: deque[_PrefetchedBatch] = deque()
        self.reset()

    def reset(self) -> None:
        """Regenerate lookahead from current consumed-boundary dataset RNG."""
        self._preview_generator.set_state(self.dataset.get_rng_state())
        self._queue.clear()
        self._fill()

    def next_batch(self) -> tuple[Tensor, Tensor]:
        """Consume one batch and commit only its RNG transition."""
        prefetched = self._queue.popleft()
        self.dataset.set_rng_state(prefetched.generator_state_after)
        self._fill()
        return prefetched.values

    def _fill(self) -> None:
        while len(self._queue) < self.prefetch_batches:
            positions = torch.randint(
                low=0,
                high=len(self.dataset),
                size=(self.batch_size,),
                generator=self._preview_generator,
                dtype=torch.long,
            )
            batch = self.dataset.batch(positions)
            if self.pin_memory:
                batch = _pin_batch(batch)
            self._queue.append(
                _PrefetchedBatch(
                    values=batch,
                    generator_state_after=self._preview_generator.get_state().clone(),
                )
            )


def create_dataloader(
    dataset: TokenDataset,
    batch_size: int,
    *,
    shuffle: bool = True,
    num_workers: int = 0,
    prefetch_factor: int | None = 2,
    pin_memory: bool = True,
    drop_last: bool = False,
) -> DataLoader[tuple[Tensor, Tensor]]:
    """Create a deterministic, pinned-memory loader for token windows."""
    if not isinstance(dataset, TokenDataset):
        raise TypeError("dataset must be a TokenDataset")
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if type(num_workers) is not int or num_workers < 0:
        raise ValueError("num_workers must be a non-negative integer")
    if prefetch_factor is not None and (type(prefetch_factor) is not int or prefetch_factor <= 0):
        raise ValueError("prefetch_factor must be a positive integer or None")

    loader_args: dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": pin_memory and _pin_memory_supported(),
        "drop_last": drop_last,
        "generator": dataset.generator,
    }
    if num_workers > 0 and prefetch_factor is not None:
        loader_args["prefetch_factor"] = prefetch_factor
    return DataLoader(dataset, **loader_args)


build_dataloader = create_dataloader


def _pin_memory_supported() -> bool:
    if torch.cuda.is_available():
        return True
    mps = getattr(torch.backends, "mps", None)
    return bool(mps is not None and mps.is_available())


def _pin_batch(batch: tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor]:
    return batch[0].pin_memory(), batch[1].pin_memory()


def _open_manifest_artifact(
    manifest: CorpusManifest,
    artifact: Any,
    *,
    artifact_dir: Path | None,
) -> np.memmap | np.ndarray:
    manifest_path = getattr(manifest, "_manifest_path", None)
    if isinstance(manifest_path, Path):
        return open_token_artifact(manifest_path.parent, artifact)
    if artifact_dir is None:
        raise ValueError("manifest has no associated path; artifact_dir is required")
    return open_token_artifact(Path(artifact_dir), artifact)
