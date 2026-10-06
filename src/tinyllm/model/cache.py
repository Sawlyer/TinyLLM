"""Layer-local key/value cache for autoregressive decoding."""

from torch import Tensor


class KVCache:
    """Store rotary-encoded keys and values for one Transformer layer."""

    def __init__(
        self,
        key: Tensor | None = None,
        value: Tensor | None = None,
        *,
        capacity: int | None = None,
    ) -> None:
        if (key is None) != (value is None):
            raise ValueError("key and value must both be set or both be empty")
        if capacity is not None and capacity <= 0:
            raise ValueError("cache capacity must be positive")

        self._key_storage: Tensor | None = None
        self._value_storage: Tensor | None = None
        self._length = 0
        self._capacity = 0 if capacity is None else capacity
        self._fixed_capacity = capacity is not None

        if key is not None and value is not None:
            self._validate_pair(key, value)
            if capacity is not None and key.shape[2] > capacity:
                raise ValueError("initial key/value sequence exceeds cache capacity")
            self.append(key, value)

    @property
    def key(self) -> Tensor | None:
        """Return populated key storage as a view."""
        if self._key_storage is None:
            return None
        return self._key_storage[:, :, : self._length]

    @property
    def value(self) -> Tensor | None:
        """Return populated value storage as a view."""
        if self._value_storage is None:
            return None
        return self._value_storage[:, :, : self._length]

    @property
    def length(self) -> int:
        """Return cached sequence length."""
        return self._length

    @property
    def capacity(self) -> int:
        """Return allocated sequence capacity."""
        return self._capacity

    def reserve(self, capacity: int) -> "KVCache":
        """Reserve a fixed maximum sequence capacity."""
        if capacity <= 0:
            raise ValueError("cache capacity must be positive")
        if capacity < self._length:
            raise ValueError("cache capacity cannot be smaller than cached sequence length")

        if self._key_storage is not None and capacity > self._capacity:
            self._reallocate(capacity)
        else:
            self._capacity = capacity
        self._fixed_capacity = True
        return self

    def append(self, key: Tensor, value: Tensor) -> "KVCache":
        """Append a sequence chunk and return this cache."""
        self._validate_pair(key, value)
        required_capacity = self._length + key.shape[2]

        if self._key_storage is None or self._value_storage is None:
            if self._fixed_capacity and required_capacity > self._capacity:
                raise ValueError("key/value append exceeds cache capacity")
            allocation = self._capacity if self._fixed_capacity else max(1, required_capacity)
            self._allocate(key, value, allocation)
        else:
            self._validate_append_compatibility(key, value)
            if required_capacity > self._capacity:
                if self._fixed_capacity:
                    raise ValueError("key/value append exceeds cache capacity")
                self._reallocate(max(required_capacity, self._capacity * 2))

        if self._key_storage is None or self._value_storage is None:
            raise RuntimeError("cache storage allocation failed")
        end = required_capacity
        self._key_storage[:, :, self._length : end].copy_(key)
        self._value_storage[:, :, self._length : end].copy_(value)
        self._length = end
        return self

    def _allocate(self, key: Tensor, value: Tensor, capacity: int) -> None:
        storage_shape = (*key.shape[:2], capacity, key.shape[3])
        self._key_storage = key.new_empty(storage_shape)
        self._value_storage = value.new_empty(storage_shape)
        self._capacity = capacity

    def _reallocate(self, capacity: int) -> None:
        if self._key_storage is None or self._value_storage is None:
            self._capacity = capacity
            return

        storage_shape = (*self._key_storage.shape[:2], capacity, self._key_storage.shape[3])
        key_storage = self._key_storage.new_empty(storage_shape)
        value_storage = self._value_storage.new_empty(storage_shape)
        key_storage[:, :, : self._length].copy_(self.key)
        value_storage[:, :, : self._length].copy_(self.value)
        self._key_storage = key_storage
        self._value_storage = value_storage
        self._capacity = capacity

    def _validate_append_compatibility(self, key: Tensor, value: Tensor) -> None:
        if self._key_storage is None or self._value_storage is None:
            return
        if key.shape[:2] + key.shape[3:] != (
            self._key_storage.shape[:2] + self._key_storage.shape[3:]
        ):
            raise ValueError("new key/value dimensions must match the cached dimensions")
        if key.device != self._key_storage.device or value.device != self._value_storage.device:
            raise ValueError("new key/value tensors must use the cached device")
        if key.dtype != self._key_storage.dtype or value.dtype != self._value_storage.dtype:
            raise ValueError("new key/value tensors must use the cached dtype")

    @staticmethod
    def _validate_pair(key: Tensor, value: Tensor) -> None:
        if key.ndim != 4 or value.ndim != 4:
            raise ValueError("key and value must have shape [batch, heads, sequence, head_dim]")
        if key.shape != value.shape:
            raise ValueError("key and value must have identical shapes")
        if key.dtype != value.dtype:
            raise ValueError("key and value must use the same dtype")
        if key.device != value.device:
            raise ValueError("key and value must use the same device")
