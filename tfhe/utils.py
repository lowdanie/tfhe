from typing import Optional

import numpy as np

INT32_MIN = np.iinfo(np.int32).min
INT32_MAX = np.iinfo(np.int32).max


def uniform_sample_int32(size: int) -> np.ndarray:
    return np.random.randint(
        low=INT32_MIN,
        high=INT32_MAX + 1,
        size=size,
        dtype=np.int32,
    )


def gaussian_sample_int32(std: float, size: Optional[float]) -> np.ndarray:
    return np.int32(INT32_MAX * np.random.normal(loc=0.0, scale=std, size=size))


def encode(i: int) -> np.int32:
    """Encode an integer in [-4, 4) as an int32"""
    return np.multiply(i, 1 << 29, dtype=np.int32)


def decode(i: np.int32) -> int:
    """Decode an int32 to an integer in the range [-4, 4) mod 8"""
    d = int(np.rint(i / (1 << 29)))
    return ((d + 4) % 8) - 4

def encode_bool(b: bool) -> np.int32:
    """Encode a bit as an int32."""
    return encode(2 * int(b))


def decode_bool(i: np.int32) -> bool:
    """Decode an int32 to a bool."""
    return bool(decode(i) / 2)
