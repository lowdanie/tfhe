import dataclasses

import numpy as np

from tfhe import utils


@dataclasses.dataclass
class LweConfig:
    # Size of the LWE encryption key.
    dimension: int

    # Standard deviation of the encryption noise.
    noise_std: float


@dataclasses.dataclass
class LwePlaintext:
    message: np.int32


@dataclasses.dataclass
class LweCiphertext:
    config: LweConfig
    a: np.ndarray  # An int32 array of size config.dimension
    b: np.int32


@dataclasses.dataclass
class LweEncryptionKey:
    config: LweConfig
    key: np.ndarray  # An int32 array of size config.dimension


def lwe_encode(i: int) -> LwePlaintext:
    return LwePlaintext(utils.encode(i))


def lwe_decode(plaintext: LwePlaintext) -> int:
    return utils.decode(plaintext.message)


def lwe_encode_bool(b: bool) -> LwePlaintext:
    return LwePlaintext(utils.encode_bool(b))


def lwe_decode_bool(plaintext: LwePlaintext) -> bool:
    return utils.decode_bool(plaintext.message)


def generate_lwe_key(config: LweConfig) -> LweEncryptionKey:
    return LweEncryptionKey(
        config=config,
        key=np.random.randint(
            low=0, high=2, size=(config.dimension,), dtype=np.int32
        ),
    )


def lwe_encrypt(
    plaintext: LwePlaintext, key: LweEncryptionKey
) -> LweCiphertext:
    a = utils.uniform_sample_int32(size=key.config.dimension)
    noise = utils.gaussian_sample_int32(std=key.config.noise_std, size=None)

    # b = (a, key) + message + noise
    b = np.add(np.dot(a, key.key), plaintext.message, dtype=np.int32)
    b = np.add(b, noise, dtype=np.int32)

    return LweCiphertext(config=key.config, a=a, b=b)


def lwe_decrypt(
    ciphertext: LweCiphertext, key: LweEncryptionKey
) -> LwePlaintext:
    return LwePlaintext(
        np.subtract(ciphertext.b, np.dot(ciphertext.a, key.key), dtype=np.int32)
    )


def lwe_trivial_ciphertext(plaintext: LwePlaintext, config: LweConfig):
    return LweCiphertext(
        config=config,
        a=np.zeros(config.dimension, dtype=np.int32),
        b=plaintext.message,
    )


def lwe_add(
    ciphertext_left: LweCiphertext, ciphertext_right: LweCiphertext
) -> LweCiphertext:
    return LweCiphertext(
        ciphertext_left.config,
        np.add(ciphertext_left.a, ciphertext_right.a, dtype=np.int32),
        np.add(ciphertext_left.b, ciphertext_right.b, dtype=np.int32),
    )


def lwe_subtract(
    ciphertext_left: LweCiphertext, ciphertext_right: LweCiphertext
) -> LweCiphertext:
    return LweCiphertext(
        ciphertext_left.config,
        np.subtract(ciphertext_left.a, ciphertext_right.a, dtype=np.int32),
        np.subtract(ciphertext_left.b, ciphertext_right.b, dtype=np.int32),
    )


def lwe_plaintext_multiply(c: int, ciphertext: LweCiphertext) -> LweCiphertext:
    return LweCiphertext(
        ciphertext.config,
        np.multiply(c, ciphertext.a, dtype=np.int32),
        np.multiply(c, ciphertext.b, dtype=np.int32),
    )
