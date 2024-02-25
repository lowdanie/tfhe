import numpy as np
import dataclasses

INT32_MIN = np.iinfo(np.int32).min
INT32_MAX = np.iinfo(np.int32).max


@dataclasses.dataclass
class LweConfig:
    dimension: int
    noise_std: float


@dataclasses.dataclass
class LwePlaintext:
    message: np.int32


@dataclasses.dataclass
class LweCiphertext:
    config: LweConfig
    a: np.ndarray
    b: np.int32


@dataclasses.dataclass
class LweEncryptionKey:
    config: LweConfig
    key: np.ndarray


# TODO: there should be a utils file with an encode and decode
# that operates on ints. These should be renamed to lwe_encode and lwe_decode
def encode(i: int) -> LwePlaintext:
    return LwePlaintext(np.multiply(i, 1 << 29, dtype=np.int32))


def decode(plaintext: LwePlaintext) -> int:
    return int(np.rint(plaintext.message / (1 << 29)))


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
    a = np.random.randint(
        low=INT32_MIN,
        high=INT32_MAX + 1,
        size=(key.config.dimension,),
        dtype=np.int32,
    )
    noise = np.int32(
        INT32_MAX * np.random.normal(loc=0.0, scale=key.config.noise_std)
    )

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
