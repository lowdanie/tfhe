import numpy as np
import dataclasses

from tfhe import lwe
from tfhe import polynomial
from tfhe.polynomial import Polynomial

# TODO: move to utils
INT32_MIN = np.iinfo(np.int32).min
INT32_MAX = np.iinfo(np.int32).max


@dataclasses.dataclass
class RlweConfig:
    degree: int
    noise_std: float


@dataclasses.dataclass
class RlweEncryptionKey:
    config: RlweConfig
    key: Polynomial


@dataclasses.dataclass
class RlwePlaintext:
    config: RlweConfig
    message: Polynomial


@dataclasses.dataclass
class RlweCiphertext:
    config: RlweConfig
    a: Polynomial
    b: Polynomial

# TODO: rename to rlwe_encode/decode and use util.encode/decode
def encode_rlwe(p: Polynomial, config: RlweConfig) -> RlwePlaintext:
    encode_coeff = np.array([lwe.encode(i).message for i in p.coeff])
    return RlwePlaintext(
        config=config, message=polynomial.Polynomial(N=p.N, coeff=encode_coeff)
    )


def decode_rlwe(plaintext: RlwePlaintext) -> Polynomial:
    decode_coeff = np.array(
        [lwe.decode(lwe.LwePlaintext(i)) for i in plaintext.message.coeff]
    )
    return Polynomial(N=plaintext.message.N, coeff=decode_coeff)


def build_zero_rlwe_plaintext(config: RlweConfig) -> RlwePlaintext:
    return RlwePlaintext(
        config=config, message=polynomial.zero_polynomial(config.degree)
    )


def build_monomial_rlwe_plaintext(
    c: int, i: int, config: RlweConfig
) -> RlwePlaintext:
    return RlwePlaintext(
        config=config, message=polynomial.build_monomial(c, i, config.degree)
    )


def generate_rlwe_key(config: RlweConfig) -> np.ndarray:
    return RlweEncryptionKey(
        config=config,
        key=Polynomial(
            N=config.degree,
            coeff=np.random.randint(
                low=0, high=2, size=config.degree, dtype=np.int32
            ),
        ),
    )


def convert_lwe_key_to_rlwe(lwe_key: lwe.LweEncryptionKey) -> RlweEncryptionKey:
    rlwe_config = RlweConfig(
        degree=lwe_key.config.dimension, noise_std=lwe_key.config.noise_std
    )
    return RlweEncryptionKey(
        config=rlwe_config,
        key=Polynomial(N=rlwe_config.degree, coeff=lwe_key.key),
    )


def rlwe_encrypt(
    plaintext: RlwePlaintext, key: RlweEncryptionKey
) -> RlweCiphertext:
    a = Polynomial(
        N=key.config.degree,
        coeff=np.random.randint(
            low=INT32_MIN,
            high=INT32_MAX + 1,
            size=key.config.degree,
            dtype=np.int32,
        ),
    )
    noise = Polynomial(
        N=key.config.degree,
        coeff=np.int32(
            INT32_MAX
            * np.random.normal(
                loc=0.0, scale=key.config.noise_std, size=key.config.degree
            )
        ),
    )

    b = polynomial.polynomial_add(
        polynomial.polynomial_multiply(a, key.key), plaintext.message
    )
    b = polynomial.polynomial_add(b, noise)

    return RlweCiphertext(config=key.config, a=a, b=b)


def rlwe_decrypt(
    ciphertext: RlweCiphertext, key: RlweEncryptionKey
) -> RlwePlaintext:
    message = polynomial.polynomial_subtract(
        ciphertext.b, polynomial.polynomial_multiply(ciphertext.a, key.key)
    )
    return RlwePlaintext(config=key.config, message=message)


def rlwe_add(
    ciphertext_left: RlweCiphertext, ciphertext_right: RlweCiphertext
) -> RlweCiphertext:
    return RlweCiphertext(
        ciphertext_left.config,
        polynomial.polynomial_add(ciphertext_left.a, ciphertext_right.a),
        polynomial.polynomial_add(ciphertext_left.b, ciphertext_right.b),
    )


def rlwe_subtract(
    ciphertext_left: RlweCiphertext, ciphertext_right: RlweCiphertext
) -> RlweCiphertext:
    return RlweCiphertext(
        ciphertext_left.config,
        polynomial.polynomial_subtract(ciphertext_left.a, ciphertext_right.a),
        polynomial.polynomial_subtract(ciphertext_left.b, ciphertext_right.b),
    )


def rlwe_plaintext_multiply(
    c: RlwePlaintext, ciphertext: RlweCiphertext
) -> RlweCiphertext:
    return RlweCiphertext(
        ciphertext.config,
        polynomial.polynomial_multiply(c.message, ciphertext.a),
        polynomial.polynomial_multiply(c.message, ciphertext.b),
    )
