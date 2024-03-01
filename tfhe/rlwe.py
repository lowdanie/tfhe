import dataclasses

import numpy as np

from tfhe import lwe, polynomial, utils
from tfhe.polynomial import Polynomial


@dataclasses.dataclass
class RlweConfig:
    degree: int  # Messages will be in the space Z[X]/(x^degree + 1)
    noise_std: float  # The std of the noise added during encryption.


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


def rlwe_encode(p: Polynomial, config: RlweConfig) -> RlwePlaintext:
    """Encode a polynomial with coefficients in [-4, 4) as an RLWE plaintext."""
    encode_coeff = np.array([utils.encode(i) for i in p.coeff])
    return RlwePlaintext(
        config=config, message=polynomial.Polynomial(N=p.N, coeff=encode_coeff)
    )


def rlwe_decode(plaintext: RlwePlaintext) -> Polynomial:
    """Decode an RLWE plaintext to a polynomial with coefficients in [-4, 4) mod 8."""
    decode_coeff = np.array([utils.decode(i) for i in plaintext.message.coeff])
    return Polynomial(N=plaintext.message.N, coeff=decode_coeff)


def build_zero_rlwe_plaintext(config: RlweConfig) -> RlwePlaintext:
    """Build a an RLWE plaintext containing the zero polynomial."""
    return RlwePlaintext(
        config=config, message=polynomial.zero_polynomial(config.degree)
    )


def build_monomial_rlwe_plaintext(
    c: int, i: int, config: RlweConfig
) -> RlwePlaintext:
    """Build an RLWE plaintext containing the monomial c*x^i"""
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
        coeff=utils.uniform_sample_int32(size=key.config.degree),
    )
    noise = Polynomial(
        N=key.config.degree,
        coeff=utils.gaussian_sample_int32(
            std=key.config.noise_std, size=key.config.degree
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


def rlwe_trivial_ciphertext(
    f: Polynomial, config: RlweConfig
) -> RlweCiphertext:
    """Generate a trivial encryption of the plaintext."""
    if f.N != config.degree:
        raise ValueError(
            f"The degree of f ({f.N}) does not match the config degree ({config.degree}) "
        )

    return RlweCiphertext(
        config=config,
        a=polynomial.zero_polynomial(config.degree),
        b=f,
    )


def rlwe_add(
    ciphertext_left: RlweCiphertext, ciphertext_right: RlweCiphertext
) -> RlweCiphertext:
    """Homomorphically add two RLWE ciphertexts."""
    return RlweCiphertext(
        ciphertext_left.config,
        polynomial.polynomial_add(ciphertext_left.a, ciphertext_right.a),
        polynomial.polynomial_add(ciphertext_left.b, ciphertext_right.b),
    )


def rlwe_subtract(
    ciphertext_left: RlweCiphertext, ciphertext_right: RlweCiphertext
) -> RlweCiphertext:
    """Homomorphically subtract two RLWE ciphertexts."""
    return RlweCiphertext(
        ciphertext_left.config,
        polynomial.polynomial_subtract(ciphertext_left.a, ciphertext_right.a),
        polynomial.polynomial_subtract(ciphertext_left.b, ciphertext_right.b),
    )


def rlwe_plaintext_multiply(
    c: RlwePlaintext, ciphertext: RlweCiphertext
) -> RlweCiphertext:
    """Homomorphically multiply an RLWE ciphertext by a plaintext polynomial."""
    return RlweCiphertext(
        ciphertext.config,
        polynomial.polynomial_multiply(c.message, ciphertext.a),
        polynomial.polynomial_multiply(c.message, ciphertext.b),
    )
