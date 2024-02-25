from collections.abc import Sequence
import dataclasses
import numpy as np

from tfhe import lwe
from tfhe import polynomial
from tfhe import rlwe


@dataclasses.dataclass
class GswConfig:
    rlwe_config: rlwe.RlweConfig
    log_p: int  # Homomorphic multiplication will use the base-2^log_p representation.


@dataclasses.dataclass
class GswPlaintext:
    config: GswConfig
    message: polynomial.Polynomial


@dataclasses.dataclass
class GswCiphertext:
    config: GswConfig
    rlwe_ciphertexts: list[rlwe.RlweCiphertext]


@dataclasses.dataclass
class GswEncryptionKey:
    config: GswConfig
    key: polynomial.Polynomial


@dataclasses.dataclass
class BootstrapKey:
    config: GswConfig
    gsw_ciphertexts: list[GswCiphertext]


def base_p_num_powers(log_p):
    return 32 // log_p


def array_to_base_p(a, log_p):
    num_powers = base_p_num_powers(log_p)
    half_p = np.int32(2 ** (log_p - 1))
    offset = half_p * sum(2 ** (i * log_p) for i in range(num_powers))
    mask = 2 ** (log_p) - 1

    a_offset = (a + offset).astype(np.uint32)

    output = []
    for i in range(num_powers):
        output.append(
            (np.right_shift(a_offset, i * log_p) & mask).astype(np.int32)
            - half_p
        )

    return output


def base_p_to_array(a_base_p, log_p):
    return sum(2 ** (i * log_p) * x for i, x in enumerate(a_base_p)).astype(
        np.int32
    )


def polynomial_to_base_p(
    f: polynomial.Polynomial, log_p: int
) -> Sequence[polynomial.Polynomial]:
    return [
        polynomial.Polynomial(coeff=v, N=f.N)
        for v in array_to_base_p(f.coeff, log_p=log_p)
    ]


def convert_lwe_key_to_gsw(
    lwe_key: lwe.LweEncryptionKey, gsw_config: GswConfig
) -> GswEncryptionKey:
    return GswEncryptionKey(
        config=gsw_config,
        key=polynomial.Polynomial(
            N=gsw_config.rlwe_config.degree, coeff=lwe_key.key
        ),
    )


def convert_rlwe_key_to_gsw(
    rlwe_key: rlwe.RlweEncryptionKey, gsw_config: GswConfig
) -> GswEncryptionKey:
    return GswEncryptionKey(config=gsw_config, key=rlwe_key.key)


def convert_gws_key_to_rlwe(
    gsw_key: GswEncryptionKey,
) -> rlwe.RlweEncryptionKey:
    return rlwe.RlweEncryptionKey(
        config=gsw_key.config.rlwe_config, key=gsw_key.key
    )


def gsw_encrypt(
    plaintext: GswPlaintext, key: GswEncryptionKey
) -> GswCiphertext:
    gsw_config = key.config
    num_powers = base_p_num_powers(log_p=gsw_config.log_p)

    # Create 2 RLWE encryptions of 0 for each element of a base-p representation.
    rlwe_key = convert_gws_key_to_rlwe(key)
    rlwe_plaintext_zero = rlwe.build_zero_rlwe_plaintext(gsw_config.rlwe_config)
    rlwe_ciphertexts = [
        rlwe.rlwe_encrypt(rlwe_plaintext_zero, rlwe_key)
        for _ in range(2 * num_powers)
    ]

    # Add multiples p^i * message to the rlwe ciphertexts
    for i in range(num_powers):
        p_i = 2 ** (i * gsw_config.log_p)
        scaled_message = polynomial.polynomial_constant_multiply(
            p_i, plaintext.message
        )

        rlwe_ciphertexts[i].a = polynomial.polynomial_add(
            rlwe_ciphertexts[i].a, scaled_message
        )

        b_idx = i + num_powers  # num_levels
        rlwe_ciphertexts[b_idx].b = polynomial.polynomial_add(
            rlwe_ciphertexts[b_idx].b, scaled_message
        )

    return GswCiphertext(gsw_config, rlwe_ciphertexts)


def gsw_multiply(
    gsw_ciphertext: GswCiphertext, rlwe_ciphertext: rlwe.RlweCiphertext
) -> rlwe.RlweCiphertext:
    gsw_config = gsw_ciphertext.config
    rlwe_config = rlwe_ciphertext.config

    # Concatenate the base-p representations of rlwe_ciphertext.a and rlwe_ciphertext.b
    rlwe_base_p = polynomial_to_base_p(
        rlwe_ciphertext.a, log_p=gsw_config.log_p
    ) + polynomial_to_base_p(rlwe_ciphertext.b, log_p=gsw_config.log_p)

    # Multiply the row vector rlwe_base_p with the
    # len(rlwe_base_p)x2 matrix gsw_ciphertext.rlwe_ciphertexts.
    rlwe_ciphertext = rlwe.RlweCiphertext(
        config=rlwe_config,
        a=polynomial.zero_polynomial(rlwe_config.degree),
        b=polynomial.zero_polynomial(rlwe_config.degree),
    )

    for i, p in enumerate(rlwe_base_p):
        rlwe_ciphertext.a = polynomial.polynomial_add(
            rlwe_ciphertext.a,
            polynomial.polynomial_multiply(
                p, gsw_ciphertext.rlwe_ciphertexts[i].a
            ),
        )
        rlwe_ciphertext.b = polynomial.polynomial_add(
            rlwe_ciphertext.b,
            polynomial.polynomial_multiply(
                p, gsw_ciphertext.rlwe_ciphertexts[i].b
            ),
        )

    return rlwe_ciphertext
