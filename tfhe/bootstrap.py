import dataclasses
from collections.abc import Sequence

import numpy as np

from tfhe import gsw, lwe, polynomial, rlwe


@dataclasses.dataclass
class BootstrapKey:
    config: gsw.GswConfig
    gsw_ciphertexts: Sequence[gsw.GswCiphertext]


def generate_bootstrap_key(
    lwe_key: lwe.LweEncryptionKey, gsw_key: gsw.GswEncryptionKey
) -> BootstrapKey:
    bootstrap_key = BootstrapKey(config=gsw_key.config, gsw_ciphertexts=[])

    for b in lwe_key.key:
        b_plaintext = rlwe.build_monomial_rlwe_plaintext(
            b, 0, gsw_key.config.rlwe_config
        )
        bootstrap_key.gsw_ciphertexts.append(
            gsw.gsw_encrypt(b_plaintext, gsw_key)
        )

    return bootstrap_key


def blind_rotate(
    lwe_ciphertext: lwe.LweCiphertext,
    rlwe_ciphertext: rlwe.RlweCiphertext,
    bootstrap_key: BootstrapKey,
) -> rlwe.RlweCiphertext:
    """Homomorphically evaluate the function: Rotate(i, f(x)) = x^i * f(x).

    Suppose lwe_ciphertext is an encryption of i and rlwe_ciphertext is an
    encryption of a polynomial f(x). Then the output will be an encryption
    of x^i * f(x).
    """
    N = rlwe_ciphertext.config.degree

    # scale the lwe_ciphertext by N / 2^31 so that the message is between -N and N
    scaled_lwe_a = np.int32(np.rint(lwe_ciphertext.a * (N * 2 ** (-31))))
    scaled_lwe_b = np.int32(np.rint(lwe_ciphertext.b * (N * 2 ** (-31))))

    # Initialize the rotation by X^b
    rotated_rlwe_ciphertext = rlwe.rlwe_plaintext_multiply(
        rlwe.build_monomial_rlwe_plaintext(
            1, scaled_lwe_b, rlwe_ciphertext.config
        ),
        rlwe_ciphertext,
    )

    # Rotate by X^-a_i if s_i = 1
    for i, a_i in enumerate(scaled_lwe_a):
        rotated_rlwe_ciphertext = gsw.cmux(
            bootstrap_key.gsw_ciphertexts[i],
            rotated_rlwe_ciphertext,
            rlwe.rlwe_plaintext_multiply(
                rlwe.build_monomial_rlwe_plaintext(
                    1, -a_i, rlwe_ciphertext.config
                ),
                rotated_rlwe_ciphertext,
            ),
        )

    return rotated_rlwe_ciphertext


def extract_sample(
    i: int, rlwe_ciphertext: rlwe.RlweCiphertext
) -> lwe.LweCiphertext:
    """Homomorphically extract the i-th coefficient from the RLWE ciphertext.

    If rlwe_ciphertext is an RLWE encryption of
    f(x) = c_0 + c_1*x + ... + c_{N-1}x^{N-1}
    then the output will be an LWE encryption of c_i.
    """
    lwe_config = lwe.LweConfig(
        dimension=rlwe_ciphertext.config.degree,
        noise_std=rlwe_ciphertext.config.noise_std,
    )
    a = np.hstack(
        [
            rlwe_ciphertext.a.coeff[: i + 1][::-1],
            -1 * rlwe_ciphertext.a.coeff[i + 1 :][::-1],
        ]
    )
    b = rlwe_ciphertext.b.coeff[i]
    return lwe.LweCiphertext(lwe_config, a, b)


def _build_test_polynomial(N: int) -> polynomial.Polynomial:
    p = polynomial.Polynomial(N=N, coeff=np.ones(N, dtype=np.int32))
    p.coeff[: N // 2] = -1
    return p


def bootstrap(
    lwe_ciphertext: lwe.LweCiphertext,
    bootstrap_key: BootstrapKey,
    scale: np.int32,
) -> lwe.LweCiphertext:
    """Homomorphically evaluate the function Step(i) = scale if |i|>2**29 else 0
    
    Suppose that lwe_ciphertext is an encryption of the int32 i. If |i| > 2**29
    then return an LWE encryption of the scale argument. Otherwise return an LWE
    encryption of 0. In both cases the ciphertext noise will be bounded and
    independent of the lwe_ciphertext noise.
    """
    N = bootstrap_key.config.rlwe_config.degree
    test_polynomial = polynomial.polynomial_constant_multiply(
        scale // 2, _build_test_polynomial(N)
    )
    test_rlwe_ciphertext = rlwe.rlwe_trivial_ciphertext(
        test_polynomial, bootstrap_key.config.rlwe_config
    )

    rotated_rlwe_ciphertext = blind_rotate(
        lwe_ciphertext, test_rlwe_ciphertext, bootstrap_key
    )

    sample_lwe_ciphertext = extract_sample(0, rotated_rlwe_ciphertext)

    offset_lwe_ciphertext = lwe.lwe_trivial_ciphertext(
        plaintext=lwe.LwePlaintext(scale // 2),
        config=sample_lwe_ciphertext.config,
    )

    return lwe.lwe_add(offset_lwe_ciphertext, sample_lwe_ciphertext)
