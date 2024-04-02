import numpy as np

from tfhe import bootstrap, lwe


def lwe_nand(
    lwe_ciphertext_left: lwe.LweCiphertext,
    lwe_ciphertext_right: lwe.LweCiphertext,
    bootstrap_key: bootstrap.BootstrapKey,
) -> lwe.LweCiphertext:
    """Homomorphically evaluate the NAND function.

    Suppose that lwe_ciphertext_left is an LWE encryption of an encoding of the
    boolean b_left and lwe_ciphertext_right is an LWE encryption of an encoding
    of the boolean b_right. Then the the output is an LWE encryption of an encoding
    of NAND(b_left, b_right).
    """
    # Compute an LWE encryption of: encode(-3) - m_left - m_right
    initial_lwe_ciphertext = lwe.lwe_trivial_ciphertext(
        plaintext=lwe.lwe_encode(-3),
        config=lwe_ciphertext_left.config,
    )

    test_lwe_ciphertext = lwe.lwe_subtract(
        initial_lwe_ciphertext, lwe_ciphertext_left
    )
    test_lwe_ciphertext = lwe.lwe_subtract(
        test_lwe_ciphertext, lwe_ciphertext_right
    )

    # Bootstrap the test_lwe_ciphertext to an encoding of True.
    return bootstrap.bootstrap(
        test_lwe_ciphertext, bootstrap_key, scale=utils.encode_bool(True)
    )
