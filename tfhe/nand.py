import numpy as np

from tfhe import bootstrap, lwe


def nand_gate(
    lwe_ciphertext_left: lwe.LweCiphertext,
    lwe_ciphertext_right: lwe.LweCiphertext,
    bootstrap_key: bootstrap.BootstrapKey,
) -> lwe.LweCiphertext:
    """The input lwe ciphertext encrypt either encode(0) or encode(2)=2**30"""
    # Compute an LWE encryption of: encode(-3) - m1 - m2
    # if m1=encode(2), m2=encode(0):  encode(-3) - encode(2) - encode(0) = encode(-5) = encode(3)
    # if m1 = m2 = encode(2): encode(-3) - encode(2) - encode(2) = encode(-7) = encode(1)
    lwe_config = lwe_ciphertext_left.config

    # TODO: this could be replaced with a trivial_ciphertext method
    initial_lwe_ciphertext = lwe.LweCiphertext(
        config=lwe_config,
        a=np.zeros(lwe_config.dimension, dtype=np.int32),
        b=lwe.encode(-3).message,
    )
    test_lwe_ciphertext = lwe.lwe_subtract(
        initial_lwe_ciphertext, lwe_ciphertext_left
    )
    test_lwe_ciphertext = lwe.lwe_subtract(
        test_lwe_ciphertext, lwe_ciphertext_right
    )

    return bootstrap.bootstrap(
        test_lwe_ciphertext, bootstrap_key, scale=lwe.encode(2).message
    )
