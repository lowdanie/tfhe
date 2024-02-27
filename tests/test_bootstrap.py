import unittest

import numpy as np

from tfhe import bootstrap, config, gsw, lwe, polynomial, rlwe, utils


class TestBootstrap(unittest.TestCase):
    # TODO: Put in utils
    def assert_polynomial_equal(self, p_left, p_right):
        self.assertEqual(p_left.N, p_right.N)
        self.assertTrue(np.all(p_left.coeff == p_right.coeff))

    def test_blind_rotate(self):
        lwe_key = lwe.generate_lwe_key(config.LWE_CONFIG)

        rlwe_key = rlwe.generate_rlwe_key(config.RLWE_CONFIG)
        gsw_key = gsw.convert_rlwe_key_to_gsw(rlwe_key, config.GSW_CONFIG)
        bootstrap_key = bootstrap.generate_bootstrap_key(lwe_key, gsw_key)

        # f(x) = -1 -x ... -x^(N/2-1) + x^(N/2) + ... + x^(N-1)
        N = config.RLWE_CONFIG.degree
        f = polynomial.Polynomial(N=N, coeff=np.ones(N, dtype=np.int32))
        f.coeff[: N // 2] = -1

        f_plaintext = rlwe.rlwe_encode(f, config.RLWE_CONFIG)
        f_ciphertext = rlwe.rlwe_encrypt(f_plaintext, rlwe_key)

        # Rotate by 3/4 * N
        index_plaintext = lwe.lwe_encode(3)
        index_ciphertext = lwe.lwe_encrypt(index_plaintext, lwe_key)

        rotated_ciphertext = bootstrap.blind_rotate(
            index_ciphertext, f_ciphertext, bootstrap_key
        )

        rotated_f = rlwe.rlwe_decode(
            rlwe.rlwe_decrypt(rotated_ciphertext, rlwe_key)
        )

        # The rotated result should be:
        # x^(3/4 * N)f(x) = 1 + x ... + x^(N/4-1) - x^(N/4) - ... - x^(N-1)
        self.assertEqual(rotated_f.coeff[0], 1)
        self.assertEqual(rotated_f.coeff[N // 2], -1)
        self.assertEqual(rotated_f.coeff[-1], -1)

    def test_extract_sample(self):
        lwe_key = lwe.generate_lwe_key(config.LWE_CONFIG)
        rlwe_key = rlwe.convert_lwe_key_to_rlwe(lwe_key)

        N = lwe_key.config.dimension
        # f = 2x
        f = polynomial.build_monomial(2, 1, N)
        f_plaintext = rlwe.rlwe_encode(f, config.RLWE_CONFIG)
        f_ciphertext = rlwe.rlwe_encrypt(f_plaintext, rlwe_key)

        sample_ciphertext = bootstrap.extract_sample(1, f_ciphertext)

        self.assertEqual(
            lwe.lwe_decode(lwe.lwe_decrypt(sample_ciphertext, lwe_key)), 2
        )

    def test_bootstrap_to_zero(self):
        lwe_key = lwe.generate_lwe_key(config.LWE_CONFIG)
        gsw_key = gsw.convert_lwe_key_to_gsw(lwe_key, config.GSW_CONFIG)
        bootstrap_key = bootstrap.generate_bootstrap_key(lwe_key, gsw_key)

        plaintext = lwe.lwe_encode(1)
        ciphertext = lwe.lwe_encrypt(plaintext, lwe_key)

        bootstrap_ciphertext = bootstrap.bootstrap(
            ciphertext, bootstrap_key, scale=utils.encode(2)
        )

        self.assertEqual(
            lwe.lwe_decode(lwe.lwe_decrypt(bootstrap_ciphertext, lwe_key)), 0
        )

    def test_bootstrap_to_scale(self):
        lwe_key = lwe.generate_lwe_key(config.LWE_CONFIG)
        gsw_key = gsw.convert_lwe_key_to_gsw(lwe_key, config.GSW_CONFIG)
        bootstrap_key = bootstrap.generate_bootstrap_key(lwe_key, gsw_key)

        plaintext = lwe.lwe_encode(-3)
        ciphertext = lwe.lwe_encrypt(plaintext, lwe_key)

        bootstrap_ciphertext = bootstrap.bootstrap(
            ciphertext, bootstrap_key, scale=utils.encode(2)
        )

        self.assertEqual(
            lwe.lwe_decode(lwe.lwe_decrypt(bootstrap_ciphertext, lwe_key)), 2
        )


if __name__ == "__main__":
    unittest.main()
