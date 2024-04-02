import unittest

import numpy as np

from tfhe import config, lwe, polynomial, rlwe, utils


class TestRlwe(unittest.TestCase):
    def assert_polynomial_equal(self, p_left, p_right):
        self.assertEqual(p_left.N, p_right.N)
        self.assertTrue(np.all(p_left.coeff == p_right.coeff))

    def test_encode_rlwe(self):
        config = rlwe.RlweConfig(degree=4, noise_std=0.1)
        p = polynomial.Polynomial(
            N=4, coeff=np.array([0, 1, 2, 3], dtype=np.int32)
        )
        p_encoded = polynomial.Polynomial(
            N=4,
            coeff=np.array(
                [
                    utils.encode(0),
                    utils.encode(1),
                    utils.encode(2),
                    utils.encode(3),
                ],
                dtype=np.int32,
            ),
        )

        plaintext = rlwe.rlwe_encode(p, config)

        self.assertEqual(plaintext.config, config)
        self.assert_polynomial_equal(plaintext.message, p_encoded)

    def test_decode_rlwe(self):
        config = rlwe.RlweConfig(degree=4, noise_std=0.1)
        p = polynomial.Polynomial(
            N=4, coeff=np.array([0, 1, 2, 3], dtype=np.int32)
        )
        plaintext = rlwe.RlwePlaintext(
            config=config,
            message=polynomial.Polynomial(
                N=4,
                coeff=np.array(
                    [
                        utils.encode(0),
                        utils.encode(1),
                        utils.encode(2),
                        utils.encode(3),
                    ],
                    dtype=np.int32,
                ),
            ),
        )

        p_decoded = rlwe.rlwe_decode(plaintext)

        self.assert_polynomial_equal(p_decoded, p)

    def test_build_zero_rlwe_plaintext(self):
        config = rlwe.RlweConfig(degree=4, noise_std=0.1)
        p = polynomial.Polynomial(
            N=4, coeff=np.array([0, 0, 0, 0], dtype=np.int32)
        )

        plaintext = rlwe.build_zero_rlwe_plaintext(config)

        self.assertEqual(plaintext.config, config)
        self.assert_polynomial_equal(plaintext.message, p)

    def test_build_monomial_rlwe_plaintext(self):
        config = rlwe.RlweConfig(degree=4, noise_std=0.1)
        monomial = polynomial.Polynomial(
            N=4, coeff=np.array([0, 2, 0, 0], dtype=np.int32)
        )

        plaintext = rlwe.build_monomial_rlwe_plaintext(2, 1, config)

        self.assertEqual(plaintext.config, config)
        self.assert_polynomial_equal(plaintext.message, monomial)

    def test_convert_lwe_key_to_rlwe(self):
        lwe_config = lwe.LweConfig(dimension=4, noise_std=0.1)
        rlwe_config = rlwe.RlweConfig(degree=4, noise_std=0.1)
        lwe_key = lwe.LweEncryptionKey(
            config=lwe_config, key=np.array([1, 0, 1, 1], dtype=np.int32)
        )

        expected_rlwe_key = rlwe.RlweEncryptionKey(
            config=rlwe_config,
            key=polynomial.Polynomial(
                N=4, coeff=np.array([1, 0, 1, 1], dtype=np.int32)
            ),
        )

        rlwe_key = rlwe.convert_lwe_key_to_rlwe(lwe_key)

        self.assertEqual(rlwe_key.config, expected_rlwe_key.config)
        self.assert_polynomial_equal(rlwe_key.key, expected_rlwe_key.key)

    def test_encrypt_decrypt(self):
        key = rlwe.generate_rlwe_key(config.RLWE_CONFIG)
        p = polynomial.build_monomial(c=1, i=1, N=config.RLWE_CONFIG.degree)

        plaintext = rlwe.rlwe_encode(p, config)
        ciphertext = rlwe.rlwe_encrypt(plaintext, key)

        self.assert_polynomial_equal(
            rlwe.rlwe_decode(rlwe.rlwe_decrypt(ciphertext, key)), p
        )

    def test_add(self):
        key = rlwe.generate_rlwe_key(config.RLWE_CONFIG)

        p_0 = polynomial.build_monomial(c=1, i=0, N=config.RLWE_CONFIG.degree)
        p_1 = polynomial.build_monomial(c=2, i=1, N=config.RLWE_CONFIG.degree)

        plaintext_0 = rlwe.rlwe_encode(p_0, config)
        plaintext_1 = rlwe.rlwe_encode(p_1, config)

        ciphertext_0 = rlwe.rlwe_encrypt(plaintext_0, key)
        ciphertext_1 = rlwe.rlwe_encrypt(plaintext_1, key)

        ciphertext_sum = rlwe.rlwe_add(ciphertext_0, ciphertext_1)

        self.assert_polynomial_equal(
            rlwe.rlwe_decode(rlwe.rlwe_decrypt(ciphertext_sum, key)),
            polynomial.polynomial_add(p_0, p_1),
        )

    def test_subtract(self):
        key = rlwe.generate_rlwe_key(config.RLWE_CONFIG)

        p_0 = polynomial.build_monomial(c=1, i=0, N=config.RLWE_CONFIG.degree)
        p_1 = polynomial.build_monomial(c=2, i=1, N=config.RLWE_CONFIG.degree)

        plaintext_0 = rlwe.rlwe_encode(p_0, config)
        plaintext_1 = rlwe.rlwe_encode(p_1, config)

        ciphertext_0 = rlwe.rlwe_encrypt(plaintext_0, key)
        ciphertext_1 = rlwe.rlwe_encrypt(plaintext_1, key)

        ciphertext_diff = rlwe.rlwe_subtract(ciphertext_0, ciphertext_1)

        self.assert_polynomial_equal(
            rlwe.rlwe_decode(rlwe.rlwe_decrypt(ciphertext_diff, key)),
            polynomial.polynomial_subtract(p_0, p_1),
        )

    def test_plaintext_multiply(self):
        key = rlwe.generate_rlwe_key(config.RLWE_CONFIG)

        p_0 = polynomial.build_monomial(c=1, i=0, N=config.RLWE_CONFIG.degree)
        p_1 = polynomial.build_monomial(c=2, i=1, N=config.RLWE_CONFIG.degree)

        plaintext_0 = rlwe.RlwePlaintext(config=config.RLWE_CONFIG, message=p_0)
        plaintext_1 = rlwe.rlwe_encode(p_1, config)

        ciphertext_1 = rlwe.rlwe_encrypt(plaintext_1, key)

        ciphertext_prod = rlwe.rlwe_plaintext_multiply(
            plaintext_0, ciphertext_1
        )

        self.assert_polynomial_equal(
            rlwe.rlwe_decode(rlwe.rlwe_decrypt(ciphertext_prod, key)),
            polynomial.polynomial_multiply(p_0, p_1),
        )

    def test_rlwe_trivial_ciphertext(self):
        key = rlwe.generate_rlwe_key(config.RLWE_CONFIG)

        f = polynomial.build_monomial(c=2, i=1, N=config.RLWE_CONFIG.degree)
        plaintext = rlwe.rlwe_encode(f, config.RLWE_CONFIG)
        ciphertext = rlwe.rlwe_trivial_ciphertext(
            plaintext.message, config.RLWE_CONFIG
        )

        self.assert_polynomial_equal(
            rlwe.rlwe_decode(rlwe.rlwe_decrypt(ciphertext, key)), f
        )


if __name__ == "__main__":
    unittest.main()
