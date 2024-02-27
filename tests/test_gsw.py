import numpy as np
import unittest

from tfhe import config
from tfhe import gsw
from tfhe import polynomial
from tfhe import rlwe


class TestGsw(unittest.TestCase):
    # TODO: Put in utils
    def assert_polynomial_equal(self, p_left, p_right):
        self.assertEqual(p_left.N, p_right.N)
        self.assertTrue(np.all(p_left.coeff == p_right.coeff))

    def test_array_to_base_p(self):
        log_p = 8
        array = np.random.randint(-(2**31), 2**31 - 1, size=100, dtype=np.int32)

        array_base_p = gsw.array_to_base_p(array, log_p)

        # Check that all of the elements in array_base_p are between -256 and
        # 256
        for level in array_base_p:
            self.assertTrue(np.all(level < 256))
            self.assertTrue(np.all(level >= -256))

        # Check that the base-p representation can be converted back
        # to the original.
        self.assertTrue(
            np.all(gsw.base_p_to_array(array_base_p, log_p) == array)
        )

    def test_polynomial_to_base_p(self):
        log_p = 8
        f = polynomial.Polynomial(
            N=64,
            coeff=np.random.randint(
                -(2**31), 2**31 - 1, size=64, dtype=np.int32
            ),
        )

        f_base_p = gsw.polynomial_to_base_p(f, log_p)

        # Check that all of the elements in array_base_p are between -256 and
        # 256
        for level in f_base_p:
            self.assertTrue(np.all(level.coeff < 256))
            self.assertTrue(np.all(level.coeff >= -256))

        self.assert_polynomial_equal(
            gsw.base_p_to_polynomial(f_base_p, log_p), f
        )

    def test_gsw_multiply(self):
        rlwe_config = config.RLWE_CONFIG
        gsw_config = config.GSW_CONFIG

        rlwe_key = rlwe.generate_rlwe_key(rlwe_config)
        gsw_key = gsw.convert_rlwe_key_to_gsw(rlwe_key, gsw_config)

        f = polynomial.build_monomial(c=2, i=0, N=rlwe_config.degree)
        g = polynomial.build_monomial(c=1, i=1, N=rlwe_config.degree)

        gsw_plaintext = gsw.GswPlaintext(config=gsw_config, message=f)
        rlwe_plaintext = rlwe.rlwe_encode(g, rlwe_config)

        gsw_ciphertext = gsw.gsw_encrypt(gsw_plaintext, gsw_key)
        rlwe_ciphertext = rlwe.rlwe_encrypt(rlwe_plaintext, rlwe_key)

        rlwe_ciphertext_prod = gsw.gsw_multiply(gsw_ciphertext, rlwe_ciphertext)

        fg = polynomial.build_monomial(c=2, i=1, N=rlwe_config.degree)
        self.assert_polynomial_equal(
            rlwe.rlwe_decode(rlwe.rlwe_decrypt(rlwe_ciphertext_prod, rlwe_key)),
            fg,
        )

    def test_cmux(self):
        rlwe_config = config.RLWE_CONFIG
        gsw_config = config.GSW_CONFIG

        rlwe_key = rlwe.generate_rlwe_key(rlwe_config)
        gsw_key = gsw.convert_rlwe_key_to_gsw(rlwe_key, gsw_config)
        
        # Selector bit is 1
        selector = polynomial.build_monomial(c=1, i=0, N=rlwe_config.degree)
        line_0 = polynomial.build_monomial(c=1, i=1, N=rlwe_config.degree)
        line_1 = polynomial.build_monomial(c=2, i=1, N=rlwe_config.degree)

        selector_plaintext = gsw.GswPlaintext(
            config=gsw_config, message=selector
        )
        line_0_plaintext = rlwe.rlwe_encode(line_0, rlwe_config)
        line_1_plaintext = rlwe.rlwe_encode(line_1, rlwe_config)

        selector_ciphertext = gsw.gsw_encrypt(selector_plaintext, gsw_key)
        line_0_ciphertext = rlwe.rlwe_encrypt(line_0_plaintext, rlwe_key)
        line_1_ciphertext = rlwe.rlwe_encrypt(line_1_plaintext, rlwe_key)

        cmux_ciphertext = gsw.cmux(
            selector_ciphertext, line_0_ciphertext, line_1_ciphertext
        )

        self.assert_polynomial_equal(
            rlwe.rlwe_decode(rlwe.rlwe_decrypt(cmux_ciphertext, rlwe_key)),
            line_1,
        )


if __name__ == "__main__":
    unittest.main()
