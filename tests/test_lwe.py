import unittest

from tfhe import lwe
from tfhe import config
from tfhe import utils


class TestLwe(unittest.TestCase):
    def test_encode_decode(self):
        self.assertEqual(lwe.lwe_decode(lwe.lwe_encode(2)), 2)

    def test_encrypt_decrypt(self):
        key = lwe.generate_lwe_key(config.LWE_CONFIG)

        plaintext = lwe.lwe_encode(-1)
        ciphertext = lwe.lwe_encrypt(plaintext, key)

        self.assertEqual(lwe.lwe_decode(lwe.lwe_decrypt(ciphertext, key)), -1)

    def test_lwe_trivial_ciphertext(self):
        key = lwe.generate_lwe_key(config.LWE_CONFIG)

        plaintext = lwe.lwe_encode(1)
        ciphertext = lwe.lwe_trivial_ciphertext(plaintext, config.LWE_CONFIG)

        self.assertEqual(lwe.lwe_decode(lwe.lwe_decrypt(ciphertext, key)), 1)

    def test_add(self):
        key = lwe.generate_lwe_key(config.LWE_CONFIG)

        plaintext_a = lwe.lwe_encode(3)
        plaintext_b = lwe.lwe_encode(-1)

        ciphertext_a = lwe.lwe_encrypt(plaintext_a, key)
        ciphertext_b = lwe.lwe_encrypt(plaintext_b, key)

        ciphertext_sum = lwe.lwe_add(ciphertext_a, ciphertext_b)

        self.assertEqual(
            lwe.lwe_decode(lwe.lwe_decrypt(ciphertext_sum, key)), 2
        )

    def test_subtract(self):
        key = lwe.generate_lwe_key(config.LWE_CONFIG)

        plaintext_a = lwe.lwe_encode(2)
        plaintext_b = lwe.lwe_encode(3)

        ciphertext_a = lwe.lwe_encrypt(plaintext_a, key)
        ciphertext_b = lwe.lwe_encrypt(plaintext_b, key)

        ciphertext_sum = lwe.lwe_subtract(ciphertext_a, ciphertext_b)

        self.assertEqual(
            lwe.lwe_decode(lwe.lwe_decrypt(ciphertext_sum, key)), -1
        )

    def test_plaintext_multiply(self):
        key = lwe.generate_lwe_key(config.LWE_CONFIG)
        plaintext = lwe.lwe_encode(2)
        ciphertext = lwe.lwe_encrypt(plaintext, key)

        ciphertext_mul = lwe.lwe_plaintext_multiply(2, ciphertext)

        pt = lwe.lwe_decrypt(ciphertext_mul, key)
        print(f"Decode {pt.message}: {utils.decode(pt.message)}")
        print(f"Decode {pt}: {lwe.lwe_decode(pt)}")

        self.assertEqual(
            lwe.lwe_decode(lwe.lwe_decrypt(ciphertext_mul, key)), -4
        )


if __name__ == "__main__":
    unittest.main()
