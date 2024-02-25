import unittest
from tfhe import lwe
from tfhe import config


class TestLwe(unittest.TestCase):
    def test_encode_decode(self):
        self.assertEqual(lwe.decode(lwe.encode(2)), 2)

    def test_encrypt_decrypt(self):
        key = lwe.generate_lwe_key(config.LWE_CONFIG)

        plaintext = lwe.encode(-1)
        ciphertext = lwe.lwe_encrypt(plaintext, key)

        self.assertEqual(lwe.decode(lwe.lwe_decrypt(ciphertext, key)), -1)

    def test_add(self):
        key = lwe.generate_lwe_key(config.LWE_CONFIG)

        plaintext_a = lwe.encode(3)
        plaintext_b = lwe.encode(-1)

        ciphertext_a = lwe.lwe_encrypt(plaintext_a, key)
        ciphertext_b = lwe.lwe_encrypt(plaintext_b, key)

        ciphertext_sum = lwe.lwe_add(ciphertext_a, ciphertext_b)

        self.assertEqual(lwe.decode(lwe.lwe_decrypt(ciphertext_sum, key)), 2)

    def test_subtract(self):
        key = lwe.generate_lwe_key(config.LWE_CONFIG)

        plaintext_a = lwe.encode(2)
        plaintext_b = lwe.encode(3)

        ciphertext_a = lwe.lwe_encrypt(plaintext_a, key)
        ciphertext_b = lwe.lwe_encrypt(plaintext_b, key)

        ciphertext_sum = lwe.lwe_subtract(ciphertext_a, ciphertext_b)

        self.assertEqual(lwe.decode(lwe.lwe_decrypt(ciphertext_sum, key)), -1)

    def test_plaintext_multiply(self):
        key = lwe.generate_lwe_key(config.LWE_CONFIG)
        plaintext = lwe.encode(2)
        ciphertext = lwe.lwe_encrypt(plaintext, key)

        ciphertext_mul = lwe.lwe_plaintext_multiply(2, ciphertext)

        self.assertEqual(lwe.decode(lwe.lwe_decrypt(ciphertext_mul, key)), -4)


if __name__ == "__main__":
    unittest.main()
