import unittest

import numpy as np

from tfhe import bootstrap, config, gsw, lwe, nand


class TestNand(unittest.TestCase):
    def test_nand_gate(self):
        lwe_key = lwe.generate_lwe_key(config.LWE_CONFIG)
        gsw_key = gsw.convert_lwe_key_to_gsw(lwe_key, config.GSW_CONFIG)
        bootstrap_key = bootstrap.generate_bootstrap_key(lwe_key, gsw_key)

        plaintext_0 = lwe.lwe_encode_bool(False)
        plaintext_1 = lwe.lwe_encode_bool(True)

        ciphertext_0 = lwe.lwe_encrypt(plaintext_0, lwe_key)
        ciphertext_1 = lwe.lwe_encrypt(plaintext_1, lwe_key)

        ciphertext_nand = nand.lwe_nand(
            ciphertext_0, ciphertext_1, bootstrap_key
        )
        plaintext_nand = lwe.lwe_decrypt(ciphertext_nand, lwe_key)

        self.assertEqual(lwe.lwe_decode_bool(plaintext_nand), True)


if __name__ == "__main__":
    unittest.main()
