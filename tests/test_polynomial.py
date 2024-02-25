import unittest
import numpy as np
from tfhe import polynomial


class TestPolynomial(unittest.TestCase):
    def assert_polynomial_equals(self, p_left, p_right):
        self.assertEqual(p_left.N, p_right.N)
        self.assertTrue(np.all(p_left.coeff == p_right.coeff))

    def test_polynomial_constant_multiply(self):
        # p = 1 + 2x + 3x^2 + 4x^3
        p = polynomial.Polynomial(
            N=4, coeff=np.array([1, 2, 3, 4], dtype=np.int32)
        )

        # 2 * p = 2 + 4x + 6x^2 + 6x^3
        p_times_2 = polynomial.Polynomial(
            N=4, coeff=np.array([2, 4, 6, 8], dtype=np.int32)
        )

        self.assert_polynomial_equals(
            polynomial.polynomial_constant_multiply(2, p), p_times_2
        )

    def test_polynomial_multiply(self):
        # p_0 = 1 + 2x + 3x^2 + 4x^3
        p_0 = polynomial.Polynomial(
            N=4, coeff=np.array([1, 2, 3, 4], dtype=np.int32)
        )

        # p_1 = x + 2x^3
        p_1 = polynomial.Polynomial(
            N=4, coeff=np.array([0, 1, 0, 2], dtype=np.int32)
        )

        # p_0 * p_1 =  (x + 2x^2 + 3x^3 + 4x^4) +
        #             2(x^3 + 2x^4 + 3x^5 + 4x^6)
        #           =  (-4 + x +  2x^2 + 3x^3) +
        #             2(-2 - 3x - 4x^2 + x^3)
        #           =   -8 - 5x - 6x^2 + 5x^3
        p_mul = polynomial.Polynomial(
            N=4, coeff=np.array([-8, -5, -6, 5], dtype=np.int32)
        )

        self.assert_polynomial_equals(
            polynomial.polynomial_multiply(p_0, p_1), p_mul
        )

    def test_polynomial_add(self):
        # p_0 = 1 + 2x + 3x^2 + 4x^3
        p_0 = polynomial.Polynomial(
            N=4, coeff=np.array([1, 2, 3, 4], dtype=np.int32)
        )

        # p_1 = x + 2x^3
        p_1 = polynomial.Polynomial(
            N=4, coeff=np.array([0, 1, 0, 2], dtype=np.int32)
        )

        p_sum = polynomial.Polynomial(
            N=4, coeff=np.array([1, 3, 3, 6], dtype=np.int32)
        )

        self.assert_polynomial_equals(
            polynomial.polynomial_add(p_0, p_1), p_sum
        )

    def test_polynomial_subtract(self):
        # p_0 = 1 + 2x + 3x^2 + 4x^3
        p_0 = polynomial.Polynomial(
            N=4, coeff=np.array([1, 2, 3, 4], dtype=np.int32)
        )

        # p_1 = x + 2x^3
        p_1 = polynomial.Polynomial(
            N=4, coeff=np.array([0, 1, 0, 2], dtype=np.int32)
        )

        p_diff = polynomial.Polynomial(
            N=4, coeff=np.array([1, 1, 3, 2], dtype=np.int32)
        )

        self.assert_polynomial_equals(
            polynomial.polynomial_subtract(p_0, p_1), p_diff
        )

    def test_zero_polynomial(self):
        p_0 = polynomial.Polynomial(
            N=4, coeff=np.array([0, 0, 0, 0], dtype=np.int32)
        )

        self.assert_polynomial_equals(polynomial.zero_polynomial(4), p_0)

    def test_build_monomial_in_range(self):
        # 3x^2
        monomial = polynomial.Polynomial(
            N=4, coeff=np.array([0, 0, 3, 0], dtype=np.int32)
        )

        self.assert_polynomial_equals(
            polynomial.build_monomial(3, 2, 4), monomial
        )

    def test_build_monomial_out_of_range_even(self):
        # 3x^10 = 3 * (x^4)^2 * x^2 = 3x^2
        monomial = polynomial.Polynomial(
            N=4, coeff=np.array([0, 0, 3, 0], dtype=np.int32)
        )

        self.assert_polynomial_equals(
            polynomial.build_monomial(3, 10, 4), monomial
        )

    def test_build_monomial_out_of_range_even(self):
        # 3x^15 = 3 * (x^4)^3 * x^3 = -3x^3
        monomial = polynomial.Polynomial(
            N=4, coeff=np.array([0, 0, 0, -3], dtype=np.int32)
        )

        self.assert_polynomial_equals(
            polynomial.build_monomial(3, 15, 4), monomial
        )


if __name__ == "__main__":
    unittest.main()
