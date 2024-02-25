import numpy as np
import dataclasses


@dataclasses.dataclass
class Polynomial:
    N: int
    coeff: np.ndarray


def polynomial_constant_multiply(c: int, p: Polynomial) -> Polynomial:
    return Polynomial(N=p.N, coeff=np.multiply(c, p.coeff, dtype=np.int32))


def polynomial_multiply(p1: Polynomial, p2: Polynomial) -> Polynomial:
    N = p1.N

    # Multiply and pad the result to have length 2N-1
    prod = np.polymul(p1.coeff[::-1], p2.coeff[::-1])[::-1]
    prod_padded = np.zeros(2 * N - 1, dtype=np.int32)
    prod_padded[: len(prod)] = prod

    # Use the relation x^N = -1 to obtain a polynomial of degree N-1
    result = prod_padded[:N]
    result[:-1] -= prod_padded[N:]
    return Polynomial(N=N, coeff=result)


def polynomial_add(p1: Polynomial, p2: Polynomial) -> Polynomial:
    return Polynomial(N=p1.N, coeff=np.add(p1.coeff, p2.coeff, dtype=np.int32))


def polynomial_subtract(p1: Polynomial, p2: Polynomial) -> Polynomial:
    return Polynomial(
        N=p1.N, coeff=np.subtract(p1.coeff, p2.coeff, dtype=np.int32)
    )


def zero_polynomial(N: int) -> Polynomial:
    return Polynomial(N=N, coeff=np.zeros(N, dtype=np.int32))


def build_monomial(c: int, i: int, N: int) -> Polynomial:
    """Build a monomial c*x^i in the ring Z[x]/(x^N + 1)"""
    coeff = np.zeros(N, dtype=np.int32)

    # Find k such that: 0 <= i + k*N < N
    i_mod_N = i % N
    k = (i_mod_N - i) // N

    # If k is odd then the monomial picks up a negative sign since:
    # x^i = (-1)^k * x^(i + k*N) = (-1)^k * x^(i % N)
    sign = 1 if k % 2 == 0 else -1

    coeff[i_mod_N] = sign * c
    return Polynomial(N=N, coeff=coeff)
