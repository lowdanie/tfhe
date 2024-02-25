from tfhe import gsw
from tfhe import lwe
from tfhe import rlwe

LWE_CONFIG = lwe.LweConfig(dimension=1024, noise_std=2 ** (-24))

RLWE_CONFIG = rlwe.RlweConfig(degree=1024, noise_std=2 ** (-24))

GSW_CONFIG = gsw.GswConfig(rlwe_config=RLWE_CONFIG, log_p=8)
