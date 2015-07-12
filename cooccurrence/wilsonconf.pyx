from math import sqrt
from scipy.stats import norm

def check_conf(freq_pair, freq_one, freq_two, z, z2, eff_sample_size, strict=True):
    if freq_pair == 0 or freq_one == 0  or freq_two == 0:
        return False
    cdef float n_one = freq_one * eff_sample_size
    cdef float n_two = freq_two * eff_sample_size
    cdef float n_pair = freq_pair * eff_sample_size
    cdef float check_one = _wilson_score_lower(n_pair, n_one, z, z2)
    check_one /= _wilson_score_upper(n_two, eff_sample_size, z, z2)
    cdef float check_two = _wilson_score_lower(n_pair, n_two, z, z2)
    check_two /= _wilson_score_upper(n_one, eff_sample_size, z, z2)
    if strict:
        conf = check_one > 1 and check_two > 1
    else:
        conf = check_one > 1 or check_two > 1
    if conf and (freq_pair / (freq_one * freq_two)) < 1:
        print check_one, check_two, n_one, n_two, n_pair
    return conf

def get_interval(num_pos, sample_size, alpha):
    z = norm.ppf(1 - alpha / 2.0)
    z2 = z ** 2.0 
    sample_size = float(sample_size)
    return _wilson_score_lower(num_pos, sample_size, z, z2), _wilson_score_upper(num_pos, sample_size, z, z2)

# Agresti-Coull interval, see
# https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval
cdef float _wilson_score_lower(float count, float sample_size, float z, float z2):
    cdef float inv_n = 1.0 / sample_size
    cdef float p_tilde = count / sample_size
    return (1.0 / (1 + inv_n * z2)) * (p_tilde + inv_n * 0.5 * z2 - z * sqrt(inv_n * p_tilde * (1-p_tilde) + 0.25 * inv_n * inv_n * z2)) 

cdef float _wilson_score_upper(float count, float sample_size, float z, float z2):
    cdef float inv_n = 1.0 / sample_size
    cdef float p_tilde = count / sample_size
    return (1.0 / (1 + inv_n * z2)) * (p_tilde + inv_n * 0.5 * z2 + z * sqrt(inv_n * p_tilde * (1-p_tilde) + 0.25 * inv_n * inv_n * z2)) 
