"""SNR selection and expression scaling for the GeneSPIDER benchmark."""

import math

import numpy as np

from bench.snr_benchmark import (
    match_snr,
    noise_var_for_snr,
    parse_snrs,
    simulate_expression,
    snr_l,
)


def test_parse_snrs_keeps_the_requested_values():
    assert parse_snrs("0.01,0.1,1") == [0.01, 0.1, 1.0]


def test_noise_variance_hits_the_requested_snr():
    rng = np.random.default_rng(0)
    A = np.diag(rng.uniform(-1.8, -0.8, size=8))
    A[np.arange(7), np.arange(1, 8)] = 0.25
    P = np.zeros((8, 16))
    P[np.arange(8), np.arange(8)] = -1.0
    P[np.arange(8), np.arange(8, 16)] = -1.0
    variance = noise_var_for_snr(A, P, 0.1)
    assert math.isclose(snr_l(A, P, variance), 0.1, rel_tol=1e-9)


def test_simulated_expression_uses_that_noise_scale():
    rng = np.random.default_rng(1)
    A = -np.eye(20)
    P = np.repeat(-np.eye(20), 40, axis=1)
    Y = simulate_expression(A, P, 0.5, rng)
    assert Y.shape == P.shape
    residual = Y - ((-np.linalg.pinv(A)) @ P)
    measured = float(np.var(residual))
    assert math.isclose(measured, noise_var_for_snr(A, P, 0.5), rel_tol=0.1)


def test_match_prefers_snr_l_and_can_fall_back_to_the_filename():
    files = [
        {"snr_l": 0.01, "filename_snr": 10, "name": "low"},
        {"snr_l": 1.0, "filename_snr": 1000, "name": "mid"},
        {"snr_l": 100.0, "filename_snr": 100000, "name": "high"},
    ]
    assert match_snr(0.01, files)["name"] == "low"
    assert match_snr(1, files)["name"] == "mid"
    assert match_snr(10, files)["name"] == "low"
    assert match_snr(0.1, files) is None
