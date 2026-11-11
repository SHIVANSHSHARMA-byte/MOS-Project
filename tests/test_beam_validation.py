import math
import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.beam_analysis import (
    BeamConfig,
    PointLoad,
    UVL,
    calculate_reactions,
    shear_force,
    bending_moment,
    uvl_resultant,
    validate_beam,
    sample_beam,
)
from src.beam_plotter import find_bending_moment_extrema
from src.beam_validation import (
    independent_uvl_resultant,
    independent_calculate_reactions,
    verify_static_equilibrium,
    analytical_central_point_load,
    analytical_full_span_udl,
    analytical_triangular_uvl,
)

class TestIndependentEquilibrium:

    def test_independent_reactions_match_engine_sample_beam(self):
        config = sample_beam()
        R_A_eng, R_B_eng = calculate_reactions(config)
        R_A_ind, R_B_ind = independent_calculate_reactions(config)

        assert pytest.approx(R_A_eng, abs=1e-10) == R_A_ind
        assert pytest.approx(R_B_eng, abs=1e-10) == R_B_ind

        audit = verify_static_equilibrium(config, R_A_eng, R_B_eng, tol=1e-9)
        assert audit["is_balanced"] is True
        assert abs(audit["residual_vertical"]) < 1e-9
        assert abs(audit["residual_moment_A"]) < 1e-9
        assert abs(audit["residual_moment_B"]) < 1e-9

    def test_uvl_resultant_and_centroid_zero_intensity(self):
        R_eng, x_bar_eng = uvl_resultant(UVL(a=2.0, b=6.0, w_a=0.0, w_b=0.0))
        R_ind, x_bar_ind = independent_uvl_resultant(2.0, 6.0, 0.0, 0.0)

        assert R_eng == 0.0
        assert R_ind == 0.0
        assert x_bar_eng == 4.0
        assert x_bar_ind == 4.0

    def test_uvl_resultant_and_centroid_triangular(self):
        R_ind, x_bar_ind = independent_uvl_resultant(0.0, 9.0, 0.0, 30.0)
        assert pytest.approx(R_ind, abs=1e-12) == 135.0
        assert pytest.approx(x_bar_ind, abs=1e-12) == 6.0


class TestAnalyticalBenchmarks:

    def test_case_A_central_point_load(self):
        L = 10.0
        P = 100.0
        expected = analytical_central_point_load(L, P)

        config = BeamConfig(
            length=L,
            point_loads=[
                PointLoad(magnitude=0.0, position=1.0),
                PointLoad(magnitude=P, position=5.0),
                PointLoad(magnitude=0.0, position=9.0),
            ],
            uvl=None,
        )

        R_A, R_B = calculate_reactions(config)
        assert pytest.approx(R_A, abs=1e-9) == expected["R_A"]
        assert pytest.approx(R_B, abs=1e-9) == expected["R_B"]

        M_mid = bending_moment(config, 5.0)
        assert pytest.approx(M_mid, abs=1e-9) == expected["M_max"]
       
        V_left = shear_force(config, 5.0 - 1e-9)
        V_right = shear_force(config, 5.0 + 1e-9)
        assert pytest.approx(V_left, abs=1e-7) == expected["V_left_mid"]
        assert pytest.approx(V_right, abs=1e-7) == expected["V_right_mid"]

        extrema = find_bending_moment_extrema(config)
        assert pytest.approx(extrema["max_positive_value"], abs=1e-7) == expected["M_max"]
        assert pytest.approx(extrema["max_positive_pos"], abs=1e-5) == expected["x_M_max"]

    def test_case_B_full_span_udl(self):
        L = 8.0
        w0 = 20.0
        expected = analytical_full_span_udl(L, w0)

        config = BeamConfig(
            length=L,
            point_loads=[
                PointLoad(magnitude=0.0, position=1.0),
                PointLoad(magnitude=0.0, position=4.0),
                PointLoad(magnitude=0.0, position=7.0),
            ],
            uvl=UVL(a=0.0, b=L, w_a=w0, w_b=w0),
        )

        R_A, R_B = calculate_reactions(config)
        assert pytest.approx(R_A, abs=1e-9) == expected["R_A"]
        assert pytest.approx(R_B, abs=1e-9) == expected["R_B"]

        M_mid = bending_moment(config, 4.0)
        assert pytest.approx(M_mid, abs=1e-9) == expected["M_max"]
       
        extrema = find_bending_moment_extrema(config)
        assert pytest.approx(extrema["max_positive_value"], abs=1e-7) == expected["M_max"]
        assert pytest.approx(extrema["max_positive_pos"], abs=1e-6) == expected["x_M_max"]

    def test_case_C_triangular_uvl(self):
        L = 6.0
        w0 = 36.0
        expected = analytical_triangular_uvl(L, w0)

        config = BeamConfig(
            length=L,
            point_loads=[
                PointLoad(magnitude=0.0, position=1.0),
                PointLoad(magnitude=0.0, position=2.0),
                PointLoad(magnitude=0.0, position=5.0),
            ],
            uvl=UVL(a=0.0, b=L, w_a=0.0, w_b=w0),
        )

        R_A, R_B = calculate_reactions(config)

        assert pytest.approx(R_A, abs=1e-9) == expected["R_A"]
        assert pytest.approx(R_B, abs=1e-9) == expected["R_B"]

        x_expected = expected["x_M_max"]
        V_at_root = shear_force(config, x_expected)
        assert pytest.approx(V_at_root, abs=1e-6) == 0.0

        M_max_analytical = expected["M_max"]
        M_at_root = bending_moment(config, x_expected)
        assert pytest.approx(M_at_root, abs=1e-7) == M_max_analytical

        extrema = find_bending_moment_extrema(config)
        assert pytest.approx(extrema["max_positive_pos"], abs=1e-5) == x_expected
        assert pytest.approx(extrema["max_positive_value"], abs=1e-6) == M_max_analytical

    def test_case_D_asymmetric_mixed_loading(self):
        config = BeamConfig(
            length=12.0,
            point_loads=[
                PointLoad(magnitude=18.0, position=2.5),
                PointLoad(magnitude=42.0, position=6.0),
                PointLoad(magnitude=24.0, position=10.0),
            ],
            uvl=UVL(a=4.0, b=9.0, w_a=10.0, w_b=35.0),
        )
        validate_beam(config)

        R_A_ind, R_B_ind = independent_calculate_reactions(config)
        R_A_eng, R_B_eng = calculate_reactions(config)
        assert pytest.approx(R_A_eng, abs=1e-10) == R_A_ind
        assert pytest.approx(R_B_eng, abs=1e-10) == R_B_ind

        audit = verify_static_equilibrium(config, R_A_eng, R_B_eng)
        assert audit["is_balanced"] is True



class TestCalculusAndConsistency:

    def test_differential_relation_dM_dx_equals_V(self):
        config = sample_beam()
        test_stations = [1.0, 4.0, 6.0, 9.0]
        h = 1e-6

        for x in test_stations:
            M_plus = bending_moment(config, x + h)
            M_minus = bending_moment(config, x - h)
            dM_dx_num = (M_plus - M_minus) / (2.0 * h)
            V_exact = shear_force(config, x)

            assert pytest.approx(dM_dx_num, abs=1e-4) == V_exact

    def test_shear_integral_across_uvl(self):
        config = sample_beam()
        x1, x2 = 3.1, 4.9
        V1 = shear_force(config, x1)
        V2 = shear_force(config, x2)
        delta_V = V2 - V1
       
        expected_load_integral = 13.5
        assert pytest.approx(delta_V, abs=1e-8) == -expected_load_integral

    def test_shear_jump_at_concentrated_loads(self):
        config = sample_beam()
        _EPS = 1e-9
        for pl in config.point_loads:
            V_left = shear_force(config, pl.position - _EPS)
            V_right = shear_force(config, pl.position + _EPS)
            jump = V_left - V_right
            assert pytest.approx(jump, abs=1e-6) == pl.magnitude

    def test_bending_moment_continuity_at_concentrated_loads(self):
        config = sample_beam()
        _EPS = 1e-9
        for pl in config.point_loads:
            M_left = bending_moment(config, pl.position - _EPS)
            M_right = bending_moment(config, pl.position + _EPS)
            assert pytest.approx(M_left, abs=1e-6) == M_right


class TestBoundaryAndEdgeCases:

    def test_loads_at_exact_supports(self):
        config = BeamConfig(
            length=10.0,
            point_loads=[
                PointLoad(magnitude=50.0, position=0.0),
                PointLoad(magnitude=30.0, position=5.0),
                PointLoad(magnitude=40.0, position=10.0),
            ],
            uvl=None,
        )
        validate_beam(config)
        R_A, R_B = calculate_reactions(config)
        assert pytest.approx(R_B, abs=1e-9) == 55.0
        assert pytest.approx(R_A, abs=1e-9) == 65.0
        assert bending_moment(config, 0.0) == 0.0
        assert bending_moment(config, 10.0) == 0.0

    def test_multiple_loads_at_same_position(self):
        config = BeamConfig(
            length=10.0,
            point_loads=[
                PointLoad(magnitude=20.0, position=4.0),
                PointLoad(magnitude=30.0, position=4.0),
                PointLoad(magnitude=15.0, position=8.0),
            ],
            uvl=None,
        )
        validate_beam(config)
        R_A, R_B = calculate_reactions(config)
        assert pytest.approx(R_B, abs=1e-9) == 32.0
        assert pytest.approx(R_A, abs=1e-9) == 33.0
        V_left = shear_force(config, 4.0 - 1e-9)
        V_right = shear_force(config, 4.0 + 1e-9)
        assert pytest.approx(V_left - V_right, abs=1e-6) == 50.0

    def test_extremely_short_beam(self):
        config = BeamConfig(
            length=0.05,
            point_loads=[
                PointLoad(magnitude=10.0, position=0.01),
                PointLoad(magnitude=20.0, position=0.025),
                PointLoad(magnitude=10.0, position=0.04),
            ],
            uvl=None,
        )
        validate_beam(config)
        R_A, R_B = calculate_reactions(config)
        assert pytest.approx(R_A, abs=1e-9) == 20.0
        assert pytest.approx(R_B, abs=1e-9) == 20.0
        assert pytest.approx(bending_moment(config, 0.0), abs=1e-9) == 0.0
        assert pytest.approx(bending_moment(config, 0.05), abs=1e-9) == 0.0

    def test_very_large_load_magnitudes(self):
        config = BeamConfig(
            length=50.0,
            point_loads=[
                PointLoad(magnitude=100000.0, position=10.0),
                PointLoad(magnitude=200000.0, position=25.0),
                PointLoad(magnitude=100000.0, position=40.0),
            ],
            uvl=UVL(a=0.0, b=50.0, w_a=5000.0, w_b=5000.0),
        )
        validate_beam(config)
        R_A, R_B = calculate_reactions(config)
        assert pytest.approx(R_A, abs=1e-6) == 325000.0
        assert pytest.approx(R_B, abs=1e-6) == 325000.0

    def test_non_finite_inputs_raise_value_error(self):
        """Verify that NaN and Inf are rejected by validate_beam."""
        with pytest.raises(ValueError, match="finite"):
            validate_beam(BeamConfig(length=float("nan"), point_loads=[
                PointLoad(10.0, 1.0), PointLoad(10.0, 2.0), PointLoad(10.0, 3.0)
            ]))

        with pytest.raises(ValueError, match="finite"):
            validate_beam(BeamConfig(length=10.0, point_loads=[
                PointLoad(10.0, 1.0), PointLoad(10.0, float("inf")), PointLoad(10.0, 3.0)
            ]))

        with pytest.raises(ValueError, match="non-negative"):
            validate_beam(BeamConfig(length=10.0, point_loads=[
                PointLoad(10.0, 1.0), PointLoad(-5.0, 2.0), PointLoad(10.0, 3.0)
            ]))
