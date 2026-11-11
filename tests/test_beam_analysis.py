import math
import sys
import os
import pytest


sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.beam_analysis import (
    BeamConfig,
    PointLoad,
    UVL,
    validate_beam,
    uvl_resultant,
    calculate_reactions,
    total_resultant_load,
    shear_force,
    bending_moment,
    sample_beam,
)

ATOL = 1e-9   


def make_beam(L, loads, uvl=None):
    """Convenience constructor."""
    return BeamConfig(length=L, point_loads=loads, uvl=uvl)



class TestThreeOrMorePointLoads:


    def test_exactly_three_point_loads_reactions(self):

        L = 8.0
        loads = [
            PointLoad(10.0, 2.0),
            PointLoad(20.0, 5.0),
            PointLoad(30.0, 7.0),
        ]
        config = make_beam(L, loads)
        validate_beam(config)
        R_A, R_B = calculate_reactions(config)

        assert abs(R_A - 18.75) < ATOL
        assert abs(R_B - 41.25) < ATOL

    def test_five_point_loads_reactions(self):

        L = 10.0
        loads = [
            PointLoad(5.0, 1.0),
            PointLoad(10.0, 3.0),
            PointLoad(15.0, 5.0),
            PointLoad(20.0, 7.0),
            PointLoad(25.0, 9.0),
        ]
        config = make_beam(L, loads)
        validate_beam(config)
        R_A, R_B = calculate_reactions(config)

        assert abs(R_A - 27.5) < ATOL
        assert abs(R_B - 47.5) < ATOL

    def test_total_resultant_no_uvl(self):
        loads = [PointLoad(10, 1), PointLoad(20, 3), PointLoad(30, 6)]
        config = make_beam(8.0, loads)
        assert abs(total_resultant_load(config) - 60.0) < ATOL

    def test_shear_between_loads(self):
        
        L = 6.0
        loads = [PointLoad(12, 2), PointLoad(12, 4), PointLoad(12, 6)]
        config = make_beam(L, loads)
        eps = 1e-9

        R_A, R_B = calculate_reactions(config)
        assert abs(R_A - 12.0) < ATOL
        assert abs(R_B - 24.0) < ATOL

        assert abs(shear_force(config, eps) - 12.0) < ATOL
        assert abs(shear_force(config, 3.0)) < ATOL          
        assert abs(shear_force(config, 5.0) - (-12.0)) < ATOL



class TestTriangularUVL:
    

    def test_triangle_ascending_resultant(self):
        
        uvl = UVL(a=2.0, b=6.0, w_a=0.0, w_b=20.0)
        R, x_bar = uvl_resultant(uvl)

        assert abs(R - 40.0) < ATOL
        assert abs(x_bar - (2.0 + 2.0 / 3.0 * 4.0)) < ATOL

    def test_triangle_descending_resultant(self):
        
        uvl = UVL(a=1.0, b=5.0, w_a=20.0, w_b=0.0)
        R, x_bar = uvl_resultant(uvl)

        assert abs(R - 40.0) < ATOL
        assert abs(x_bar - (1.0 + 1.0 / 3.0 * 4.0)) < ATOL

    def test_triangle_reactions(self):
        
        L = 10.0
        loads = [PointLoad(10, 2), PointLoad(10, 5), PointLoad(10, 8)]
        uvl = UVL(a=0.0, b=10.0, w_a=0.0, w_b=40.0)
        config = make_beam(L, loads, uvl)
        validate_beam(config)

        R_uvl, x_bar = uvl_resultant(uvl)
        assert abs(R_uvl - 200.0) < ATOL
        assert abs(x_bar - 20.0 / 3.0) < ATOL

        R_A, R_B = calculate_reactions(config)
        expected_moment = 10*2 + 10*5 + 10*8 + 200 * (20.0/3.0)
        expected_R_B = expected_moment / L
        expected_R_A = 230.0 - expected_R_B

        assert abs(R_A - expected_R_A) < ATOL
        assert abs(R_B - expected_R_B) < ATOL




class TestTrapezoidalUVL:
   
    def test_trapezoid_resultant(self):
        
        uvl = UVL(a=1.0, b=5.0, w_a=10.0, w_b=30.0)
        R, x_bar = uvl_resultant(uvl)

        assert abs(R - 80.0) < ATOL
        assert abs(x_bar - 10.0 / 3.0) < ATOL

    def test_trapezoid_reactions(self):
        
        L = 12.0
        loads = [PointLoad(15, 3), PointLoad(25, 6), PointLoad(10, 9)]
        uvl = UVL(a=2.0, b=10.0, w_a=8.0, w_b=24.0)
        config = make_beam(L, loads, uvl)
        validate_beam(config)

        R_A, R_B = calculate_reactions(config)

        x_bar = (8 * (2*2 + 10) + 24 * (2 + 2*10)) / (3 * (8 + 24))
        expected_moment = 15*3 + 25*6 + 10*9 + 128 * x_bar
        expected_R_B = expected_moment / L
        expected_R_A = 178.0 - expected_R_B

        assert abs(R_A - expected_R_A) < ATOL
        assert abs(R_B - expected_R_B) < ATOL

    def test_uniform_special_case(self):
        
        uvl = UVL(a=0.0, b=10.0, w_a=10.0, w_b=10.0)
        R, x_bar = uvl_resultant(uvl)

        assert abs(R - 100.0) < ATOL
        assert abs(x_bar - 5.0) < ATOL



class TestUVLFullSpan:
   

    def test_uniform_full_span_reactions_symmetric(self):

        L = 10.0
        loads = [PointLoad(0, 0), PointLoad(0, 5), PointLoad(0, L)] 
        uvl = UVL(a=0.0, b=L, w_a=20.0, w_b=20.0)
        config = make_beam(L, loads, uvl)
        validate_beam(config)

        R_A, R_B = calculate_reactions(config)
        assert abs(R_A - 100.0) < ATOL
        assert abs(R_B - 100.0) < ATOL

    def test_triangular_full_span_reactions(self):
        
        L = 6.0
        w0 = 12.0
        loads = [PointLoad(0, 0), PointLoad(0, 3), PointLoad(0, 6)]
        uvl = UVL(a=0.0, b=L, w_a=0.0, w_b=w0)
        config = make_beam(L, loads, uvl)
        validate_beam(config)

        R_A, R_B = calculate_reactions(config)
        assert abs(R_A - w0 * L / 6.0) < ATOL
        assert abs(R_B - w0 * L / 3.0) < ATOL



class TestUVLPartialSpan:
    

    def test_partial_uvl_shear_piecewise(self):

        L = 12.0
        loads = [PointLoad(0, 0), PointLoad(0, 6), PointLoad(0, 12)]
        uvl = UVL(a=3.0, b=9.0, w_a=10.0, w_b=10.0)
        config = make_beam(L, loads, uvl)
        validate_beam(config)

        R_A, R_B = calculate_reactions(config)
        assert abs(R_A - 30.0) < ATOL
       
        assert abs(shear_force(config, 1.0) - 30.0) < ATOL
        assert abs(shear_force(config, 6.0)) < ATOL
        assert abs(shear_force(config, 10.0) - (-30.0)) < ATOL

    def test_partial_uvl_moment_boundary(self):
        """M(0) == 0 and M(L) == 0 for the beam with partial UVL."""
        L = 12.0
        loads = [PointLoad(0, 0), PointLoad(0, 6), PointLoad(0, 12)]
        uvl = UVL(a=3.0, b=9.0, w_a=10.0, w_b=10.0)
        config = make_beam(L, loads, uvl)

        assert abs(bending_moment(config, 0.0)) < ATOL
        assert abs(bending_moment(config, L)) < ATOL



class TestBoundaryPositions:


    def test_loads_at_supports(self):

        L = 5.0
        loads = [
            PointLoad(20.0, 0.0),
            PointLoad(40.0, 3.0),
            PointLoad(20.0, 5.0),
        ]
        config = make_beam(L, loads)
        validate_beam(config)
        R_A, R_B = calculate_reactions(config)

        assert abs(R_A - 36.0) < ATOL
        assert abs(R_B - 44.0) < ATOL



class TestInvalidInputs:

    def test_non_positive_length_zero(self):
        config = make_beam(0.0, [PointLoad(10, 0), PointLoad(10, 0), PointLoad(10, 0)])
        with pytest.raises(ValueError, match="length"):
            validate_beam(config)

    def test_non_positive_length_negative(self):
        config = make_beam(-5.0, [PointLoad(10, 0), PointLoad(10, 0), PointLoad(10, 0)])
        with pytest.raises(ValueError, match="length"):
            validate_beam(config)

    def test_fewer_than_three_point_loads(self):
        config = make_beam(10.0, [PointLoad(10, 2), PointLoad(20, 5)])
        with pytest.raises(ValueError, match="three"):
            validate_beam(config)

    def test_point_load_before_beam(self):
        loads = [PointLoad(10, -1), PointLoad(10, 5), PointLoad(10, 8)]
        config = make_beam(10.0, loads)
        with pytest.raises(ValueError, match="outside"):
            validate_beam(config)

    def test_point_load_after_beam(self):
        loads = [PointLoad(10, 2), PointLoad(10, 5), PointLoad(10, 11)]
        config = make_beam(10.0, loads)
        with pytest.raises(ValueError, match="outside"):
            validate_beam(config)

    def test_uvl_extends_before_beam(self):
        loads = [PointLoad(10, 2), PointLoad(10, 5), PointLoad(10, 8)]
        uvl = UVL(a=-1.0, b=5.0, w_a=10.0, w_b=10.0)
        config = make_beam(10.0, loads, uvl)
        with pytest.raises(ValueError, match="outside"):
            validate_beam(config)

    def test_uvl_extends_after_beam(self):
        loads = [PointLoad(10, 2), PointLoad(10, 5), PointLoad(10, 8)]
        uvl = UVL(a=5.0, b=11.0, w_a=10.0, w_b=10.0)
        config = make_beam(10.0, loads, uvl)
        with pytest.raises(ValueError, match="outside"):
            validate_beam(config)

    def test_uvl_start_equals_end(self):
        loads = [PointLoad(10, 2), PointLoad(10, 5), PointLoad(10, 8)]
        uvl = UVL(a=4.0, b=4.0, w_a=10.0, w_b=10.0)
        config = make_beam(10.0, loads, uvl)
        with pytest.raises(ValueError, match="strictly less"):
            validate_beam(config)

    def test_uvl_start_greater_than_end(self):
        loads = [PointLoad(10, 2), PointLoad(10, 5), PointLoad(10, 8)]
        uvl = UVL(a=7.0, b=3.0, w_a=10.0, w_b=10.0)
        config = make_beam(10.0, loads, uvl)
        with pytest.raises(ValueError, match="strictly less"):
            validate_beam(config)

    def test_negative_uvl_intensity_w_a(self):
        loads = [PointLoad(10, 2), PointLoad(10, 5), PointLoad(10, 8)]
        uvl = UVL(a=2.0, b=8.0, w_a=-5.0, w_b=10.0)
        config = make_beam(10.0, loads, uvl)
        with pytest.raises(ValueError, match="non-negative"):
            validate_beam(config)

    def test_negative_uvl_intensity_w_b(self):
        loads = [PointLoad(10, 2), PointLoad(10, 5), PointLoad(10, 8)]
        uvl = UVL(a=2.0, b=8.0, w_a=10.0, w_b=-5.0)
        config = make_beam(10.0, loads, uvl)
        with pytest.raises(ValueError, match="non-negative"):
            validate_beam(config)


class TestVerticalEquilibrium:
    

    def _check_equilibrium(self, config):
        validate_beam(config)
        R_A, R_B = calculate_reactions(config)
        W = total_resultant_load(config)
        assert abs(R_A + R_B - W) < ATOL, (
            f"Force equilibrium violated: R_A={R_A}, R_B={R_B}, W={W}"
        )

    def test_equilibrium_three_point_loads(self):
        config = make_beam(10.0, [PointLoad(10,2), PointLoad(20,5), PointLoad(30,8)])
        self._check_equilibrium(config)

    def test_equilibrium_trapezoidal_uvl(self):
        loads = [PointLoad(5, 1), PointLoad(15, 4), PointLoad(10, 9)]
        uvl = UVL(a=2.0, b=8.0, w_a=6.0, w_b=18.0)
        config = make_beam(10.0, loads, uvl)
        self._check_equilibrium(config)

    def test_equilibrium_full_span_triangular(self):
        L = 8.0
        loads = [PointLoad(0, 0), PointLoad(0, 4), PointLoad(0, 8)]
        uvl = UVL(a=0.0, b=L, w_a=0.0, w_b=16.0)
        config = make_beam(L, loads, uvl)
        self._check_equilibrium(config)


class TestMomentEquilibrium:

    def _check_moment_equilibrium(self, config):
        validate_beam(config)
        R_A, R_B = calculate_reactions(config)
        L = config.length

        moment_loads = sum(pl.magnitude * pl.position for pl in config.point_loads)
        if config.uvl is not None:
            R_uvl, x_bar = uvl_resultant(config.uvl)
            moment_loads += R_uvl * x_bar

        assert abs(R_B * L - moment_loads) < ATOL, (
            f"Moment equilibrium violated: R_B*L={R_B*L}, ΣM_loads={moment_loads}"
        )

    def test_moment_eq_point_loads_only(self):
        config = make_beam(8.0, [PointLoad(10,2), PointLoad(20,5), PointLoad(30,7)])
        self._check_moment_equilibrium(config)

    def test_moment_eq_with_trapezoidal_uvl(self):
        loads = [PointLoad(15, 3), PointLoad(25, 6), PointLoad(10, 9)]
        uvl = UVL(a=2.0, b=10.0, w_a=8.0, w_b=24.0)
        config = make_beam(12.0, loads, uvl)
        self._check_moment_equilibrium(config)




class TestBoundaryConditions:

    def _check_moment_boundaries(self, config):
        validate_beam(config)
        L = config.length
        assert abs(bending_moment(config, 0.0)) < ATOL, "M(0) ≠ 0"
        assert abs(bending_moment(config, L)) < ATOL, "M(L) ≠ 0"

    def test_boundary_three_point_loads(self):
        config = make_beam(10.0, [PointLoad(10,2), PointLoad(20,5), PointLoad(30,8)])
        self._check_moment_boundaries(config)

    def test_boundary_triangular_uvl(self):
        loads = [PointLoad(5, 1), PointLoad(10, 5), PointLoad(5, 9)]
        uvl = UVL(a=0.0, b=10.0, w_a=0.0, w_b=20.0)
        config = make_beam(10.0, loads, uvl)
        self._check_moment_boundaries(config)

    def test_boundary_trapezoidal_partial_uvl(self):
        loads = [PointLoad(15, 3), PointLoad(25, 6), PointLoad(10, 9)]
        uvl = UVL(a=2.0, b=10.0, w_a=8.0, w_b=24.0)
        config = make_beam(12.0, loads, uvl)
        self._check_moment_boundaries(config)

    def test_boundary_five_point_loads(self):
        loads = [
            PointLoad(5.0, 1.0), PointLoad(10.0, 3.0), PointLoad(15.0, 5.0),
            PointLoad(20.0, 7.0), PointLoad(25.0, 9.0),
        ]
        config = make_beam(10.0, loads)
        self._check_moment_boundaries(config)

    def test_boundary_sample_beam(self):
        config = sample_beam()
        self._check_moment_boundaries(config)



class TestShearJump:

    def test_shear_jump_single_load(self):
        L = 6.0
        loads = [PointLoad(0, 0), PointLoad(30, 3), PointLoad(0, 6)]
        config = make_beam(L, loads)
        validate_beam(config)

        eps = 1e-8
        V_before = shear_force(config, 3.0 - eps)
        V_after  = shear_force(config, 3.0 + eps)
        jump = V_after - V_before

        assert abs(jump - (-30.0)) < 1e-5   

    def test_bending_moment_continuous_at_load(self):
        L = 6.0
        loads = [PointLoad(0, 0), PointLoad(30, 3), PointLoad(0, 6)]
        config = make_beam(L, loads)
        eps = 1e-8
        M_left  = bending_moment(config, 3.0 - eps)
        M_right = bending_moment(config, 3.0 + eps)
        M_at    = bending_moment(config, 3.0)

        assert abs(M_left - M_at) < 1e-4
        assert abs(M_right - M_at) < 1e-4



class TestSampleBeam:

    def test_sample_beam_is_valid(self):
        config = sample_beam()
        validate_beam(config)   

    def test_sample_beam_equilibrium(self):
        config = sample_beam()
        R_A, R_B = calculate_reactions(config)
        W = total_resultant_load(config)
        assert abs(R_A + R_B - W) < ATOL

    def test_sample_beam_moment_boundaries(self):
        config = sample_beam()
        L = config.length
        assert abs(bending_moment(config, 0.0)) < ATOL
        assert abs(bending_moment(config, L)) < ATOL

    def test_sample_beam_shear_at_supports(self):
        config = sample_beam()
        L = config.length
        R_A, R_B = calculate_reactions(config)
        eps = 1e-8

        assert abs(shear_force(config, eps) - R_A) < 1e-5
        assert abs(shear_force(config, L - eps) + R_B) < 1e-5
