from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any

from src.beam_analysis import BeamConfig, PointLoad, UVL


def independent_udl_resultant(a: float, b: float, w: float) -> Tuple[float, float]:

    span = b - a
    if span <= 0:
        raise ValueError(f"Invalid UDL span: a={a}, b={b}")
    return w * span, 0.5 * (a + b)


def independent_uvl_resultant(a: float, b: float, w_a: float, w_b: float) -> Tuple[float, float]:

    span = b - a
    if span <= 0:
        raise ValueError(f"Invalid UVL span: a={a}, b={b}")

    R = 0.5 * (w_a + w_b) * span

    if abs(w_a + w_b) < 1e-15:

        x_bar = 0.5 * (a + b)
    elif abs(w_a - w_b) < 1e-15:

        x_bar = 0.5 * (a + b)
    else:

        x_bar = (w_a * (2.0 * a + b) + w_b * (a + 2.0 * b)) / (3.0 * (w_a + w_b))

    return R, x_bar


def independent_calculate_reactions(config: BeamConfig) -> Tuple[float, float]:

    L = config.length
    if L <= 0:
        raise ValueError(f"Beam length must be positive. Got L={L}")

    moment_about_A = 0.0
    moment_about_B = 0.0
    total_vertical_load = 0.0

    for pl in config.point_loads:
        total_vertical_load += pl.magnitude
        moment_about_A += pl.magnitude * pl.position
        moment_about_B += pl.magnitude * (L - pl.position)

    all_uvls = config.get_all_uvls() if hasattr(config, "get_all_uvls") else ([config.uvl] if config.uvl else [])
    for uvl in all_uvls:
        R_uvl, x_bar = independent_uvl_resultant(
            uvl.a, uvl.b, uvl.w_a, uvl.w_b
        )
        total_vertical_load += R_uvl
        moment_about_A += R_uvl * x_bar
        moment_about_B += R_uvl * (L - x_bar)

    moments = getattr(config, "point_moments", [])
    for pm in moments:
        moment_about_A += pm.magnitude
        moment_about_B -= pm.magnitude

    R_B = moment_about_A / L
    R_A = moment_about_B / L

    return R_A, R_B


def verify_static_equilibrium(
    config: BeamConfig,
    R_A: float,
    R_B: float,
    tol: float = 1e-9,
) -> Dict[str, Any]:

    L = config.length
    
    total_downward_load = sum(pl.magnitude for pl in config.point_loads)
    moment_A_loads = sum(pl.magnitude * pl.position for pl in config.point_loads)
    moment_B_loads = sum(pl.magnitude * (L - pl.position) for pl in config.point_loads)

    all_uvls = config.get_all_uvls() if hasattr(config, "get_all_uvls") else ([config.uvl] if config.uvl else [])
    for uvl in all_uvls:
        R_uvl, x_bar = independent_uvl_resultant(
            uvl.a, uvl.b, uvl.w_a, uvl.w_b
        )
        total_downward_load += R_uvl
        moment_A_loads += R_uvl * x_bar
        moment_B_loads += R_uvl * (L - x_bar)

    moments = getattr(config, "point_moments", [])
    for pm in moments:
        moment_A_loads += pm.magnitude
        moment_B_loads -= pm.magnitude

    # Residuals
    res_Fy = (R_A + R_B) - total_downward_load
    res_MA = (R_B * L) - moment_A_loads
    res_MB = (R_A * L) - moment_B_loads

    is_Fy_balanced = abs(res_Fy) <= tol
    is_MA_balanced = abs(res_MA) <= tol
    is_MB_balanced = abs(res_MB) <= tol

    return {
        "is_balanced": is_Fy_balanced and is_MA_balanced and is_MB_balanced,
        "residual_vertical": res_Fy,
        "residual_moment_A": res_MA,
        "residual_moment_B": res_MB,
        "total_load": total_downward_load,
    }

def analytical_central_point_load(L: float, P: float) -> Dict[str, Any]:

    return {
        "L": L,
        "P": P,
        "R_A": P / 2.0,
        "R_B": P / 2.0,
        "M_max": (P * L) / 4.0,
        "x_M_max": L / 2.0,
        "V_left_mid": P / 2.0,
        "V_right_mid": -P / 2.0,
    }


def analytical_full_span_udl(L: float, w0: float) -> Dict[str, Any]:

    return {
        "L": L,
        "w0": w0,
        "R_A": 0.5 * w0 * L,
        "R_B": 0.5 * w0 * L,
        "M_max": (w0 * L**2) / 8.0,
        "x_M_max": L / 2.0,
    }


def analytical_triangular_uvl(L: float, w0: float) -> Dict[str, Any]:

    x_max = L / math.sqrt(3.0)
    M_max = (w0 * L**2) / (9.0 * math.sqrt(3.0))
    return {
        "L": L,
        "w0": w0,
        "total_load": 0.5 * w0 * L,
        "centroid": (2.0 / 3.0) * L,
        "R_A": (w0 * L) / 6.0,
        "R_B": (w0 * L) / 3.0,
        "x_M_max": x_max,
        "M_max": M_max,
    }


def analytical_decreasing_triangular_uvl(L: float, w0: float) -> Dict[str, Any]:

    x_max = L * (1.0 - 1.0 / math.sqrt(3.0))
    M_max = (w0 * L**2) / (9.0 * math.sqrt(3.0))
    return {
        "L": L,
        "w0": w0,
        "total_load": 0.5 * w0 * L,
        "centroid": (1.0 / 3.0) * L,
        "R_A": (w0 * L) / 3.0,
        "R_B": (w0 * L) / 6.0,
        "x_M_max": x_max,
        "M_max": M_max,
    }

