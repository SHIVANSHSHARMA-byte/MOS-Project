from __future__ import annotations

import sys
import os
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


matplotlib.rcParams["figure.facecolor"] = "#0B0F14"
matplotlib.rcParams["axes.facecolor"] = "#161D27"
matplotlib.rcParams["savefig.facecolor"] = "#0B0F14"
matplotlib.rcParams["savefig.edgecolor"] = "#0B0F14"
from matplotlib.figure import Figure
from matplotlib.axes import Axes


sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.beam_analysis import (
    BeamConfig,
    PointLoad,
    UVL,
    PointMoment,
    calculate_reactions,
    shear_force,
    bending_moment,
    uvl_resultant,
    find_points_of_contraflexure,
)



ExtremaResult = Dict[str, Any]


def find_bending_moment_extrema(
    config: BeamConfig,
    num_sample: int = 2001,
) -> ExtremaResult:
    
    L = config.length


    critical_xs: List[float] = [0.0, L]

    for pl in config.point_loads:
        critical_xs.append(pl.position)

    all_uvls = config.get_all_uvls() if hasattr(config, "get_all_uvls") else ([config.uvl] if config.uvl else [])
    for u in all_uvls:
        critical_xs.extend([u.a, u.b])

    for pm in getattr(config, "point_moments", []):
        critical_xs.append(pm.position)

    critical_xs = sorted(set(critical_xs))

    _EPS = 1e-9

    def _V(x: float) -> float:
        return shear_force(config, x)

    zero_shear_xs: List[float] = []

    for i in range(len(critical_xs) - 1):
        x_lo = critical_xs[i] + _EPS
        x_hi = critical_xs[i + 1] - _EPS
        if x_hi <= x_lo:
            continue
        V_lo = _V(x_lo)
        V_hi = _V(x_hi)
        if V_lo * V_hi <= 0.0:         

            a_b, b_b = x_lo, x_hi
            for _ in range(60):
                mid = (a_b + b_b) / 2.0
                if _V(mid) * _V(a_b) <= 0.0:
                    b_b = mid
                else:
                    a_b = mid
                if (b_b - a_b) < 1e-10:
                    break
            zero_shear_xs.append((a_b + b_b) / 2.0)


    candidate_xs = (
        critical_xs
        + zero_shear_xs
        + list(np.linspace(0.0, L, num_sample))
    )

    candidate_xs = [max(0.0, min(L, x)) for x in candidate_xs]


    candidates: List[Tuple[float, float]] = [
        (x, bending_moment(config, x)) for x in candidate_xs
    ]

    M_vals = [m for _, m in candidates]
    max_idx = int(np.argmax(M_vals))
    min_idx = int(np.argmin(M_vals))

    cp_pts = find_points_of_contraflexure(config) if "find_points_of_contraflexure" in globals() else []

    return {
        "max_positive_value": candidates[max_idx][1],
        "max_positive_pos":   candidates[max_idx][0],
        "min_negative_value": candidates[min_idx][1],
        "min_negative_pos":   candidates[min_idx][0],
        "candidates":         candidates,
        "contraflexure_points": cp_pts,
        "note": (
            "Zero-shear positions found by bisection (< 1e-10 m accuracy). "
            "Dense uniform sampling used as a safety net; positional accuracy "
            f"~{L / num_sample:.2e} m for sampled points."
        ),
    }


def _build_sfd_arrays(
    config: BeamConfig,
    num_smooth: int = 500,
) -> Tuple[np.ndarray, np.ndarray]:

    L = config.length
    _EPS = 1e-9

    jump_xs = sorted({pl.position for pl in config.point_loads})


    breakpoints = sorted({0.0} | set(jump_xs) | {L})

    x_segments: List[float] = []
    V_segments: List[float] = []

    for i, bp in enumerate(breakpoints):
        if i == 0:

            x0 = bp + _EPS
            x_segments.append(x0)
            V_segments.append(shear_force(config, x0))
        else:

            x_r = bp + _EPS if bp < L else bp
            x_segments.append(x_r)
            V_segments.append(shear_force(config, x_r))


        if bp in jump_xs and bp != L:
            x_segments.append(bp)
            V_segments.append(float("nan"))


    critical = sorted({0.0} | set(jump_xs) | {L})

    x_out: List[float] = []
    V_out: List[float] = []

    for i in range(len(critical) - 1):
        x_lo = critical[i]
        x_hi = critical[i + 1]


        x_start = x_lo + _EPS
        x_end   = x_hi - _EPS

        xs = np.linspace(x_start, x_end, num_smooth)
        for xv in xs:
            x_out.append(xv)
            V_out.append(shear_force(config, xv))


        if i < len(critical) - 2:
            x_out.append(x_hi)
            V_out.append(float("nan"))
        else:

            x_out.append(x_hi - _EPS)
            V_out.append(shear_force(config, x_hi - _EPS))

    return np.array(x_out), np.array(V_out)


def _build_bmd_arrays(
    config: BeamConfig,
    num_points: int = 1001,
) -> Tuple[np.ndarray, np.ndarray]:

    L = config.length


    critical_xs: List[float] = [0.0, L]
    for pl in config.point_loads:
        critical_xs.append(pl.position)
    if config.uvl is not None:
        critical_xs.extend([config.uvl.a, config.uvl.b])

    base_xs = np.linspace(0.0, L, num_points)
    all_xs  = np.union1d(base_xs, np.array(sorted(set(critical_xs))))
    all_xs  = np.clip(all_xs, 0.0, L)

    M_vals = np.array([bending_moment(config, float(x)) for x in all_xs])
    return all_xs, M_vals


def _draw_beam_diagram(
    ax: Axes,
    config: BeamConfig,
    R_A: float,
    R_B: float,
) -> None:

    L = config.length
    ax.set_facecolor("#161D27")

    margin_x = max(0.08 * L, 0.45)
    ax.set_xlim(-margin_x, L + margin_x)

    ax.axhline(0, color="#64748B", linewidth=4.5, solid_capstyle="round", zorder=2)


    pl_magnitudes = [pl.magnitude for pl in config.point_loads if pl.magnitude > 0]
    all_uvls = config.get_all_uvls() if hasattr(config, "get_all_uvls") else ([config.uvl] if config.uvl else [])
    uvl_max = max([max(uvl.w_a, uvl.w_b) for uvl in all_uvls], default=0.0)
    point_moments = getattr(config, "point_moments", [])

    all_values = pl_magnitudes + ([uvl_max] if uvl_max > 0 else [])
    max_val = max(all_values) if all_values else 1.0
    arrow_scale = 0.35 * L / max(max_val, 1.0)

    y_top = max(max_val * arrow_scale, 0.20 * L)

    tri_h = 0.16 * y_top
    tri_w = min(0.035 * L, 0.45)

    has_udl = any(abs(u.w_a - u.w_b) < 1e-6 for u in all_uvls)
    has_uvl = any(abs(u.w_a - u.w_b) >= 1e-6 for u in all_uvls)
    active_pls = [pl for pl in config.point_loads if abs(pl.magnitude) > 1e-12]

    for idx, uvl in enumerate(all_uvls):
        a, b = uvl.a, uvl.b
        h_a = uvl.w_a * arrow_scale
        h_b = uvl.w_b * arrow_scale
        is_udl = abs(uvl.w_a - uvl.w_b) < 1e-6

        poly_x = [a, b, b, a]
        poly_y = [0.0, 0.0, h_b, h_a]

        if is_udl:
            ax.fill(poly_x, poly_y, color="#0284C7", alpha=0.30, zorder=1, label="UDL (kN/m)" if idx == 0 else "")
            ax.plot([a, a], [0.0, h_a], color="#38BDF8", linewidth=1.6, zorder=3)
            ax.plot([b, b], [0.0, h_b], color="#38BDF8", linewidth=1.6, zorder=3)
            ax.plot([a, b], [h_a, h_b], color="#38BDF8", linewidth=2.0, linestyle="-", zorder=3)
        else:
            ax.fill(poly_x, poly_y, color="#6366F1", alpha=0.30, zorder=1, label="UVL (kN/m)" if idx == 0 else "")
            ax.plot([a, a], [0.0, h_a], color="#818CF8", linewidth=1.6, zorder=3)
            ax.plot([b, b], [0.0, h_b], color="#818CF8", linewidth=1.6, zorder=3)
            ax.plot([a, b], [h_a, h_b], color="#818CF8", linewidth=2.0, linestyle="-", zorder=3)


        n_ticks = min(12, max(4, int((b - a) / L * 18)))
        tick_color = "#38BDF8" if is_udl else "#818CF8"
        for xv in np.linspace(a, b, n_ticks):
            h_v = (uvl.w_a + (uvl.w_b - uvl.w_a) / max(b - a, 1e-6) * (xv - a)) * arrow_scale
            if h_v > 0.01 * arrow_scale:
                ax.annotate(
                    "", xy=(xv, 0.0),
                    xytext=(xv, h_v),
                    arrowprops=dict(
                        arrowstyle="-|>",
                        color=tick_color,
                        lw=1.1,
                        mutation_scale=9,
                    ),
                    zorder=3,
                )


        if is_udl:
            mid_x = (a + b) / 2.0
            ax.text(
                mid_x, h_a + 0.08 * y_top,
                f"UDL: {uvl.w_a:.1f} kN/m",
                ha="center", va="bottom", fontsize=8.5,
                color="#38BDF8", fontweight="bold", zorder=4,
                bbox=dict(boxstyle="round,pad=0.25", facecolor="#161D27", edgecolor="#0284C7", linewidth=1.2, alpha=0.95),
            )
        else:
            label_a = f"w_a={uvl.w_a:.1f} kN/m" if uvl.w_a > 0 else "w_a=0"
            label_b = f"w_b={uvl.w_b:.1f} kN/m" if uvl.w_b > 0 else "w_b=0"
            
            near_pl_a = any(abs(pl.position - a) < 0.06 * L for pl in active_pls)
            near_pl_b = any(abs(pl.position - b) < 0.06 * L for pl in active_pls)

            pos_x_a = a if a > 0.08 * L else (a + 0.03 * (b - a))
            ha_a = "right" if a > 0.12 * L else "left"
            y_shift_a = 0.16 * y_top if near_pl_a else 0.06 * y_top
            ax.text(
                pos_x_a, max(h_a, 0.02 * y_top) + y_shift_a,
                label_a, ha=ha_a, va="bottom", fontsize=8,
                color="#A5B4FC", fontweight="bold", zorder=4,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#161D27", edgecolor="#6366F1", linewidth=1.1, alpha=0.95),
            )

            pos_x_b = b if (L - b) > 0.08 * L else (b - 0.03 * (b - a))
            ha_b = "left" if (L - b) > 0.12 * L else "right"
            y_shift_b = 0.16 * y_top if near_pl_b else 0.06 * y_top
            ax.text(
                pos_x_b, max(h_b, 0.02 * y_top) + y_shift_b,
                label_b, ha=ha_b, va="bottom", fontsize=8,
                color="#A5B4FC", fontweight="bold", zorder=4,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#161D27", edgecolor="#6366F1", linewidth=1.1, alpha=0.95),
            )


    sorted_pls = sorted(active_pls, key=lambda p: p.position)
    
    stagger_levels = {}
    for i in range(len(sorted_pls)):
        level = 0
        if i > 0:
            dist_prev = sorted_pls[i].position - sorted_pls[i - 1].position
            if dist_prev < 0.12 * L:
                used = {stagger_levels.get(i - 1, 0)}
                if i > 1 and (sorted_pls[i].position - sorted_pls[i - 2].position) < 0.18 * L:
                    used.add(stagger_levels.get(i - 2, 0))
                for cand in (0, 1, 2):
                    if cand not in used:
                        level = cand
                        break
        stagger_levels[i] = level

    for i, pl in enumerate(sorted_pls):
        h = pl.magnitude * arrow_scale
        level = stagger_levels[i]
        
        ax.annotate(
            "", xy=(pl.position, 0.0),
            xytext=(pl.position, h),
            arrowprops=dict(
                arrowstyle="-|>",
                color="#F43F5E",
                lw=2.0,
                mutation_scale=14,
            ),
            zorder=4,
        )
        
        y_text_offsets = {0: 0.07, 1: 0.22, 2: 0.37}
        y_text = h + y_text_offsets[level] * y_top
        if level > 0:
            ax.plot([pl.position, pl.position], [h, y_text], color="#F43F5E", lw=0.9, linestyle=":", zorder=4)

        ax.text(
            pl.position, y_text,
            f"{pl.magnitude:.1f} kN",
            ha="center", va="bottom", fontsize=8.5,
            color="#FDA4AF", fontweight="bold",
            zorder=5,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#1E293B", edgecolor="#F43F5E", linewidth=1.2, alpha=0.95),
        )

    for idx, pm in enumerate(point_moments):
        if abs(pm.magnitude) < 1e-12:
            continue
        is_cw = (pm.magnitude > 0)
        arc = mpatches.Arc(
            (pm.position, 0.0),
            width=min(0.09 * L, 0.9), height=0.24 * y_top,
            angle=0, theta1=0, theta2=360,
            color="#A855F7", linewidth=2.0, zorder=4
        )
        ax.add_patch(arc)
        
        arr_x = pm.position + min(0.045 * L, 0.45)
        arr_y = 0.0
        ax.annotate(
            "",
            xy=(arr_x, arr_y + (-0.03 if is_cw else 0.03) * y_top),
            xytext=(arr_x, arr_y + (0.03 if is_cw else -0.03) * y_top),
            arrowprops=dict(arrowstyle="-|>", color="#A855F7", lw=2.0, mutation_scale=11),
            zorder=5,
        )

        direction_label = "CW" if is_cw else "CCW"
        ax.text(
            pm.position, 0.24 * y_top,
            f"{abs(pm.magnitude):.1f} kN·m ({direction_label})",
            ha="center", va="bottom", fontsize=8,
            color="#E9D5FF", fontweight="bold", zorder=5,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#161D27", edgecolor="#A855F7", linewidth=1.1, alpha=0.95),
        )


    pin_triangle = plt.Polygon(
        [[0.0, 0.0], [-tri_w, -tri_h], [tri_w, -tri_h]],
        closed=True,
        facecolor="#0B0F14",
        edgecolor="#38BDF8",
        linewidth=2.0,
        zorder=5,
    )
    ax.add_patch(pin_triangle)
    ax.plot([-tri_w * 1.3, tri_w * 1.3], [-tri_h, -tri_h], color="#64748B", linewidth=1.5, zorder=5)

    roller_triangle = plt.Polygon(
        [[L, 0.0], [L - tri_w * 0.9, -tri_h * 0.65], [L + tri_w * 0.9, -tri_h * 0.65]],
        closed=True,
        facecolor="#0B0F14",
        edgecolor="#38BDF8",
        linewidth=2.0,
        zorder=5,
    )
    ax.add_patch(roller_triangle)
    roller = mpatches.Ellipse(
        (L, -tri_h * 0.85),
        width=tri_w * 1.5, height=tri_h * 0.32,
        facecolor="#0B0F14",
        edgecolor="#38BDF8",
        linewidth=1.8,
        zorder=5,
    )
    ax.add_patch(roller)
    ax.plot([L - tri_w, L + tri_w], [-tri_h, -tri_h], color="#64748B", linewidth=1.5, zorder=5)

    y_reaction_pos = -tri_h - 0.14 * y_top
    ax.text(
        0.0, y_reaction_pos,
        f"Support A\n↑ R_A = {R_A:.2f} kN",
        ha="center", va="top", fontsize=8.5,
        color="#34D399", fontweight="bold", zorder=6,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#161D27", edgecolor="#34D399", linewidth=1.2, alpha=0.95),
    )
    ax.text(
        L, y_reaction_pos,
        f"Support B\n↑ R_B = {R_B:.2f} kN",
        ha="center", va="top", fontsize=8.5,
        color="#34D399", fontweight="bold", zorder=6,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#161D27", edgecolor="#34D399", linewidth=1.2, alpha=0.95),
    )

    ax.set_title("Beam Loading Diagram", fontsize=11.5, fontweight="bold", pad=8, color="#F1F5F9")
    ax.set_ylabel("Load Intensity / Force", fontsize=9, color="#CBD5E1")
    
    max_stagger = max(stagger_levels.values(), default=0)
    y_min_bound = -tri_h - 0.42 * y_top
    y_max_bound = y_top * (1.72 if max_stagger == 2 else (1.55 if max_stagger == 1 else 1.42))
    ax.set_ylim(y_min_bound, y_max_bound)

    ax.axhline(0, color="#475569", linewidth=0.6, zorder=1)
    ax.set_xticks(np.linspace(0, L, min(11, int(L) + 1)))
    ax.tick_params(axis="x", labelbottom=False, colors="#94A3B8")
    ax.tick_params(axis="y", labelsize=8, colors="#94A3B8")
    ax.grid(True, axis="x", linestyle=":", linewidth=0.5, color="#334155", alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color("#334155")
    ax.spines["left"].set_color("#334155")

    handles = []
    if any(pl.magnitude > 0 for pl in config.point_loads):
        handles.append(mpatches.Patch(color="#F43F5E", label="Point Load (kN)"))
    if has_udl:
        handles.append(mpatches.Patch(color="#0284C7", alpha=0.6, label="UDL (kN/m)"))
    if has_uvl:
        handles.append(mpatches.Patch(color="#6366F1", alpha=0.6, label="UVL (kN/m)"))
    if not has_udl and not has_uvl and not config.point_loads:
        handles.append(mpatches.Patch(color="#0284C7", alpha=0.6, label="UDL (kN/m)"))
    handles.append(mpatches.Patch(color="#34D399", label="Reactions (kN)"))
    if point_moments:
        handles.append(mpatches.Patch(color="#C084FC", label="Moment (kN·m)"))

    has_loads_right = any(p.position > 0.72 * L for p in active_pls) or any(pm.position > 0.72 * L for pm in point_moments)
    has_loads_left = any(p.position < 0.28 * L for p in active_pls) or any(pm.position < 0.28 * L for pm in point_moments)
    has_loads_center = any(0.35 * L <= p.position <= 0.65 * L for p in active_pls)

    if has_loads_right and not has_loads_left:
        leg_loc = "upper left"
        leg_ncol = 1
    elif has_loads_right and has_loads_left:
        if not has_loads_center:
            leg_loc = "upper center"
            leg_ncol = min(len(handles), 4)
        else:
            leg_loc = "upper right"
            leg_ncol = 1
    else:
        leg_loc = "upper right"
        leg_ncol = 1

    leg = ax.legend(handles=handles, loc=leg_loc, ncol=leg_ncol, fontsize=8, framealpha=0.9, facecolor="#1E293B", edgecolor="#334155")
    for t_elem in leg.get_texts():
        t_elem.set_color("#CBD5E1")


def _draw_sfd(
    ax: Axes,
    config: BeamConfig,
    R_A: float,
    R_B: float,
    num_smooth: int = 500,
) -> None:

    L = config.length
    ax.set_facecolor("#161D27")
    margin_x = max(0.08 * L, 0.45)
    ax.set_xlim(-margin_x, L + margin_x)

    x_arr, V_arr = _build_sfd_arrays(config, num_smooth)

    ax.plot(x_arr, V_arr, color="#FB923C", linewidth=2.2, solid_capstyle="round", label="V(x)")
    ax.fill_between(x_arr, 0, V_arr, where=~np.isnan(V_arr), alpha=0.20, color="#F97316")


    ax.axhline(0, color="#64748B", linewidth=0.8, linestyle="--", zorder=1)

    finite_v = V_arr[~np.isnan(V_arr)]
    v_max_val = float(np.max(np.abs(finite_v))) if len(finite_v) > 0 else 1.0
    v_span = max(v_max_val, 1.0)

    _EPS = 1e-9
    active_pls = [pl for pl in config.point_loads if abs(pl.magnitude) > 1e-12]
    sorted_pls = sorted(active_pls, key=lambda p: p.position)

    for idx, pl in enumerate(sorted_pls):
        V_left = shear_force(config, pl.position - _EPS)
        V_right = shear_force(config, pl.position + _EPS)
        ax.plot([pl.position, pl.position], [V_left, V_right], color="#FB923C", linewidth=2.2, linestyle="-", zorder=4)

        mid_y = (V_left + V_right) / 2.0
        

        if pl.position > 0.75 * L:
            ha_jump = "right"
            x_jump_offset = -0.05 * L
        elif pl.position < 0.25 * L:
            ha_jump = "left"
            x_jump_offset = 0.05 * L
        else:
            ha_jump = "left" if idx % 2 == 0 else "right"
            x_jump_offset = 0.04 * L if ha_jump == "left" else -0.04 * L

        stagger_y = 0.0
        if idx > 0 and (pl.position - sorted_pls[idx - 1].position) < 0.12 * L:
            stagger_y = 0.15 * v_span if idx % 2 == 1 else -0.15 * v_span

        if pl.position < 0.15 * L:
            stagger_y = -0.14 * v_span if R_A >= 0 else 0.14 * v_span
        elif pl.position > 0.85 * L:
            stagger_y = 0.14 * v_span if -R_B <= 0 else -0.14 * v_span

        ax.annotate(
            f"ΔV={pl.magnitude:.1f} kN",
            xy=(pl.position, mid_y),
            xytext=(pl.position + x_jump_offset, mid_y + stagger_y),
            fontsize=7.5, color="#FDBA74", fontweight="bold",
            arrowprops=dict(arrowstyle="-|>", color="#FB923C", lw=0.9),
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#1E293B", edgecolor="#FB923C", linewidth=1.1, alpha=0.95),
            ha=ha_jump, va="center", zorder=5,
        )

    ax.plot(0.0, R_A, "o", color="#34D399", markersize=5, zorder=6)
    offset_ra_y = 0.12 * v_span if R_A >= 0 else -0.12 * v_span
    ax.annotate(
        f"V(0⁺) = R_A = {R_A:.2f} kN",
        xy=(0.0, R_A),
        xytext=(0.02 * L, R_A + offset_ra_y),
        fontsize=8.0, color="#34D399", fontweight="bold",
        arrowprops=dict(arrowstyle="-|>", color="#34D399", lw=0.9),
        bbox=dict(boxstyle="round,pad=0.2", facecolor="#161D27", edgecolor="#34D399", linewidth=1.1, alpha=0.95),
        ha="left", va="bottom" if offset_ra_y > 0 else "top", zorder=6,
    )

    ax.plot(L, -R_B, "o", color="#34D399", markersize=5, zorder=6)
    offset_rb_y = -0.12 * v_span if -R_B <= 0 else 0.12 * v_span
    ax.annotate(
        f"V(L⁻) = −R_B = {-R_B:.2f} kN",
        xy=(L, -R_B),
        xytext=(L - 0.02 * L, -R_B + offset_rb_y),
        fontsize=8.0, color="#34D399", fontweight="bold",
        arrowprops=dict(arrowstyle="-|>", color="#34D399", lw=0.9),
        bbox=dict(boxstyle="round,pad=0.2", facecolor="#161D27", edgecolor="#34D399", linewidth=1.1, alpha=0.95),
        ha="right", va="top" if offset_rb_y < 0 else "bottom", zorder=6,
    )

    ax.set_title("Shear Force Diagram (SFD)", fontsize=11.5, fontweight="bold", pad=8, color="#F1F5F9")
    ax.set_ylabel("Shear Force V(x)  [kN]", fontsize=9, color="#CBD5E1")
    ax.set_ylim(-1.42 * v_span, 1.42 * v_span)
    ax.set_xticks(np.linspace(0, L, min(11, int(L) + 1)))
    ax.tick_params(axis="x", labelbottom=False, colors="#94A3B8")
    ax.tick_params(axis="y", labelsize=8, colors="#94A3B8")
    ax.grid(True, linestyle=":", linewidth=0.5, color="#334155", alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color("#334155")
    ax.spines["left"].set_color("#334155")
    leg = ax.legend(loc="upper right", fontsize=8.5, framealpha=0.9, facecolor="#1E293B", edgecolor="#334155")
    for t_elem in leg.get_texts():
        t_elem.set_color("#CBD5E1")


def _draw_bmd(
    ax: Axes,
    config: BeamConfig,
    extrema: ExtremaResult,
    num_points: int = 1001,
) -> None:

    L = config.length
    ax.set_facecolor("#161D27")
    margin_x = max(0.08 * L, 0.45)
    ax.set_xlim(-margin_x, L + margin_x)

    x_arr, M_arr = _build_bmd_arrays(config, num_points)

    ax.plot(x_arr, M_arr, color="#38BDF8", linewidth=2.4, label="M(x)")
    ax.fill_between(x_arr, 0, M_arr, where=(M_arr >= 0), alpha=0.22, color="#0284C7", label="Sagging (+)")
    ax.fill_between(x_arr, 0, M_arr, where=(M_arr < 0), alpha=0.22, color="#EF4444", label="Hogging (−)")

    ax.axhline(0, color="#64748B", linewidth=0.8, linestyle="--", zorder=1)

    M_max = extrema["max_positive_value"]
    x_max = extrema["max_positive_pos"]
    M_min = extrema["min_negative_value"]
    x_min = extrema["min_negative_pos"]
    y_span = max(abs(M_max), abs(M_min), 1.0)


    if abs(M_max) > 1e-6:
        if x_max < 0.20 * L:
            xy_t_pos = (x_max + 0.05 * L, M_max + 0.14 * y_span)
            ha_m = "left"
        elif x_max > 0.80 * L:
            xy_t_pos = (x_max - 0.05 * L, M_max + 0.14 * y_span)
            ha_m = "right"
        else:
            xy_t_pos = (x_max, M_max + 0.18 * y_span)
            ha_m = "center"

        ax.annotate(
            f"M_max = {M_max:.2f} kN·m\n@ x = {x_max:.3f} m",
            xy=(x_max, M_max),
            xytext=xy_t_pos,
            fontsize=8.5, color="#38BDF8", fontweight="bold",
            arrowprops=dict(arrowstyle="-|>", color="#38BDF8", lw=1.2, mutation_scale=10),
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#1E293B", edgecolor="#38BDF8", linewidth=1.2, alpha=0.95),
            ha=ha_m, va="bottom", zorder=5,
        )
        ax.plot(x_max, M_max, "o", color="#38BDF8", markersize=6, zorder=5)


    if M_min < -1e-6:
        if x_min < 0.20 * L:
            xy_t_neg = (x_min + 0.05 * L, M_min - 0.14 * y_span)
            ha_mn = "left"
        elif x_min > 0.80 * L:
            xy_t_neg = (x_min - 0.05 * L, M_min - 0.14 * y_span)
            ha_mn = "right"
        else:
            xy_t_neg = (x_min, M_min - 0.18 * y_span)
            ha_mn = "center"

        ax.annotate(
            f"M_min = {M_min:.2f} kN·m\n@ x = {x_min:.3f} m",
            xy=(x_min, M_min),
            xytext=xy_t_neg,
            fontsize=8.5, color="#F87171", fontweight="bold",
            arrowprops=dict(arrowstyle="-|>", color="#EF4444", lw=1.2, mutation_scale=10),
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#1E293B", edgecolor="#EF4444", linewidth=1.2, alpha=0.95),
            ha=ha_mn, va="top", zorder=5,
        )
        ax.plot(x_min, M_min, "o", color="#EF4444", markersize=6, zorder=5)


    contraflexure_pts = extrema.get("contraflexure_points", [])
    prev_y_sign = 0.0
    prev_pos = -999.0
    for idx, cp in enumerate(contraflexure_pts):
        pos = getattr(cp, "position", cp.get("position", 0.0) if isinstance(cp, dict) else 0.0)
        mom = getattr(cp, "moment", cp.get("moment", 0.0) if isinstance(cp, dict) else 0.0)
        ax.plot(pos, mom, "D", color="#34D399", markersize=7, zorder=6, label="Point of Contraflexure" if idx == 0 else "")
        
        near_max = (abs(pos - x_max) < 0.15 * L) and (abs(M_max) > 1e-6)
        near_min = (abs(pos - x_min) < 0.15 * L) and (M_min < -1e-6)

        if near_max and not near_min:
            y_sign = -1.0
        elif near_min and not near_max:
            y_sign = 1.0
        elif near_max and near_min:
            y_sign = -1.0 if abs(pos - x_max) <= abs(pos - x_min) else 1.0
        else:
            y_sign = 1.0 if idx % 2 == 0 else -1.0

        if abs(pos - prev_pos) < 0.22 * L and y_sign == prev_y_sign:
            y_sign = -prev_y_sign

        y_offset = y_sign * 0.24 * y_span
        prev_y_sign = y_sign
        prev_pos = pos

        ax.annotate(
            f"Contraflexure (M=0)\nx = {pos:.3f} m",
            xy=(pos, mom),
            xytext=(pos, y_offset),
            fontsize=8.0, color="#A7F3D0", fontweight="bold",
            arrowprops=dict(arrowstyle="-|>", color="#34D399", lw=1.0, mutation_scale=8),
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#064E3B", edgecolor="#10B981", linewidth=1.1, alpha=0.95),
            ha="center", va="center", zorder=6,
        )

    ax.set_title("Bending Moment Diagram (BMD)", fontsize=11.5, fontweight="bold", pad=8, color="#F1F5F9")
    ax.set_ylabel("Bending Moment M(x)  [kN·m]", fontsize=9, color="#CBD5E1")
    ax.set_xlabel("Position along beam  x  [m]", fontsize=9.5, color="#CBD5E1")
    
    y_bmd_min = min(0.0, M_min) - 0.40 * y_span
    y_bmd_max = max(0.0, M_max) + 0.40 * y_span
    ax.set_ylim(y_bmd_min, y_bmd_max)

    ax.set_xticks(np.linspace(0, L, min(11, int(L) + 1)))
    ax.tick_params(axis="x", colors="#94A3B8")
    ax.tick_params(axis="y", labelsize=8, colors="#94A3B8")
    ax.grid(True, linestyle=":", linewidth=0.5, color="#334155", alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color("#334155")
    ax.spines["left"].set_color("#334155")
    leg = ax.legend(loc="upper right", fontsize=8.5, framealpha=0.9, facecolor="#1E293B", edgecolor="#334155")
    for t_elem in leg.get_texts():
        t_elem.set_color("#CBD5E1")


def plot_beam_diagrams(
    config: BeamConfig,
    num_points: int = 1001,
    show: bool = True,
    figsize: Tuple[float, float] = (11.5, 11.0),
) -> Tuple[Figure, Tuple[Axes, Axes, Axes]]:

    R_A, R_B = calculate_reactions(config)


    extrema = find_bending_moment_extrema(config, num_sample=2001)


    fig, axes = plt.subplots(
        nrows=3, ncols=1,
        figsize=figsize,
        gridspec_kw={"hspace": 0.32, "height_ratios": [1.25, 1.0, 1.0]},
        sharex=False,
    )
    fig.patch.set_facecolor("#0B0F14")
    fig.set_facecolor("#0B0F14")
    ax_load, ax_sfd, ax_bmd = axes
    for ax in axes:
        ax.set_facecolor("#161D27")

    _draw_beam_diagram(ax_load, config, R_A, R_B)
    _draw_sfd(ax_sfd, config, R_A, R_B, num_smooth=max(200, num_points // 2))
    _draw_bmd(ax_bmd, config, extrema, num_points=num_points)

    fig.suptitle(
        f"Simply Supported Beam Analysis  (L = {config.length:.2f} m)\n"
        f"Reactions:  R_A = {R_A:.3f} kN  |  R_B = {R_B:.3f} kN",
        fontsize=12.5,
        fontweight="bold",
        color="#F1F5F9",
        y=0.992,
    )

    fig.subplots_adjust(top=0.93, bottom=0.06, left=0.08, right=0.96)

    if show:
        plt.show()

    return fig, (ax_load, ax_sfd, ax_bmd)
