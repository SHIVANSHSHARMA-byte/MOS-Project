from __future__ import annotations

import io
import os
import sys
from datetime import datetime
from typing import Optional, Dict, Any, List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
    KeepTogether,
    HRFlowable,
)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.beam_analysis import (
    BeamConfig,
    PointLoad,
    UVL,
    calculate_reactions,
    shear_force,
    bending_moment,
    validate_beam,
    uvl_resultant,
)
from src.beam_plotter import (
    plot_beam_diagrams,
    find_bending_moment_extrema,
)
from src.beam_validation import (
    verify_static_equilibrium,
)


def generate_beam_pdf_report(
    config: BeamConfig,
    output_path_or_buffer: Any = None,
    num_table_rows: int = 15,
    section_props: Optional[Any] = None,
    stress_results: Optional[Any] = None,
    min_point_loads: Optional[int] = 3,
) -> bytes:

    validate_beam(config, min_point_loads=min_point_loads)


    R_A, R_B = calculate_reactions(config)
    L = config.length


    audit = verify_static_equilibrium(config, R_A, R_B)


    extrema = find_bending_moment_extrema(config)

    if output_path_or_buffer is None:
        buffer = io.BytesIO()
    elif isinstance(output_path_or_buffer, (str, os.PathLike)):
        buffer = open(output_path_or_buffer, "wb")
    else:
        buffer = output_path_or_buffer


    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40,
    )

    styles = getSampleStyleSheet()


    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#1A365D"),
        spaceAfter=4,
    )
    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=11,
        leading=15,
        textColor=colors.HexColor("#4A5568"),
        spaceAfter=12,
    )
    h1_style = ParagraphStyle(
        "SectionH1",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=16,
        textColor=colors.HexColor("#2B6CB0"),
        spaceBefore=12,
        spaceAfter=6,
    )
    body_style = ParagraphStyle(
        "BodyDark",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#2D3748"),
    )
    body_bold = ParagraphStyle(
        "BodyDarkBold",
        parent=body_style,
        fontName="Helvetica-Bold",
    )
    callout_style = ParagraphStyle(
        "Callout",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#4A5568"),
    )

    story = []

    story.append(Paragraph("MOS Project — Simply Supported Beam Analysis", title_style))
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    story.append(
        Paragraph(
            f"<b>Comprehensive Engineering Verification &amp; Diagram Report</b> &nbsp;|&nbsp; "
            f"Generated: {now_str} &nbsp;|&nbsp; Units: SI (m, kN, kN/m, kN·m)",
            subtitle_style,
        )
    )
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#2B6CB0"), spaceAfter=10))


    model_box_data = [[
        Paragraph(
            "<b>Structural Classification:</b> Statically Determinate Simply Supported Beam (Euler-Bernoulli idealization). "
            "Left Support (x = 0): Ideal Pin (horizontal/vertical constraint). "
            f"Right Support (x = {L:.2f} m): Ideal Roller (vertical constraint). "
            "All loads act vertically downward in the transverse plane.",
            callout_style,
        )
    ]]
    t_model = Table(model_box_data, colWidths=[532])
    t_model.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#EBF8FF")),
        ("BOX", (0, 0), (-1, -1), 0.75, colors.HexColor("#90CDF4")),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(t_model)
    story.append(Spacer(1, 10))

    story.append(Paragraph("1. Beam Geometry &amp; Applied Loading Configuration", h1_style))


    config_rows = [
        [Paragraph("<b>Parameter</b>", body_bold), Paragraph("<b>Value &amp; Specification</b>", body_bold)],
        [Paragraph("Beam Span (L)", body_style), Paragraph(f"{L:.3f} m", body_style)],
        [Paragraph("Point Loads Count", body_style), Paragraph(f"{len(config.point_loads)} concentrated load(s)", body_style)],
    ]

    for idx, pl in enumerate(config.point_loads):
        config_rows.append([
            Paragraph(f"&nbsp;&nbsp;• Point Load #{idx + 1}", body_style),
            Paragraph(f"Magnitude = <b>{pl.magnitude:.2f} kN</b> at position x = <b>{pl.position:.3f} m</b>", body_style)
        ])

    all_uvls = config.get_all_uvls() if hasattr(config, "get_all_uvls") else ([config.uvl] if config.uvl else [])
    if all_uvls:
        for idx, u in enumerate(all_uvls, 1):
            R_u, x_bar_u = uvl_resultant(u)
            is_udl = abs(u.w_a - u.w_b) < 1e-6
            if is_udl:
                udl_label = f"Uniformly Distributed Load (UDL #{idx})" if len(all_uvls) > 1 else "Uniformly Distributed Load (UDL)"
                config_rows.extend([
                    [Paragraph(udl_label, body_style), Paragraph(f"Active from a = <b>{u.a:.3f} m</b> to b = <b>{u.b:.3f} m</b> (Span = {u.b - u.a:.3f} m)", body_style)],
                    [Paragraph("&nbsp;&nbsp;• Constant Intensity", body_style), Paragraph(f"w = <b>{u.w_a:.2f} kN/m</b> (uniform over span)", body_style)],
                    [Paragraph("&nbsp;&nbsp;• Resultant &amp; Centroid", body_style), Paragraph(f"W = w·(b−a) = <b>{R_u:.3f} kN</b> acting at midspan x̄ = (a+b)/2 = <b>{x_bar_u:.3f} m</b>", body_style)],
                ])
            else:
                uvl_label = f"Uniformly Varying Load (UVL #{idx})" if len(all_uvls) > 1 else "Uniformly Varying Load (UVL)"
                config_rows.extend([
                    [Paragraph(uvl_label, body_style), Paragraph(f"Active from a = <b>{u.a:.3f} m</b> to b = <b>{u.b:.3f} m</b> (Span = {u.b - u.a:.3f} m)", body_style)],
                    [Paragraph("&nbsp;&nbsp;• Endpoint Intensities", body_style), Paragraph(f"w(a) = <b>{u.w_a:.2f} kN/m</b> at start (x = {u.a:.3f} m), &nbsp; w(b) = <b>{u.w_b:.2f} kN/m</b> at end (x = {u.b:.3f} m)", body_style)],
                    [Paragraph("&nbsp;&nbsp;• Resultant &amp; Centroid", body_style), Paragraph(f"W = (w_a+w_b)/2·(b−a) = <b>{R_u:.3f} kN</b> acting at centroid x̄ = <b>{x_bar_u:.3f} m</b>", body_style)],
                ])
    else:
        config_rows.append([Paragraph("Distributed Load", body_style), Paragraph("None applied", body_style)])


    point_moments = getattr(config, "point_moments", [])
    if point_moments:
        for idx, pm in enumerate(point_moments, 1):
            dir_str = "Clockwise" if pm.magnitude > 0 else "Counter-Clockwise"
            config_rows.append([
                [Paragraph(f"Applied Moment #{idx}", body_style), Paragraph(f"<b>{abs(pm.magnitude):.2f} kN·m</b> ({dir_str}) at position x = <b>{pm.position:.3f} m</b>", body_style)]
            ])

    t_config = Table(config_rows, colWidths=[200, 332])
    t_config.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EDF2F7")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(t_config)
    story.append(Spacer(1, 10))


    story.append(Paragraph("2. Static Equilibrium &amp; Support Reaction Derivations", h1_style))


    sum_P_parts = [f"{pl.magnitude:.1f}" for pl in config.point_loads if abs(pl.magnitude) > 1e-9]
    sum_M_parts = [f"({pl.magnitude:.1f} × {pl.position:.2f})" for pl in config.point_loads if abs(pl.magnitude) > 1e-9]

    for idx, u in enumerate(all_uvls, 1):
        R_u, x_bar_u = uvl_resultant(u)
        if R_u > 1e-9:
            tag = f"UDL#{idx}" if abs(u.w_a - u.w_b) < 1e-6 else f"UVL#{idx}"
            sum_P_parts.append(f"{R_u:.1f} ({tag})")
            sum_M_parts.append(f"({R_u:.1f} × {x_bar_u:.2f} {tag})")


    for idx, pm in enumerate(point_moments, 1):
        if abs(pm.magnitude) > 1e-9:
            sum_M_parts.append(f"{pm.magnitude:+.1f} (M#{idx})")

    sum_P_str = " + ".join(sum_P_parts) if sum_P_parts else "0.0"
    sum_M_str = " + ".join(sum_M_parts) if sum_M_parts else "0.0"

    eq_text = (
        f"<b>Moment Equilibrium about Support A (x = 0):</b><br/>"
        f"&nbsp;&nbsp;Σ M_A = 0 &nbsp;⇒&nbsp; R_B · L = Σ (P_i · x_i) + Σ (R_uvl · x̄_uvl) + Σ M_j<br/>"
        f"&nbsp;&nbsp;R_B × {L:.2f} m = {sum_M_str}<br/>"
        f"&nbsp;&nbsp;<b>R_B = {R_B:.3f} kN</b><br/><br/>"
        f"<b>Vertical Force Equilibrium:</b><br/>"
        f"&nbsp;&nbsp;Σ F_y = 0 &nbsp;⇒&nbsp; R_A + R_B = Σ P_i + Σ R_uvl<br/>"
        f"&nbsp;&nbsp;R_A + {R_B:.3f} = {sum_P_str} = {audit['total_load']:.3f} kN<br/>"
        f"&nbsp;&nbsp;<b>R_A = {R_A:.3f} kN</b>"
    )
    story.append(Paragraph(eq_text, body_style))
    story.append(Spacer(1, 8))


    audit_status_str = "PASSED (< 1e-9 tolerance)" if audit["is_balanced"] else "FAILED"
    audit_rows = [
        [Paragraph("<b>Equilibrium Verification Check</b>", body_bold), Paragraph("<b>Residual Value</b>", body_bold), Paragraph("<b>Audit Status</b>", body_bold)],
        [Paragraph("Vertical Equilibrium: |(R_A + R_B) - W_total|", body_style), Paragraph(f"{abs(audit['residual_vertical']):.2e} kN", body_style), Paragraph(audit_status_str, body_bold)],
        [Paragraph("Moment Equilibrium about A: |R_B · L - M_loads|", body_style), Paragraph(f"{abs(audit['residual_moment_A']):.2e} kN·m", body_style), Paragraph(audit_status_str, body_bold)],
        [Paragraph("Moment Equilibrium about B: |R_A · L - M_loads_B|", body_style), Paragraph(f"{abs(audit['residual_moment_B']):.2e} kN·m", body_style), Paragraph(audit_status_str, body_bold)],
    ]
    t_audit = Table(audit_rows, colWidths=[270, 140, 122])
    t_audit.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#FEFCBF")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(t_audit)
    story.append(Spacer(1, 12))


    story.append(Paragraph("3. Critical Sectional Values &amp; Extrema", h1_style))

    max_pos = extrema["max_positive_value"]
    x_pos = extrema["max_positive_pos"]
    min_neg = extrema["min_negative_value"]
    x_neg = extrema["min_negative_pos"]

    v_start = shear_force(config, 1e-9)
    v_end = shear_force(config, L - 1e-9)

    extrema_rows = [
        [Paragraph("<b>Response Quantity</b>", body_bold), Paragraph("<b>Extremum Value</b>", body_bold), Paragraph("<b>Position (x)</b>", body_bold), Paragraph("<b>Evaluation Method</b>", body_bold)],
        [Paragraph("Max Sagging Moment (+M)", body_style), Paragraph(f"<b>{max_pos:.3f} kN·m</b>", body_style), Paragraph(f"{x_pos:.3f} m", body_style), Paragraph("Exact Zero-Shear Root / Critical Point", body_style)],
        [Paragraph("Max Hogging Moment (-M)", body_style), Paragraph(f"{min_neg:.3f} kN·m" if min_neg < -1e-6 else "None (0.000 kN·m)", body_style), Paragraph(f"{x_neg:.3f} m" if min_neg < -1e-6 else "N/A", body_style), Paragraph("Zero on simple span (downward loads)", body_style)],
        [Paragraph("Max Positive Shear (+V)", body_style), Paragraph(f"+{v_start:.3f} kN", body_style), Paragraph("x = 0⁺ m", body_style), Paragraph("At Left Support R_A", body_style)],
        [Paragraph("Max Negative Shear (-V)", body_style), Paragraph(f"{v_end:.3f} kN", body_style), Paragraph(f"x = {L:.3f}⁻ m", body_style), Paragraph("At Right Support -R_B", body_style)],
    ]
    t_extrema = Table(extrema_rows, colWidths=[160, 110, 92, 170])
    t_extrema.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EDF2F7")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(t_extrema)


    contraflexure_pts = extrema.get("contraflexure_points", [])
    if contraflexure_pts:
        story.append(Spacer(1, 8))
        story.append(Paragraph("<b>Points of Contraflexure (Moment Sign Changes, M(x) = 0):</b>", body_bold))
        cp_rows = [
            [Paragraph("<b>#</b>", body_bold), Paragraph("<b>Position (x)</b>", body_bold), Paragraph("<b>Moment M(x)</b>", body_bold), Paragraph("<b>Transition Type</b>", body_bold)]
        ]
        for c_i, cp in enumerate(contraflexure_pts, 1):
            cp_rows.append([
                Paragraph(f"{c_i}", body_style),
                Paragraph(f"<b>{cp.position:.4f} m</b>", body_style),
                Paragraph(f"{cp.moment:.4f} kN·m", body_style),
                Paragraph(cp.type, body_style),
            ])
        t_cp = Table(cp_rows, colWidths=[40, 160, 160, 172])
        t_cp.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#D1FAE5")),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#A7F3D0")),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]))
        story.append(t_cp)

    story.append(Spacer(1, 12))

    if section_props is not None:
        story.append(Paragraph("4. Cross-Section Geometric Properties &amp; Stress Analysis", h1_style))
        sec_rows = [
            [Paragraph("<b>Property</b>", body_bold), Paragraph("<b>Value &amp; Unit</b>", body_bold), Paragraph("<b>Property</b>", body_bold), Paragraph("<b>Value &amp; Unit</b>", body_bold)],
            [Paragraph("Cross-Section Shape", body_style), Paragraph(f"<b>{section_props.shape_name}</b>", body_style), Paragraph("Total Area (A)", body_style), Paragraph(f"{section_props.area:.2f} mm² ({section_props.area * 1e-6:.6f} m²)", body_style)],
            [Paragraph("Centroid Height (ȳ)", body_style), Paragraph(f"{getattr(section_props, 'y_bar', getattr(section_props, 'centroid_y', 0.0)):.2f} mm", body_style), Paragraph("Overall Depth (h)", body_style), Paragraph(f"{getattr(section_props, 'total_height', getattr(section_props, 'height', 0.0)):.2f} mm", body_style)],
            [Paragraph("Second Moment of Area (I_xx)", body_style), Paragraph(f"{section_props.I_xx:.2e} mm⁴", body_style), Paragraph("Second Moment of Area (I_yy)", body_style), Paragraph(f"{section_props.I_yy:.2e} mm⁴", body_style)],
            [Paragraph("Elastic Modulus (Z_xx,top)", body_style), Paragraph(f"{section_props.Z_xx_top:.2e} mm³", body_style), Paragraph("Elastic Modulus (Z_xx,bottom)", body_style), Paragraph(f"{section_props.Z_xx_bottom:.2e} mm³", body_style)],
        ]
        if stress_results is not None:
            sec_rows.extend([
                [Paragraph("<b>Max Normal Stress (σ_max)</b>", body_bold), Paragraph(f"<b>{stress_results.sigma_max:.2f} MPa</b> (@ x = {stress_results.sigma_max_location:.3f} m)", body_bold),
                 Paragraph("<b>Max Shear Stress (τ_max)</b>", body_bold), Paragraph(f"<b>{stress_results.tau_max:.2f} MPa</b> (@ x = {stress_results.tau_max_location:.3f} m)", body_bold)],
            ])
            if stress_results.deflection_curve is not None:
                sec_rows.append([
                    Paragraph("<b>Max Deflection (δ_max)</b>", body_bold),
                    Paragraph(f"<b>{abs(stress_results.deflection_curve.max_deflection * 1000.0):.3f} mm</b> (@ x = {stress_results.deflection_curve.max_deflection_location:.3f} m)", body_bold),
                    Paragraph("Span/Deflection Ratio (L/δ)", body_style),
                    Paragraph(f"L / {stress_results.deflection_curve.span_to_deflection_ratio:.0f}" if stress_results.deflection_curve.span_to_deflection_ratio > 0 else "—", body_style)
                ])
        t_sec = Table(sec_rows, colWidths=[140, 126, 140, 126])
        t_sec.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F3E8FF")),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#D8B4FE")),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]))
        story.append(t_sec)
        story.append(Spacer(1, 10))


    section_num = 5 if section_props is not None else 4
    story.append(Paragraph(f"{section_num}. Structural Diagrams (Loading, SFD &amp; BMD)", h1_style))

    fig, axes = plot_beam_diagrams(config, show=False, figsize=(9.0, 10.5))
    img_buf = io.BytesIO()
    fig.savefig(img_buf, format="png", dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)  
    img_buf.seek(0)


    report_img = Image(img_buf, width=510, height=595)
    story.append(report_img)
    story.append(Spacer(1, 12))

    section_num += 1
    story.append(Paragraph(f"{section_num}. Numerical Sectional Results Table", h1_style))
    story.append(
        Paragraph(
            "<b>Shear Convention at Discontinuities:</b> At concentrated load locations x_p, "
            "V(x_p⁻) is the shear immediately left, and V(x_p⁺) is the shear immediately right. "
            "Bending moment M(x) is continuous across all sections.",
            callout_style,
        )
    )
    story.append(Spacer(1, 4))

    crit_xs = {0.0, L}
    for pl in config.point_loads:
        crit_xs.add(pl.position)
    for u in all_uvls:
        crit_xs.add(u.a)
        crit_xs.add(u.b)
    for pm in point_moments:
        crit_xs.add(pm.position)
    for cp in contraflexure_pts:
        crit_xs.add(cp.position)

    grid_xs = set(np.linspace(0.0, L, max(num_table_rows, 5)))
    all_table_xs = sorted(crit_xs | grid_xs)

    table_data = [
        [
            Paragraph("<b>x (m)</b>", body_bold),
            Paragraph("<b>V(x⁻) (kN)</b>", body_bold),
            Paragraph("<b>V(x⁺) (kN)</b>", body_bold),
            Paragraph("<b>M(x) (kN·m)</b>", body_bold),
            Paragraph("<b>Section Type</b>", body_bold),
        ]
    ]

    _EPS = 1e-9
    for x in all_table_xs:
        is_pl = any(abs(pl.position - x) < 1e-7 for pl in config.point_loads)
        is_udl_bd = any(abs(u.w_a - u.w_b) < 1e-6 and (abs(u.a - x) < 1e-7 or abs(u.b - x) < 1e-7) for u in all_uvls)
        is_uvl_bd = any(abs(u.w_a - u.w_b) >= 1e-6 and (abs(u.a - x) < 1e-7 or abs(u.b - x) < 1e-7) for u in all_uvls)

        if x == 0.0:
            vl_str = "—"
            vr_str = f"{shear_force(config, _EPS):.3f}"
            sec_type = "Pin Support A"
        elif abs(x - L) < 1e-7:
            vl_str = f"{shear_force(config, L - _EPS):.3f}"
            vr_str = "—"
            sec_type = "Roller Support B"
        elif is_pl:
            vl_str = f"{shear_force(config, x - _EPS):.3f}"
            vr_str = f"{shear_force(config, x + _EPS):.3f}"
            sec_type = "Point Load Discontinuity"
        else:
            v_val = shear_force(config, x)
            vl_str = f"{v_val:.3f}"
            vr_str = f"{v_val:.3f}"
            sec_type = "UDL Boundary" if is_udl_bd else ("UVL Boundary" if is_uvl_bd else "Continuous Station")


        m_val = bending_moment(config, x)

        table_data.append([
            Paragraph(f"{x:.3f}", body_style),
            Paragraph(vl_str, body_style),
            Paragraph(vr_str, body_style),
            Paragraph(f"{m_val:.3f}", body_style),
            Paragraph(sec_type, body_style),
        ])

    t_stations = Table(table_data, colWidths=[65, 95, 95, 110, 167])
    t_stations.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EDF2F7")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
        ("TOPPADDING", (0, 0), (-1, -1), 2.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
    ]))
    story.append(t_stations)
    story.append(Spacer(1, 12))

    story.append(Paragraph("6. Engineering Assumptions &amp; Scope Limitations", h1_style))
    limitations_text = (
        "• <b>Boundary Conditions:</b> Valid exclusively for simply supported beams (pin at x = 0, roller at x = L). "
        "Overhanging cantilevers, continuous spans, and internal hinges are not supported.<br/>"
        "• <b>Static Determinacy:</b> Analysis assumes linear elastic small-deflection Euler-Bernoulli beam theory.<br/>"
        "• <b>Loading Constraints:</b> Supports arbitrary concentrated downward point loads (≥ 3 required) and up to one "
        "continuous uniformly varying load (UVL) segment (triangular, trapezoidal, or uniform).<br/>"
        "• <b>Internal Sign Conventions:</b> Downward applied loads are positive; upward reactions are positive. "
        "Shear force V(x) is positive when the left segment pushes upward. Bending moment M(x) is positive for sagging tension in the bottom fibres.<br/>"
        "• <b>Exclusions:</b> No structural compliance checks, shear deformation (Timoshenko), deflection, or material yield checks are implied."
    )
    story.append(Paragraph(limitations_text, body_style))

    doc.build(story)

    if output_path_or_buffer is None:
        return buffer.getvalue()
    elif isinstance(output_path_or_buffer, (str, os.PathLike)):
        buffer.close()
        with open(output_path_or_buffer, "rb") as f:
            return f.read()
    else:
        return buffer.getvalue() if hasattr(buffer, "getvalue") else b""
