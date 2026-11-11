from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional, List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.figure import Figure
from matplotlib.axes import Axes


@dataclass
class SectionProperties:

    shape_name: str
    dimensions: Dict[str, float]
    area: float                 # mm²
    area_cm2: float             # cm²
    x_bar: float                # mm from left
    y_bar: float                # mm from bottom
    I_xx: float                 # mm⁴ about centroidal horizontal axis
    I_xx_cm4: float             # cm⁴
    I_yy: float                 # mm⁴ about centroidal vertical axis
    I_yy_cm4: float             # cm⁴
    y_top: float                # mm (distance from centroid to top extreme fibre)
    y_bottom: float             # mm (distance from centroid to bottom extreme fibre)
    x_left: float               # mm (distance from centroid to left extreme fibre)
    x_right: float              # mm (distance from centroid to right extreme fibre)
    Z_xx_top: float             # mm³
    Z_xx_bottom: float          # mm³
    Z_xx_min: float             # mm³
    Z_yy: float                 # mm³
    r_xx: float                 # mm
    r_yy: float                 # mm
    total_height: float         # mm
    total_width: float          # mm

    @property
    def centroid_y(self) -> float:
        return self.y_bar

    @property
    def height(self) -> float:
        return self.total_height

    @property
    def width(self) -> float:
        return self.total_width

    def to_dict(self) -> Dict[str, Any]:
        return {
            "Shape": self.shape_name,
            "Area (mm²)": self.area,
            "Area (cm²)": self.area_cm2,
            "Centroid x̄ (mm)": self.x_bar,
            "Centroid ȳ (mm)": self.y_bar,
            "I_xx (mm⁴)": self.I_xx,
            "I_xx (cm⁴)": self.I_xx_cm4,
            "I_yy (mm⁴)": self.I_yy,
            "I_yy (cm⁴)": self.I_yy_cm4,
            "Z_xx,min (mm³)": self.Z_xx_min,
            "Z_xx,min (cm³)": self.Z_xx_min / 1e3,
            "Z_yy (mm³)": self.Z_yy,
            "Z_yy (cm³)": self.Z_yy / 1e3,
            "r_xx (mm)": self.r_xx,
            "r_yy (mm)": self.r_yy,
        }


class CrossSection(ABC):


    @abstractmethod
    def compute_properties(self) -> SectionProperties:
        pass

    @abstractmethod
    def max_shear_stress(self, V_kN: float, props: SectionProperties) -> float:
        pass

    @abstractmethod
    def draw(self, ax: Axes, props: SectionProperties) -> None:
        pass

class RectangularSection(CrossSection):

    def __init__(self, b: float, h: float):
        if not (math.isfinite(b) and math.isfinite(h)):
            raise ValueError("Dimensions must be finite numbers.")
        if b <= 0.0 or h <= 0.0:
            raise ValueError(f"Width and height must be strictly positive. Got b={b}, h={h}.")
        self.b = float(b)
        self.h = float(h)

    def compute_properties(self) -> SectionProperties:
        b, h = self.b, self.h
        A = b * h
        x_bar = b / 2.0
        y_bar = h / 2.0
        I_xx = (b * h**3) / 12.0
        I_yy = (h * b**3) / 12.0
        y_top = h / 2.0
        y_bot = h / 2.0
        x_left = b / 2.0
        x_right = b / 2.0
        Z_xx = I_xx / y_top
        Z_yy = I_yy / x_right
        r_xx = math.sqrt(I_xx / A)
        r_yy = math.sqrt(I_yy / A)

        return SectionProperties(
            shape_name="Rectangular",
            dimensions={"Width b (mm)": b, "Height h (mm)": h},
            area=A,
            area_cm2=A / 100.0,
            x_bar=x_bar,
            y_bar=y_bar,
            I_xx=I_xx,
            I_xx_cm4=I_xx / 1e4,
            I_yy=I_yy,
            I_yy_cm4=I_yy / 1e4,
            y_top=y_top,
            y_bottom=y_bot,
            x_left=x_left,
            x_right=x_right,
            Z_xx_top=Z_xx,
            Z_xx_bottom=Z_xx,
            Z_xx_min=Z_xx,
            Z_yy=Z_yy,
            r_xx=r_xx,
            r_yy=r_yy,
            total_height=h,
            total_width=b,
        )

    def max_shear_stress(self, V_kN: float, props: SectionProperties) -> float:
        # τ_max = 1.5 * V / A at neutral axis
        V_N = abs(V_kN) * 1000.0
        return 1.5 * V_N / props.area

    def draw(self, ax: Axes, props: SectionProperties) -> None:
        b, h = self.b, self.h
        rect = mpatches.Rectangle((0, 0), b, h, facecolor="#0284C7", edgecolor="#38BDF8", lw=2, alpha=0.65)
        ax.add_patch(rect)

class SolidCircularSection(CrossSection):
    def __init__(self, d: float):
        if not math.isfinite(d):
            raise ValueError("Diameter must be a finite number.")
        if d <= 0.0:
            raise ValueError(f"Diameter must be strictly positive. Got d={d}.")
        self.d = float(d)

    def compute_properties(self) -> SectionProperties:
        d = self.d
        r = d / 2.0
        A = math.pi * d**2 / 4.0
        x_bar = r
        y_bar = r
        I = math.pi * d**4 / 64.0
        Z = math.pi * d**3 / 32.0
        r_gyr = d / 4.0

        return SectionProperties(
            shape_name="Solid Circular",
            dimensions={"Diameter d (mm)": d},
            area=A,
            area_cm2=A / 100.0,
            x_bar=x_bar,
            y_bar=y_bar,
            I_xx=I,
            I_xx_cm4=I / 1e4,
            I_yy=I,
            I_yy_cm4=I / 1e4,
            y_top=r,
            y_bottom=r,
            x_left=r,
            x_right=r,
            Z_xx_top=Z,
            Z_xx_bottom=Z,
            Z_xx_min=Z,
            Z_yy=Z,
            r_xx=r_gyr,
            r_yy=r_gyr,
            total_height=d,
            total_width=d,
        )

    def max_shear_stress(self, V_kN: float, props: SectionProperties) -> float:
        # τ_max = 4/3 * V / A
        V_N = abs(V_kN) * 1000.0
        return (4.0 / 3.0) * V_N / props.area

    def draw(self, ax: Axes, props: SectionProperties) -> None:
        r = self.d / 2.0
        circle = mpatches.Circle((r, r), r, facecolor="#0284C7", edgecolor="#38BDF8", lw=2, alpha=0.65)
        ax.add_patch(circle)

class HollowCircularSection(CrossSection):

    def __init__(self, D: Optional[float] = None, d: Optional[float] = None, *, d_out: Optional[float] = None, d_in: Optional[float] = None):
        D_val = D if D is not None else d_out
        d_val = d if d is not None else d_in
        if D_val is None or d_val is None:
            raise ValueError("Both outer diameter (D or d_out) and inner diameter (d or d_in) are required.")
        if not (math.isfinite(D_val) and math.isfinite(d_val)):
            raise ValueError("Diameters must be finite numbers.")
        if D_val <= 0.0 or d_val <= 0.0:
            raise ValueError(f"Outer and inner diameter must be positive. Got D={D_val}, d={d_val}.")
        if d_val >= D_val:
            raise ValueError(f"Inner diameter d={d_val} must be strictly less than outer diameter D={D_val}.")
        self.D = float(D_val)
        self.d = float(d_val)
        self.t = (self.D - self.d) / 2.0

    def compute_properties(self) -> SectionProperties:
        D, d = self.D, self.d
        R = D / 2.0
        A = math.pi * (D**2 - d**2) / 4.0
        x_bar = R
        y_bar = R
        I = math.pi * (D**4 - d**4) / 64.0
        Z = I / R
        r_gyr = math.sqrt(D**2 + d**2) / 4.0

        return SectionProperties(
            shape_name="Hollow Circular",
            dimensions={
                "Outer Diameter D (mm)": D,
                "Inner Diameter d (mm)": d,
                "Wall Thickness t (mm)": self.t,
            },
            area=A,
            area_cm2=A / 100.0,
            x_bar=x_bar,
            y_bar=y_bar,
            I_xx=I,
            I_xx_cm4=I / 1e4,
            I_yy=I,
            I_yy_cm4=I / 1e4,
            y_top=R,
            y_bottom=R,
            x_left=R,
            x_right=R,
            Z_xx_top=Z,
            Z_xx_bottom=Z,
            Z_xx_min=Z,
            Z_yy=Z,
            r_xx=r_gyr,
            r_yy=r_gyr,
            total_height=D,
            total_width=D,
        )

    def max_shear_stress(self, V_kN: float, props: SectionProperties) -> float:
        V_N = abs(V_kN) * 1000.0
        D, d = self.D, self.d
        factor = (D**2 + D * d + d**2) / (D**2 + d**2)
        return (4.0 / 3.0) * (V_N / props.area) * factor

    def draw(self, ax: Axes, props: SectionProperties) -> None:
        R = self.D / 2.0
        r_in = self.d / 2.0
        circle_out = mpatches.Circle((R, R), R, facecolor="#0284C7", edgecolor="#38BDF8", lw=2, alpha=0.65)
        circle_in = mpatches.Circle((R, R), r_in, facecolor="#161D27", edgecolor="#38BDF8", lw=1.5, zorder=3)
        ax.add_patch(circle_out)
        ax.add_patch(circle_in)

class ISection(CrossSection):

    def __init__(self, H: Optional[float] = None, B: Optional[float] = None, t_f: Optional[float] = None, t_w: Optional[float] = None, *, b: Optional[float] = None, h: Optional[float] = None):
        H_val = H if H is not None else h
        B_val = B if B is not None else b
        if H_val is None or B_val is None or t_f is None or t_w is None:
            raise ValueError("All dimensions (height, width, flange thickness, web thickness) are required.")
        if not all(math.isfinite(v) for v in (H_val, B_val, t_f, t_w)):
            raise ValueError("All dimensions must be finite numbers.")
        if any(v <= 0 for v in (H_val, B_val, t_f, t_w)):
            raise ValueError("All dimensions must be strictly positive.")
        if 2 * t_f >= H_val:
            raise ValueError(f"Total flange thickness (2*t_f = {2*t_f}) must be less than height H = {H_val}.")
        if t_w >= B_val:
            raise ValueError(f"Web thickness t_w = {t_w} must be less than flange width B = {B_val}.")
        self.H = float(H_val)
        self.B = float(B_val)
        self.t_f = float(t_f)
        self.t_w = float(t_w)

    def compute_properties(self) -> SectionProperties:
        H, B, t_f, t_w = self.H, self.B, self.t_f, self.t_w
        h_w = H - 2 * t_f  
        A = 2 * B * t_f + h_w * t_w
        x_bar = B / 2.0
        y_bar = H / 2.0

        I_xx = (B * H**3 - (B - t_w) * h_w**3) / 12.0

        I_yy = 2 * (t_f * B**3 / 12.0) + (h_w * t_w**3 / 12.0)

        y_half = H / 2.0
        x_half = B / 2.0
        Z_xx = I_xx / y_half
        Z_yy = I_yy / x_half
        r_xx = math.sqrt(I_xx / A)
        r_yy = math.sqrt(I_yy / A)

        return SectionProperties(
            shape_name="I-Section",
            dimensions={
                "Height H (mm)": H,
                "Flange Width B (mm)": B,
                "Flange Thickness t_f (mm)": t_f,
                "Web Thickness t_w (mm)": t_w,
            },
            area=A,
            area_cm2=A / 100.0,
            x_bar=x_bar,
            y_bar=y_bar,
            I_xx=I_xx,
            I_xx_cm4=I_xx / 1e4,
            I_yy=I_yy,
            I_yy_cm4=I_yy / 1e4,
            y_top=y_half,
            y_bottom=y_half,
            x_left=x_half,
            x_right=x_half,
            Z_xx_top=Z_xx,
            Z_xx_bottom=Z_xx,
            Z_xx_min=Z_xx,
            Z_yy=Z_yy,
            r_xx=r_xx,
            r_yy=r_yy,
            total_height=H,
            total_width=B,
        )

    def max_shear_stress(self, V_kN: float, props: SectionProperties) -> float:
        V_N = abs(V_kN) * 1000.0
        H, B, t_f, t_w = self.H, self.B, self.t_f, self.t_w
        h_w_half = (H / 2.0) - t_f
        Q_NA = (B * t_f) * (H / 2.0 - t_f / 2.0) + (t_w * h_w_half) * (h_w_half / 2.0)
        tau_max = (V_N * Q_NA) / (props.I_xx * t_w)
        return tau_max

    def draw(self, ax: Axes, props: SectionProperties) -> None:
        H, B, t_f, t_w = self.H, self.B, self.t_f, self.t_w
        x_web = (B - t_w) / 2.0
        h_w = H - 2 * t_f

        pts = [
            (0, 0), (B, 0), (B, t_f), (x_web + t_w, t_f),
            (x_web + t_w, t_f + h_w), (B, t_f + h_w), (B, H),
            (0, H), (0, t_f + h_w), (x_web, t_f + h_w),
            (x_web, t_f), (0, t_f),
        ]
        poly = plt.Polygon(pts, closed=True, facecolor="#0284C7", edgecolor="#38BDF8", lw=2, alpha=0.65)
        ax.add_patch(poly)

class TSection(CrossSection):
    def __init__(self, H: Optional[float] = None, B: Optional[float] = None, t_f: Optional[float] = None, t_w: Optional[float] = None, *, b: Optional[float] = None, h: Optional[float] = None):
        H_val = H if H is not None else h
        B_val = B if B is not None else b
        if H_val is None or B_val is None or t_f is None or t_w is None:
            raise ValueError("All dimensions (height, width, flange thickness, web thickness) are required.")
        if not all(math.isfinite(v) for v in (H_val, B_val, t_f, t_w)):
            raise ValueError("All dimensions must be finite numbers.")
        if any(v <= 0 for v in (H_val, B_val, t_f, t_w)):
            raise ValueError("All dimensions must be strictly positive.")
        if t_f >= H_val:
            raise ValueError(f"Flange thickness t_f={t_f} must be less than height H={H_val}.")
        if t_w >= B_val:
            raise ValueError(f"Web thickness t_w={t_w} must be less than flange width B={B_val}.")
        self.H = float(H_val)
        self.B = float(B_val)
        self.t_f = float(t_f)
        self.t_w = float(t_w)

    def compute_properties(self) -> SectionProperties:
        H, B, t_f, t_w = self.H, self.B, self.t_f, self.t_w
        h_w = H - t_f
        A_f = B * t_f
        A_w = h_w * t_w
        A = A_f + A_w

        y_w = h_w / 2.0
        y_f = h_w + t_f / 2.0
        y_bar = (A_w * y_w + A_f * y_f) / A
        x_bar = B / 2.0

        I_xx_w = (t_w * h_w**3) / 12.0 + A_w * (y_w - y_bar)**2
        I_xx_f = (B * t_f**3) / 12.0 + A_f * (y_f - y_bar)**2
        I_xx = I_xx_w + I_xx_f

        I_yy = (t_f * B**3) / 12.0 + (h_w * t_w**3) / 12.0

        y_top = H - y_bar
        y_bot = y_bar
        x_half = B / 2.0
        Z_xx_top = I_xx / y_top
        Z_xx_bot = I_xx / y_bot
        Z_xx_min = min(Z_xx_top, Z_xx_bot)
        Z_yy = I_yy / x_half
        r_xx = math.sqrt(I_xx / A)
        r_yy = math.sqrt(I_yy / A)

        return SectionProperties(
            shape_name="T-Section",
            dimensions={
                "Height H (mm)": H,
                "Flange Width B (mm)": B,
                "Flange Thickness t_f (mm)": t_f,
                "Web Thickness t_w (mm)": t_w,
            },
            area=A,
            area_cm2=A / 100.0,
            x_bar=x_bar,
            y_bar=y_bar,
            I_xx=I_xx,
            I_xx_cm4=I_xx / 1e4,
            I_yy=I_yy,
            I_yy_cm4=I_yy / 1e4,
            y_top=y_top,
            y_bottom=y_bot,
            x_left=x_half,
            x_right=x_half,
            Z_xx_top=Z_xx_top,
            Z_xx_bottom=Z_xx_bot,
            Z_xx_min=Z_xx_min,
            Z_yy=Z_yy,
            r_xx=r_xx,
            r_yy=r_yy,
            total_height=H,
            total_width=B,
        )

    def max_shear_stress(self, V_kN: float, props: SectionProperties) -> float:
        V_N = abs(V_kN) * 1000.0
        if props.y_bar <= (self.H - self.t_f):
            Q_NA = (self.t_w * props.y_bar) * (props.y_bar / 2.0)
            b_na = self.t_w
        else:
            y_from_top = self.H - props.y_bar
            Q_NA = (self.B * y_from_top) * (y_from_top / 2.0)
            b_na = self.B
        return (V_N * Q_NA) / (props.I_xx * b_na)

    def draw(self, ax: Axes, props: SectionProperties) -> None:
        H, B, t_f, t_w = self.H, self.B, self.t_f, self.t_w
        x_web = (B - t_w) / 2.0
        h_w = H - t_f
        pts = [
            (x_web, 0), (x_web + t_w, 0), (x_web + t_w, h_w),
            (B, h_w), (B, H), (0, H), (0, h_w), (x_web, h_w),
        ]
        poly = plt.Polygon(pts, closed=True, facecolor="#0284C7", edgecolor="#38BDF8", lw=2, alpha=0.65)
        ax.add_patch(poly)

class LSection(CrossSection):
    def __init__(self, H: Optional[float] = None, B: Optional[float] = None, t_v: Optional[float] = None, t_h: Optional[float] = None, *, b: Optional[float] = None, h: Optional[float] = None, t: Optional[float] = None):
        H_val = H if H is not None else h
        B_val = B if B is not None else b
        tv_val = t_v if t_v is not None else t
        th_val = t_h if t_h is not None else (t if t is not None else tv_val)
        if H_val is None or B_val is None or tv_val is None or th_val is None:
            raise ValueError("All dimensions (height, width, thickness) are required.")
        if not all(math.isfinite(v) for v in (H_val, B_val, tv_val, th_val)):
            raise ValueError("All dimensions must be finite numbers.")
        if any(v <= 0 for v in (H_val, B_val, tv_val, th_val)):
            raise ValueError("All dimensions must be strictly positive.")
        if th_val >= H_val:
            raise ValueError(f"Horizontal leg thickness t_h={th_val} must be less than height H={H_val}.")
        if tv_val >= B_val:
            raise ValueError(f"Vertical leg thickness t_v={tv_val} must be less than width B={B_val}.")
        self.H = float(H_val)
        self.B = float(B_val)
        self.t_v = float(tv_val)
        self.t_h = float(th_val)

    def compute_properties(self) -> SectionProperties:
        H, B, t_v, t_h = self.H, self.B, self.t_v, self.t_h
        A1 = B * t_h
        x1 = B / 2.0
        y1 = t_h / 2.0

        h2 = H - t_h
        A2 = t_v * h2
        x2 = t_v / 2.0
        y2 = t_h + h2 / 2.0

        A = A1 + A2
        x_bar = (A1 * x1 + A2 * x2) / A
        y_bar = (A1 * y1 + A2 * y2) / A

        I_xx1 = (B * t_h**3) / 12.0 + A1 * (y1 - y_bar)**2
        I_xx2 = (t_v * h2**3) / 12.0 + A2 * (y2 - y_bar)**2
        I_xx = I_xx1 + I_xx2

        I_yy1 = (t_h * B**3) / 12.0 + A1 * (x1 - x_bar)**2
        I_yy2 = (h2 * t_v**3) / 12.0 + A2 * (x2 - x_bar)**2
        I_yy = I_yy1 + I_yy2

        y_top = H - y_bar
        y_bot = y_bar
        x_left = x_bar
        x_right = B - x_bar

        Z_xx_top = I_xx / y_top
        Z_xx_bot = I_xx / y_bot
        Z_xx_min = min(Z_xx_top, Z_xx_bot)
        Z_yy = I_yy / max(x_left, x_right)
        r_xx = math.sqrt(I_xx / A)
        r_yy = math.sqrt(I_yy / A)

        return SectionProperties(
            shape_name="L-Section",
            dimensions={
                "Vertical Height H (mm)": H,
                "Horizontal Width B (mm)": B,
                "Vertical Thickness t_v (mm)": t_v,
                "Horizontal Thickness t_h (mm)": t_h,
            },
            area=A,
            area_cm2=A / 100.0,
            x_bar=x_bar,
            y_bar=y_bar,
            I_xx=I_xx,
            I_xx_cm4=I_xx / 1e4,
            I_yy=I_yy,
            I_yy_cm4=I_yy / 1e4,
            y_top=y_top,
            y_bottom=y_bot,
            x_left=x_left,
            x_right=x_right,
            Z_xx_top=Z_xx_top,
            Z_xx_bottom=Z_xx_bot,
            Z_xx_min=Z_xx_min,
            Z_yy=Z_yy,
            r_xx=r_xx,
            r_yy=r_yy,
            total_height=H,
            total_width=B,
        )

    def max_shear_stress(self, V_kN: float, props: SectionProperties) -> float:
        V_N = abs(V_kN) * 1000.0
        if props.y_bar >= self.t_h:
            y_above = self.H - props.y_bar
            Q_NA = (self.t_v * y_above) * (y_above / 2.0)
            b_na = self.t_v
        else:
            Q_NA = (self.B * props.y_bar) * (props.y_bar / 2.0)
            b_na = self.B
        return (V_N * Q_NA) / (props.I_xx * b_na)

    def draw(self, ax: Axes, props: SectionProperties) -> None:
        H, B, t_v, t_h = self.H, self.B, self.t_v, self.t_h
        pts = [
            (0, 0), (B, 0), (B, t_h), (t_v, t_h), (t_v, H), (0, H)
        ]
        poly = plt.Polygon(pts, closed=True, facecolor="#0284C7", edgecolor="#38BDF8", lw=2, alpha=0.65)
        ax.add_patch(poly)

class HollowRectangularSection(CrossSection):

    def __init__(self, H: Optional[float] = None, B: Optional[float] = None, t_f: Optional[float] = None, t_w: Optional[float] = None, *, b: Optional[float] = None, h: Optional[float] = None):
        H_val = H if H is not None else h
        B_val = B if B is not None else b
        tf_val = t_f
        tw_val = t_w if t_w is not None else tf_val
        if H_val is None or B_val is None or tf_val is None or tw_val is None:
            raise ValueError("All dimensions (height, width, top/bottom thickness, side thickness) are required.")
        if not all(math.isfinite(v) for v in (H_val, B_val, tf_val, tw_val)):
            raise ValueError("All dimensions must be finite numbers.")
        if any(v <= 0 for v in (H_val, B_val, tf_val, tw_val)):
            raise ValueError("All dimensions must be strictly positive.")
        if 2 * tf_val >= H_val:
            raise ValueError(f"Total top/bottom thickness 2*t_f = {2*tf_val} must be less than height H = {H_val}.")
        if 2 * tw_val >= B_val:
            raise ValueError(f"Total side thickness 2*t_w = {2*tw_val} must be less than width B = {B_val}.")
        self.H = float(H_val)
        self.B = float(B_val)
        self.t_f = float(tf_val)
        self.t_w = float(tw_val)

    def compute_properties(self) -> SectionProperties:
        H, B, t_f, t_w = self.H, self.B, self.t_f, self.t_w
        b_in = B - 2 * t_w
        h_in = H - 2 * t_f
        A = B * H - b_in * h_in
        x_bar = B / 2.0
        y_bar = H / 2.0

        I_xx = (B * H**3 - b_in * h_in**3) / 12.0
        I_yy = (H * B**3 - h_in * b_in**3) / 12.0

        y_half = H / 2.0
        x_half = B / 2.0
        Z_xx = I_xx / y_half
        Z_yy = I_yy / x_half
        r_xx = math.sqrt(I_xx / A)
        r_yy = math.sqrt(I_yy / A)

        return SectionProperties(
            shape_name="Hollow Rectangular (Box)",
            dimensions={
                "Outer Height H (mm)": H,
                "Outer Width B (mm)": B,
                "Flange Thickness t_f (mm)": t_f,
                "Web Thickness t_w (mm)": t_w,
            },
            area=A,
            area_cm2=A / 100.0,
            x_bar=x_bar,
            y_bar=y_bar,
            I_xx=I_xx,
            I_xx_cm4=I_xx / 1e4,
            I_yy=I_yy,
            I_yy_cm4=I_yy / 1e4,
            y_top=y_half,
            y_bottom=y_half,
            x_left=x_half,
            x_right=x_half,
            Z_xx_top=Z_xx,
            Z_xx_bottom=Z_xx,
            Z_xx_min=Z_xx,
            Z_yy=Z_yy,
            r_xx=r_xx,
            r_yy=r_yy,
            total_height=H,
            total_width=B,
        )

    def max_shear_stress(self, V_kN: float, props: SectionProperties) -> float:
        V_N = abs(V_kN) * 1000.0
        H, B, t_f, t_w = self.H, self.B, self.t_f, self.t_w
        h_in = H - 2 * t_f
        Q_NA = (B * t_f) * (H / 2.0 - t_f / 2.0) + (2 * t_w * (h_in / 2.0)) * (h_in / 4.0)
        return (V_N * Q_NA) / (props.I_xx * (2.0 * t_w))

    def draw(self, ax: Axes, props: SectionProperties) -> None:
        H, B, t_f, t_w = self.H, self.B, self.t_f, self.t_w
        b_in = B - 2 * t_w
        h_in = H - 2 * t_f
        outer = mpatches.Rectangle((0, 0), B, H, facecolor="#0284C7", edgecolor="#38BDF8", lw=2, alpha=0.65)
        inner = mpatches.Rectangle((t_w, t_f), b_in, h_in, facecolor="#161D27", edgecolor="#38BDF8", lw=1.5, zorder=3)
        ax.add_patch(outer)
        ax.add_patch(inner)

@dataclass
class BeamStressResults:

    max_bending_moment_kNm: float
    max_shear_force_kN: float
    sigma_top_MPa: float     
    sigma_bottom_MPa: float   
    max_tensile_stress_MPa: float
    max_compressive_stress_MPa: float
    max_shear_stress_MPa: float
    bending_stress_description: str
    assumptions_note: str

    @property
    def sigma_max(self) -> float:
        return self.max_tensile_stress_MPa

    @property
    def tau_max(self) -> float:
        return self.max_shear_stress_MPa

    @property
    def sigma_max_location(self) -> float:
        return 0.0

    @property
    def tau_max_location(self) -> float:
        return 0.0

    @property
    def deflection_curve(self) -> None:
        return None


def calculate_beam_stresses(
    max_pos_moment_kNm: float,
    max_neg_moment_kNm: float,
    max_shear_force_kN: float,
    section: CrossSection,
    props: SectionProperties,
) -> BeamStressResults:

    M_pos_Nmm = max(0.0, max_pos_moment_kNm) * 1e6
    M_neg_Nmm = min(0.0, max_neg_moment_kNm) * 1e6

    sigma_top_sag = -(M_pos_Nmm * props.y_top) / props.I_xx
    sigma_bot_sag = +(M_pos_Nmm * props.y_bottom) / props.I_xx

    sigma_top_hog = -(M_neg_Nmm * props.y_top) / props.I_xx
    sigma_bot_hog = +(M_neg_Nmm * props.y_bottom) / props.I_xx

    all_top = [sigma_top_sag, sigma_top_hog]
    all_bot = [sigma_bot_sag, sigma_bot_hog]

    all_stresses = all_top + all_bot
    max_tension = max(0.0, max(all_stresses))
    max_compression = max(0.0, abs(min(all_stresses))) if min(all_stresses) < 0 else 0.0

    governing_M = max_pos_moment_kNm if abs(max_pos_moment_kNm) >= abs(max_neg_moment_kNm) else max_neg_moment_kNm
    gov_Nmm = governing_M * 1e6
    gov_top = -(gov_Nmm * props.y_top) / props.I_xx
    gov_bot = +(gov_Nmm * props.y_bottom) / props.I_xx

    tau_max = section.max_shear_stress(max_shear_force_kN, props)

    return BeamStressResults(
        max_bending_moment_kNm=governing_M,
        max_shear_force_kN=max_shear_force_kN,
        sigma_top_MPa=gov_top,
        sigma_bottom_MPa=gov_bot,
        max_tensile_stress_MPa=max_tension,
        max_compressive_stress_MPa=max_compression,
        max_shear_stress_MPa=tau_max,
        bending_stress_description=(
            f"Under dominant M = {governing_M:.2f} kN·m: "
            f"Top fibre = {gov_top:.2f} MPa ({'Tension' if gov_top > 0 else 'Compression'}), "
            f"Bottom fibre = {gov_bot:.2f} MPa ({'Tension' if gov_bot > 0 else 'Compression'})."
        ),
        assumptions_note=(
            "Based on Euler-Bernoulli beam theory (σ = My/I, plane sections remain plane). "
            "Linear elastic, homogeneous, isotropic material. Valid for slender beams (L/h ≥ 10)."
        ),
    )


@dataclass
class BeamDeflectionResults:
    """Calculated deflection profile and extrema."""
    E_GPa: float
    EI_kNm2: float          
    max_deflection_mm: float   
    max_deflection_pos_m: float 
    span_deflection_ratio: float
    x_coords: np.ndarray      
    deflection_mm: np.ndarray 
    standard_limit_mm: float    
    passes_L_250: bool
    assumptions_note: str


def calculate_beam_deflection(
    config: Any,
    props: SectionProperties,
    E_GPa: float = 200.0,
    num_points: int = 501,
) -> BeamDeflectionResults:
    try:
        from src.beam_analysis import bending_moment
    except ImportError:
        from beam_analysis import bending_moment

    L = config.length
    E_kNm2 = E_GPa * 1e6
    I_m4 = props.I_xx * 1e-12
    EI = E_kNm2 * I_m4

    xs = np.linspace(0.0, L, num_points)
    dx = L / (num_points - 1)

    M_vals = np.array([bending_moment(config, float(x)) for x in xs])
    curv = M_vals / EI

    theta_raw = np.zeros_like(xs)
    for i in range(1, len(xs)):
        theta_raw[i] = theta_raw[i - 1] + 0.5 * (curv[i - 1] + curv[i]) * dx

    w_raw = np.zeros_like(xs)
    for i in range(1, len(xs)):
        w_raw[i] = w_raw[i - 1] + 0.5 * (theta_raw[i - 1] + theta_raw[i]) * dx

    C1 = -w_raw[-1] / L
    w_m = w_raw + C1 * xs
    w_mm = w_m * 1000.0

    max_idx = int(np.argmax(np.abs(w_mm)))
    max_def = float(np.abs(w_mm[max_idx]))
    max_pos = float(xs[max_idx])

    ratio = (L * 1000.0) / max_def if max_def > 1e-9 else float("inf")
    limit_L_250 = (L * 1000.0) / 250.0
    passes_limit = max_def <= limit_L_250

    return BeamDeflectionResults(
        E_GPa=E_GPa,
        EI_kNm2=EI,
        max_deflection_mm=max_def,
        max_deflection_pos_m=max_pos,
        span_deflection_ratio=ratio,
        x_coords=xs,
        deflection_mm=w_mm,
        standard_limit_mm=limit_L_250,
        passes_L_250=passes_limit,
        assumptions_note=(
            f"Calculated via double integration of curvature M(x)/EI with simply supported "
            f"boundary conditions w(0)=0, w(L)=0. Valid for small linear elastic deflections."
        ),
    )

def plot_cross_section_preview(
    section: CrossSection,
    props: SectionProperties,
    figsize: Tuple[float, float] = (5.0, 5.0),
) -> Figure:
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("#0B0F14")
    ax.set_facecolor("#161D27")

    section.draw(ax, props)

    W = props.total_width
    H = props.total_height
    xc = props.x_bar
    yc = props.y_bar

    margin_x = 0.25 * W
    margin_y = 0.25 * H

    ax.plot([xc - margin_x, xc + margin_x], [yc, yc],
            color="#F43F5E", linestyle="--", linewidth=1.5, zorder=6, label="x-x Axis (NA)")
    ax.plot([xc, xc], [yc - margin_y, yc + margin_y],
            color="#34D399", linestyle="--", linewidth=1.5, zorder=6, label="y-y Axis")

    ax.plot(xc, yc, "o", color="#F43F5E", markersize=6, zorder=7)
    ax.text(xc + 0.04 * W, yc + 0.04 * H,
            f"C ({xc:.1f}, {yc:.1f}) mm",
            fontsize=8.5, fontweight="bold", color="#FDA4AF", zorder=7,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#1E293B", edgecolor="#F43F5E", alpha=0.9))

    ax.set_xlim(-0.15 * W, 1.15 * W)
    ax.set_ylim(-0.15 * H, 1.15 * H)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Width (mm)", fontsize=9, color="#CBD5E1")
    ax.set_ylabel("Height (mm)", fontsize=9, color="#CBD5E1")
    ax.set_title(f"Cross-Section: {props.shape_name}", fontsize=11, fontweight="bold", pad=8, color="#F1F5F9")
    ax.tick_params(colors="#94A3B8")
    for spine in ax.spines.values():
        spine.set_color("#334155")
    ax.grid(True, linestyle=":", linewidth=0.5, color="#334155", alpha=0.5)
    leg = ax.legend(loc="upper right", fontsize=8, framealpha=0.9, facecolor="#161D27", edgecolor="#334155")
    for text in leg.get_texts():
        text.set_color("#E2E8F0")

    return fig
