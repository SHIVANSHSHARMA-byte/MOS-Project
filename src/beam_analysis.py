import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any

import numpy as np


@dataclass
class PointLoad:

    magnitude: float   # kN, positive = downward
    position: float    # m


@dataclass
class PointMoment:

    magnitude: float   # kN·m, positive = clockwise
    position: float    # m


@dataclass
class ContraflexurePoint:

    position: float
    moment: float
    type: str = "continuous_crossing"


@dataclass
class UVL:

    a: float    
    b: float  
    w_a: float  
    w_b: float  

    @property
    def is_udl(self) -> bool:
        return abs(self.w_a - self.w_b) < 1e-9

    @classmethod
    def from_udl(cls, a: float, b: float, w: float) -> "UVL":
        return cls(a=a, b=b, w_a=w, w_b=w)


@dataclass
class UDL:
    a: float
    b: float
    w: float

    @property
    def w_a(self) -> float:
        return self.w

    @property
    def w_b(self) -> float:
        return self.w

    def to_uvl(self) -> UVL:
        return UVL(a=self.a, b=self.b, w_a=self.w, w_b=self.w)


@dataclass
class BeamConfig:
    length: float
    point_loads: List[PointLoad] = field(default_factory=list)
    uvl: Optional[UVL] = None
    uvls: List[UVL] = field(default_factory=list)
    point_moments: List[PointMoment] = field(default_factory=list)

    def get_all_uvls(self) -> List[UVL]:
        all_uvls: List[UVL] = []
        if self.uvl is not None:
            all_uvls.append(self.uvl)
        for u in self.uvls:
            if u not in all_uvls:
                all_uvls.append(u)
        return all_uvls

def validate_beam(config: BeamConfig, min_point_loads: Optional[int] = 3) -> None:

    L = config.length

    if not math.isfinite(L) or L <= 0:
        raise ValueError(f"Beam length must be positive and finite. Got L = {L}.")

    if min_point_loads is not None and len(config.point_loads) < min_point_loads:
        raise ValueError(
            f"At least three point loads are required. "
            f"Got {len(config.point_loads)}."
        )

    for i, pl in enumerate(config.point_loads):
        if not math.isfinite(pl.position) or not (0 <= pl.position <= L):
            raise ValueError(
                f"Point load {i} position {pl.position} m is outside "
                f"[0, {L}] m or non-finite."
            )
        if not math.isfinite(pl.magnitude) or pl.magnitude < 0:
            raise ValueError(
                f"Point load {i} magnitude must be non-negative and finite. "
                f"Got {pl.magnitude}."
            )

    all_uvls = config.get_all_uvls() if hasattr(config, "get_all_uvls") else ([config.uvl] if config.uvl else [])
    for idx, uvl in enumerate(all_uvls):
        if not (math.isfinite(uvl.a) and math.isfinite(uvl.b)):
            raise ValueError(f"UVL {idx} boundary positions must be finite. Got a={uvl.a}, b={uvl.b}.")
        if uvl.a < 0 or uvl.b > L:
            raise ValueError(
                f"UVL {idx} region [{uvl.a}, {uvl.b}] extends outside beam "
                f"span [0, {L}]."
            )
        if uvl.a >= uvl.b:
            raise ValueError(
                f"UVL {idx} start position {uvl.a} must be strictly less than "
                f"end position {uvl.b}."
            )
        if not (math.isfinite(uvl.w_a) and math.isfinite(uvl.w_b)):
            raise ValueError(f"UVL {idx} intensities must be finite. Got w_a={uvl.w_a}, w_b={uvl.w_b}.")
        if uvl.w_a < 0 or uvl.w_b < 0:
            raise ValueError(
                "UVL intensities must be non-negative (downward positive). "
                f"Got w_a = {uvl.w_a}, w_b = {uvl.w_b}."
            )

    moments = getattr(config, "point_moments", [])
    for idx, pm in enumerate(moments):
        if not math.isfinite(pm.position) or not (0 <= pm.position <= L):
            raise ValueError(
                f"Applied moment {idx} position {pm.position} m is outside "
                f"[0, {L}] m or non-finite."
            )
        if not math.isfinite(pm.magnitude):
            raise ValueError(
                f"Applied moment {idx} magnitude must be finite. Got {pm.magnitude}."
            )

def udl_resultant(a: float, b: float, w: float) -> Tuple[float, float]:
    d = b - a
    R = w * d
    x_bar = (a + b) / 2.0
    return R, x_bar


def uvl_resultant(uvl: UVL) -> Tuple[float, float]:
    d = uvl.b - uvl.a           
    R = (uvl.w_a + uvl.w_b) / 2.0 * d

    if abs(uvl.w_a + uvl.w_b) < 1e-12:
        x_bar = (uvl.a + uvl.b) / 2.0
    elif abs(uvl.w_a - uvl.w_b) < 1e-12:
        x_bar = (uvl.a + uvl.b) / 2.0
    else:
        numerator = uvl.w_a * (2 * uvl.a + uvl.b) + uvl.w_b * (uvl.a + 2 * uvl.b)
        x_bar = numerator / (3.0 * (uvl.w_a + uvl.w_b))

    return R, x_bar


def calculate_reactions(config: BeamConfig) -> Tuple[float, float]:
    L = config.length

    total_moment_about_A = 0.0
    total_vertical_load  = 0.0

    for pl in config.point_loads:
        total_vertical_load  += pl.magnitude
        total_moment_about_A += pl.magnitude * pl.position


    all_uvls = config.get_all_uvls() if hasattr(config, "get_all_uvls") else ([config.uvl] if config.uvl else [])
    for uvl in all_uvls:
        R_uvl, x_bar = uvl_resultant(uvl)
        total_vertical_load  += R_uvl
        total_moment_about_A += R_uvl * x_bar

    moments = getattr(config, "point_moments", [])
    for pm in moments:
        total_moment_about_A += pm.magnitude

    R_B = total_moment_about_A / L 
    R_A = total_vertical_load - R_B   

    return R_A, R_B


def total_resultant_load(config: BeamConfig) -> float:

    total = sum(pl.magnitude for pl in config.point_loads)
    all_uvls = config.get_all_uvls() if hasattr(config, "get_all_uvls") else ([config.uvl] if config.uvl else [])
    for uvl in all_uvls:
        R_uvl, _ = uvl_resultant(uvl)
        total += R_uvl
    return total



def _uvl_shear_contribution(uvl: UVL, x: float) -> float:

    a, b = uvl.a, uvl.b
    w_a, w_b = uvl.w_a, uvl.w_b

    if x <= a:
        return 0.0                

    t = min(x, b)                
    d = b - a                    

  
    w_t = w_a + (w_b - w_a) / d * (t - a)

    F = (w_a + w_t) / 2.0 * (t - a)
    return F


def _uvl_moment_contribution(uvl: UVL, x: float) -> float:

    a, b = uvl.a, uvl.b
    w_a, w_b = uvl.w_a, uvl.w_b

    if x <= a:
        return 0.0                  

    t = min(x, b)           
    d = b - a
    s = t - a                    

    slope = (w_b - w_a) / d
    lever = x - a                  

    M = (w_a * lever * s
         - w_a * s**2 / 2.0
         + slope * lever * s**2 / 2.0
         - slope * s**3 / 3.0)
    return M


def shear_force(config: BeamConfig, x: float) -> float:

    R_A, _ = calculate_reactions(config)

    V = R_A

    for pl in config.point_loads:
        if pl.position < x:
            V -= pl.magnitude

    all_uvls = config.get_all_uvls() if hasattr(config, "get_all_uvls") else ([config.uvl] if config.uvl else [])
    for uvl in all_uvls:
        V -= _uvl_shear_contribution(uvl, x)

    return V


def bending_moment(config: BeamConfig, x: float) -> float:

    R_A, _ = calculate_reactions(config)

    M = R_A * x


    for pl in config.point_loads:
        if pl.position <= x:
            M -= pl.magnitude * (x - pl.position)


    all_uvls = config.get_all_uvls() if hasattr(config, "get_all_uvls") else ([config.uvl] if config.uvl else [])
    for uvl in all_uvls:
        M -= _uvl_moment_contribution(uvl, x)

    moments = getattr(config, "point_moments", [])
    for pm in moments:
        if pm.position <= x:
            M += pm.magnitude

    return M



def find_points_of_contraflexure(
    config: BeamConfig,
    tol: float = 1e-9,
) -> List[ContraflexurePoint]:

    L = config.length
    if L <= 0:
        return []


    critical_xs: List[float] = [0.0, L]
    for pl in config.point_loads:
        if 0.0 <= pl.position <= L:
            critical_xs.append(pl.position)

    all_uvls = config.get_all_uvls() if hasattr(config, "get_all_uvls") else ([config.uvl] if config.uvl else [])
    for uvl in all_uvls:
        if 0.0 <= uvl.a <= L:
            critical_xs.append(uvl.a)
        if 0.0 <= uvl.b <= L:
            critical_xs.append(uvl.b)

    moments = getattr(config, "point_moments", [])
    for pm in moments:
        if 0.0 <= pm.position <= L:
            critical_xs.append(pm.position)

    sorted_crit = sorted(set(critical_xs))

    raw_candidates: List[Tuple[float, str]] = []

    for i in range(len(sorted_crit) - 1):
        x_start = sorted_crit[i]
        x_end = sorted_crit[i + 1]
        if x_end - x_start < 1e-10:
            continue


        num_sub = max(20, int(math.ceil((x_end - x_start) / L * 100)))
        sub_grid = np.linspace(x_start, x_end, num_sub + 1)

        for k in range(num_sub):
            a_sub = float(sub_grid[k])
            b_sub = float(sub_grid[k + 1])
            M_a = bending_moment(config, a_sub)
            M_b = bending_moment(config, b_sub)

            if M_a * M_b <= 0.0:

                lo, hi = a_sub, b_sub
                M_lo = M_a
                for _ in range(70):
                    mid = 0.5 * (lo + hi)
                    M_mid = bending_moment(config, mid)
                    if abs(M_mid) < 1e-12 or (hi - lo) < tol:
                        lo = mid
                        hi = mid
                        break
                    if M_mid * M_lo <= 0.0:
                        hi = mid
                    else:
                        lo = mid
                        M_lo = M_mid
                root = 0.5 * (lo + hi)
                if 1e-7 < root < (L - 1e-7):
                    raw_candidates.append((root, "continuous_crossing"))

    _EPS = 1e-8
    for pm in moments:
        pos = pm.position
        if 1e-7 < pos < (L - 1e-7):
            M_left = bending_moment(config, pos - _EPS)
            M_right = bending_moment(config, pos + _EPS)
            if M_left * M_right < 0.0:
                raw_candidates.append((pos, "moment_jump"))


    confirmed_points: List[ContraflexurePoint] = []

    raw_candidates.sort(key=lambda item: item[0])

    for cand_x, c_type in raw_candidates:
        is_near_pm = False
        for pm in moments:
            if abs(cand_x - pm.position) < 1e-5:
                cand_x = pm.position
                c_type = "moment_jump"
                is_near_pm = True
                break


        if not (1e-6 < cand_x < (L - 1e-6)):
            continue

        if any(abs(cand_x - cp.position) < 1e-5 for cp in confirmed_points):
            continue


        delta = min(1e-4, cand_x * 0.1, (L - cand_x) * 0.1)
        if c_type == "moment_jump":
            M_l = bending_moment(config, cand_x - 1e-7)
            M_r = bending_moment(config, cand_x + 1e-7)
            has_sign_change = (M_l * M_r < 0.0)
        else:
            M_l = bending_moment(config, cand_x - delta)
            M_r = bending_moment(config, cand_x + delta)
            if M_l * M_r >= 0.0:
                delta_small = 1e-6
                M_l = bending_moment(config, cand_x - delta_small)
                M_r = bending_moment(config, cand_x + delta_small)
            has_sign_change = (M_l * M_r < 0.0)

        if has_sign_change:
            confirmed_points.append(
                ContraflexurePoint(
                    position=cand_x,
                    moment=bending_moment(config, cand_x) if c_type != "moment_jump" else 0.0,
                    type=c_type,
                )
            )

    return confirmed_points


def sample_beam() -> BeamConfig:
    return BeamConfig(
        length=10.0,
        point_loads=[
            PointLoad(magnitude=20.0, position=2.0),
            PointLoad(magnitude=35.0, position=5.0),
            PointLoad(magnitude=15.0, position=8.0),
        ],
        uvl=UVL(a=3.0, b=7.0, w_a=0.0, w_b=30.0),
    )
