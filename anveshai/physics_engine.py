"""
Physics Engine — deterministic formula solver for common physics problems.

Supported domains:
  · Kinematics  : SUVAT equations (v=u+at, s=ut+½at², v²=u²+2as)
  · Dynamics    : F=ma, weight, friction, Newton's laws
  · Energy      : KE=½mv², PE=mgh, W=Fd, conservation of energy
  · Power       : P=W/t, P=Fv
  · Momentum    : p=mv, impulse=FΔt, conservation
  · Circular    : F=mv²/r, a_c=v²/r, ω=v/r, T=2πr/v
  · Gravitation : F=Gm₁m₂/r², g at altitude
  · Waves       : v=fλ, T=1/f, E=hf (photon energy)
  · Optics      : Snell's law n₁sinθ₁=n₂sinθ₂, lens 1/f=1/v+1/u
  · Electricity : V=IR, P=VI=I²R=V²/R, series/parallel R
  · Thermodynamics: Q=mcΔT, PV=nRT, efficiency η=W/Q_h
  · Fluid/Pressure: P=F/A, P=ρgh, ρ=m/V
  · Relativity  : time dilation, length contraction (conceptual)

All quantities extracted from natural language via regex.
Returns (success, result_string, formula_used).
"""

from __future__ import annotations

import re
import math
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Physical constants
# ─────────────────────────────────────────────────────────────────────────────

G    = 6.674e-11   # Gravitational constant (N·m²/kg²)
g    = 9.81        # Standard gravity (m/s²)
h    = 6.626e-34   # Planck constant (J·s)
c    = 3e8         # Speed of light (m/s)
R    = 8.314       # Ideal gas constant (J/mol·K)
e    = 1.602e-19   # Elementary charge (C)
k_B  = 1.381e-23   # Boltzmann constant (J/K)
k_e  = 8.987e9     # Coulomb constant (N·m²/C²)
m_e  = 9.109e-31   # Electron mass (kg)
m_p  = 1.673e-27   # Proton mass (kg)


# ─────────────────────────────────────────────────────────────────────────────
# Unit conversion helpers
# ─────────────────────────────────────────────────────────────────────────────

def _km_h_to_m_s(v: float) -> float:
    return v / 3.6

def _m_s_to_km_h(v: float) -> float:
    return v * 3.6

def _deg_to_rad(deg: float) -> float:
    return math.radians(deg)


# ─────────────────────────────────────────────────────────────────────────────
# Number extraction utilities
# ─────────────────────────────────────────────────────────────────────────────

def _num(text: str, *patterns: str) -> Optional[float]:
    """
    Try each pattern in order; return the first matched float.
    Patterns should contain one capture group for the number.
    """
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            try:
                return float(m.group(1).replace(',', ''))
            except ValueError:
                continue
    return None


def _extract(text: str) -> dict[str, Optional[float]]:
    """
    Extract all recognisable physical quantities from a natural-language string.
    Returns a dict of quantity_name -> value (or None if not found).
    """
    t = text.lower()
    vals: dict[str, Optional[float]] = {}

    # ── Kinematics ────────────────────────────────────────────────────────────
    vals['initial_velocity'] = _num(t,
        r'initial\s+(?:velocity|speed)\s+(?:of\s+|=\s*|:?\s*)([\d.]+)',
        r'(?:starts?|begins?|initial)\s+(?:at|with)?\s*([\d.]+)\s*(?:m/s|km/h|mph)',
        r'u\s*=\s*([\d.]+)',
        r'(?:thrown|launched|projected|fired)\s+(?:\w+\s+)?at\s+([\d.]+)\s*(?:m/s|km/h)',
        r'(?:decelerates?|slows?)\s+from\s+([\d.]+)\s*m/s',
    )
    if re.search(r'starts?\s+from\s+rest|initial\s+velocity\s+(?:is\s+)?zero|u\s*=\s*0'
                 r'|starting\s+from\s+rest|begins?\s+from\s+rest|from\s+rest\b', t):
        vals['initial_velocity'] = 0.0

    vals['final_velocity'] = _num(t,
        r'final\s+(?:velocity|speed)\s+(?:of\s+|=\s*|:?\s*)([\d.]+)',
        r'reaches?\s+(?:a\s+speed\s+of\s+|velocity\s+of\s+)?([\d.]+)\s*(?:m/s|km/h)',
        r'v\s*=\s*([\d.]+)',
        r'(?:decelerates?|slows?)\s+from\s+[\d.]+\s*m/s\s+to\s+([\d.]+)',
    )
    # "to rest" / "to 0 m/s" / "at maximum/max height" → final_velocity = 0
    if re.search(r'\bto\s+(?:a\s+)?(?:complete\s+)?(?:stop|rest)\b|to\s+0\s*(?:m/s)?\b'
                 r'|\bmax(?:imum)?\s+height\b|\bat\s+the\s+top\b', t):
        vals['final_velocity'] = 0.0

    vals['speed'] = _num(t,
        r'(?:^|[^a-z])speed\s+(?:of\s+|=\s*|:?\s*)([\d.]+)',
        r'(?:traveling|moving|travelling|goes?|drives?)\s+at\s+([\d.]+)\s*(?:m/s|km/h|mph|kph)?',
        r'velocity\s+(?:of\s+|=\s*)([\d.]+)',
        r'([\d.]+)\s*(?:km/h|kph|mph|m/s|ms-1)\b',
    )

    vals['acceleration'] = _num(t,
        r'acceleration\s+(?:of\s+|=\s*|:?\s*)([\d.]+)',
        r'accelerates?\s+at\s+([\d.]+)',
        r'a\s*=\s*([\d.]+)',
        r'([\d.]+)\s*m/s[²2]',
    )

    vals['time'] = _num(t,
        r'(?:for|in|after|over|takes?|duration\s+of)\s+([\d.]+)\s*(?:s|sec(?:ond)?s?|h(?:ours?)?|min(?:utes?)?)\b',
        r'time\s+(?:of\s+|=\s*|:?\s*)([\d.]+)',
        r't\s*=\s*([\d.]+)',
        r'([\d.]+)\s*(?:seconds?|minutes?)\b',
    )

    vals['displacement'] = _num(t,
        r'(?:displacement|distance)\s+(?:of\s+|=\s*|:?\s*)([\d.]+)',
        r'travels?\s+([\d.]+)\s*(?:m|km|miles?|feet|ft)\b',
        r'covers?\s+([\d.]+)\s*(?:m|km)',
        r's\s*=\s*([\d.]+)',
        r'([\d.]+)\s*(?:\bkm\b|\bm\b)\s*(?:away|far|distant)',
        r'(?:over|through|across|along)\s+([\d.]+)\s*m\b',
        r'(?:over|through|across|along)\s+([\d.]+)\s*km\b',
        r'd\s*=\s*([\d.]+)',
    )

    # Unit correction — convert km to m for displacement if needed
    if vals['displacement'] is not None:
        m_km = re.search(r'(?:travels?|covers?|distance\s+of)\s+([\d.]+)\s*km\b', t)
        if m_km:
            vals['displacement'] = float(m_km.group(1)) * 1000

    # ── Mass / weight ──────────────────────────────────────────────────────────
    vals['mass'] = _num(t,
        r'mass\s+(?:of\s+|=\s*|:?\s*)([\d.]+)\s*(?:kg|g|lb|pounds?)?',
        r'([\d.]+)\s*kg\b',
        r'm\s*=\s*([\d.]+)',
        r'(?:weighs?|weight)\s+(?:of\s+|=\s*)?([\d.]+)\s*kg',
    )
    # Convert g → kg
    m_grams = re.search(r'mass\s+(?:of\s+|=\s*)?([\d.]+)\s*g\b', t)
    if m_grams and vals['mass'] is None:
        vals['mass'] = float(m_grams.group(1)) / 1000

    # ── Force ─────────────────────────────────────────────────────────────────
    vals['force'] = _num(t,
        r'force\s+(?:of\s+|=\s*|:?\s*)([\d.]+)',
        r'F\s*=\s*([\d.]+)',
        r'([\d.]+)\s*(?:N|newtons?)\b',
    )

    # ── Energy ────────────────────────────────────────────────────────────────
    vals['height'] = _num(t,
        r'height\s+(?:of\s+|=\s*|:?\s*)([\d.]+)',
        r'h\s*=\s*([\d.]+)',
        r'([\d.]+)\s*m\s+(?:high|above|tall)',
        r'(?:drops?|falls?|raised?)\s+(?:from\s+|by\s+)?([\d.]+)\s*m',
        r'at\s+([\d.]+)\s*m\b',
    )

    vals['kinetic_energy'] = _num(t,
        r'kinetic\s+energy\s+(?:of\s+|=\s*|:?\s*)([\d.]+)',
        r'KE\s*=\s*([\d.]+)',
        r'k\.?e\.?\s*=\s*([\d.]+)',
    )

    vals['potential_energy'] = _num(t,
        r'(?:potential|gravitational\s+potential)\s+energy\s+(?:of\s+|=\s*)?([\d.]+)',
        r'PE\s*=\s*([\d.]+)',
        r'p\.?e\.?\s*=\s*([\d.]+)',
    )

    vals['work'] = _num(t,
        r'work\s+(?:done|of)\s+(?:of\s+|=\s*|:?\s*)?([\d.]+)',
        r'W\s*=\s*([\d.]+)',
        r'([\d.]+)\s*(?:J|joules?)\b',
    )

    vals['power'] = _num(t,
        r'power\s+(?:of\s+|=\s*|:?\s*)([\d.]+)',
        r'P\s*=\s*([\d.]+)',
        r'([\d.]+)\s*(?:W|watts?|kW)\b',
    )

    # ── Waves ─────────────────────────────────────────────────────────────────
    vals['frequency'] = _num(t,
        r'frequency\s+(?:of\s+|=\s*|:?\s*)([\d.e\-]+)',
        r'f\s*=\s*([\d.e\-]+)',
        r'([\d.]+)\s*(?:Hz|hertz|kHz|MHz|GHz)\b',
    )
    # Unit conversion for frequency
    hz_m = re.search(r'([\d.]+)\s*kHz\b', t)
    if hz_m and vals['frequency'] is None:
        vals['frequency'] = float(hz_m.group(1)) * 1e3
    hz_m = re.search(r'([\d.]+)\s*MHz\b', t)
    if hz_m and vals['frequency'] is None:
        vals['frequency'] = float(hz_m.group(1)) * 1e6

    vals['wavelength'] = _num(t,
        r'wavelength\s+(?:of\s+|=\s*|:?\s*)([\d.e\-]+)',
        r'λ\s*=\s*([\d.e\-]+)',
        r'([\d.e\-]+)\s*(?:nm|µm|μm|mm|cm)\s+(?:wave|light)',
    )
    # Unit conversions for wavelength (always apply to convert to metres)
    nm_m  = re.search(r'([\d.]+)\s*nm\b', t)
    um_m  = re.search(r'([\d.]+)\s*[µμ]m\b', t)
    mm_m  = re.search(r'([\d.]+)\s*mm\b', t)
    if nm_m:
        vals['wavelength'] = float(nm_m.group(1)) * 1e-9
    elif um_m:
        vals['wavelength'] = float(um_m.group(1)) * 1e-6
    elif mm_m:
        vals['wavelength'] = float(mm_m.group(1)) * 1e-3

    vals['period'] = _num(t,
        r'period\s+(?:of\s+|=\s*|:?\s*)([\d.]+)',
        r'T\s*=\s*([\d.]+)',
    )

    vals['wave_speed'] = _num(t,
        r'wave\s+speed\s+(?:of\s+|=\s*|:?\s*)([\d.]+)',
        r'speed\s+(?:of\s+(?:the\s+)?(?:wave|sound|light)\s+)?(?:of\s+|=\s*|:?\s*)([\d.]+)\s*m/s',
        r'speed\s+(?:is\s+)?([\d.]+)\s*m/s',
    )

    # ── Friction ──────────────────────────────────────────────────────────────
    mu_m = re.search(r'(?:μ|mu)\s*=\s*([\d.]+)', t)
    coeff_m = re.search(
        r'coefficient\s+(?:of\s+(?:friction|kinetic|static)\s+)?(?:μ\s*=\s*|mu\s*=\s*|=\s*)?([\d.]+)', t
    )
    if mu_m:
        vals['friction_coefficient'] = float(mu_m.group(1))
    elif coeff_m:
        vals['friction_coefficient'] = float(coeff_m.group(1))
    else:
        vals['friction_coefficient'] = None

    # ── Electricity ───────────────────────────────────────────────────────────
    vals['voltage'] = _num(t,
        r'voltage\s+(?:of\s+|=\s*|:?\s*)([\d.]+)',
        r'potential\s+difference\s+(?:of\s+|=\s*)?([\d.]+)',
        r'V\s*=\s*([\d.]+)',
        r'([\d.]+)\s*(?:V|volts?)\b',
    )

    vals['current'] = _num(t,
        r'current\s+(?:of\s+|=\s*|:?\s*)([\d.]+)',
        r'I\s*=\s*([\d.]+)',
        r'([\d.]+)\s*(?:A|amps?|amperes?)\b',
    )

    vals['resistance'] = _num(t,
        r'resistance\s+(?:of\s+|=\s*|:?\s*)([\d.]+)',
        r'R\s*=\s*([\d.]+)',
        r'([\d.]+)\s*(?:Ω|ohm|ohms?)\b',
    )

    # ── Thermodynamics ────────────────────────────────────────────────────────
    vals['temperature_change'] = _num(t,
        r'temperature\s+(?:change|rise|drop|difference|increases?\s+by|decreases?\s+by)\s+(?:of\s+|=\s*)?([\d.]+)',
        r'ΔT\s*=\s*([\d.]+)',
        r'delta\s*T\s*=\s*([\d.]+)',
        r'changes?\s+(?:by|from\s+[\d.]+\s+to\s+[\d.]+)\s*([\d.]+)',
    )
    # ΔT from "from X to Y"
    if vals['temperature_change'] is None:
        m_range = re.search(r'from\s+([\d.]+)\s*[°℃C]?\s+to\s+([\d.]+)\s*[°℃C]?', t)
        if m_range:
            vals['temperature_change'] = abs(float(m_range.group(2)) - float(m_range.group(1)))

    vals['specific_heat'] = _num(t,
        r'specific\s+heat\s+(?:capacity\s+)?(?:of\s+|=\s*|:?\s*)?([\d.]+)',
        r'c\s*=\s*([\d.]+)',
        r'([\d.]+)\s*J/\(?kg[·.·]?K?\)?',
    )

    vals['heat'] = _num(t,
        r'heat\s+(?:energy\s+)?(?:of\s+|=\s*|:?\s*)([\d.]+)',
        r'Q\s*=\s*([\d.]+)',
        r'([\d.]+)\s*(?:kJ|kcal)',  # will convert below
    )

    # ── Pressure / Fluid ──────────────────────────────────────────────────────
    vals['pressure'] = _num(t,
        r'pressure\s+(?:of\s+|=\s*|:?\s*)([\d.]+)',
        r'P\s*=\s*([\d.]+)',
        r'([\d.]+)\s*(?:Pa|pascal|atm|bar|kPa)\b',
    )

    vals['area'] = _num(t,
        r'(?:cross[\-\s]?sectional\s+)?area\s+(?:of\s+|=\s*|:?\s*)([\d.]+)',
        r'A\s*=\s*([\d.]+)',
        r'([\d.]+)\s*m[²2]',
    )

    vals['density'] = _num(t,
        r'density\s+(?:of\s+|=\s*|:?\s*)([\d.]+)',
        r'ρ\s*=\s*([\d.]+)',
        r'([\d.]+)\s*kg/m[³3]',
    )

    vals['volume'] = _num(t,
        r'volume\s+(?:of\s+|=\s*|:?\s*)([\d.]+)',
        r'V\s*=\s*([\d.]+)',
        r'([\d.]+)\s*(?:m[³3]|L|liters?|litres?)',
    )

    # ── Optics ────────────────────────────────────────────────────────────────
    vals['angle'] = _num(t,
        r'angle\s+(?:of\s+|=\s*|:?\s*)([\d.]+)',
        r'θ\s*=\s*([\d.]+)',
        r'([\d.]+)\s*°',
    )

    vals['refractive_index_1'] = _num(t,
        r'n[_₁1]\s*=\s*([\d.]+)',
        r'refractive\s+index\s+(?:of\s+)?(?:medium\s+)?1\s*[=:]\s*([\d.]+)',
    )

    vals['refractive_index_2'] = _num(t,
        r'n[_₂2]\s*=\s*([\d.]+)',
        r'refractive\s+index\s+(?:of\s+)?(?:medium\s+)?2\s*[=:]\s*([\d.]+)',
    )

    vals['radius'] = _num(t,
        r'radius\s+(?:of\s+|=\s*|:?\s*)([\d.]+)',
        r'r\s*=\s*([\d.]+)',
    )

    vals['focal_length'] = _num(t,
        r'focal\s+length\s+(?:of\s+|=\s*|:?\s*)([\d.]+)',
        r'f\s*=\s*([\d.]+)\s*(?:cm|m)',
    )

    vals['object_distance'] = _num(t,
        r'object\s+distance\s+(?:of\s+|=\s*|:?\s*)?([\d.]+)',
        r'u\s*=\s*([\d.]+)',
    )

    vals['image_distance'] = _num(t,
        r'image\s+distance\s+(?:of\s+|=\s*|:?\s*)?([\d.]+)',
        r'(?:v|image)\s*=\s*([\d.]+)',
    )

    # ── Spring ────────────────────────────────────────────────────────────────
    vals['spring_constant'] = _num(t,
        r'spring\s+constant\s+(?:k\s*=\s*|of\s*|=\s*)?([\d.]+)',
        r'\bk\s*=\s*([\d.]+)\s*N/m',
        r'([\d.]+)\s*N/m\b',
    )
    vals['spring_compression'] = _num(t,
        r'compressed?\s+(?:by\s+)?([\d.]+)\s*m',
        r'extended?\s+(?:by\s+)?([\d.]+)\s*m',
        r'stretch(?:ed)?\s+(?:by\s+)?([\d.]+)\s*m',
        r'\bx\s*=\s*([\d.]+)',
    )

    # ── Electrostatics ────────────────────────────────────────────────────────
    # Extract charges — look for q= or µC / microC / C values
    # preprocess converts µC → microC, so match both
    UC_PAT = r'[µμ]C\b|micro\s*C\b'
    q_vals = re.findall(
        r'(?:q[₁₂12]?\s*=\s*)?([\d.e\-]+)\s*(?:[µμ]C\b|micro\s*C\b)', t, re.I
    )
    is_microcoulomb = bool(re.search(UC_PAT, t, re.I))
    if not q_vals:
        q_vals = re.findall(r'(?:q[₁₂12]?\s*=\s*)?([\d.e\-]+)\s*C\b', t, re.I)
        is_microcoulomb = False
    if q_vals:
        scale = 1e-6 if is_microcoulomb else 1.0
        vals['charge1'] = float(q_vals[0]) * scale
        vals['charge2'] = float(q_vals[1]) * scale if len(q_vals) > 1 else None
    else:
        vals['charge1'] = None
        vals['charge2'] = None
    # Single charge for magnetic force
    q_m = re.search(r'(?:charge\s+)?q\s*=\s*([\d.e\-]+)\s*([µμm]?C|micro\s*C)', t, re.I)
    if q_m:
        val_q = float(q_m.group(1))
        unit  = q_m.group(2).lower().replace(' ', '')
        if unit in ('µc', 'μc', 'mc', 'microc'):
            val_q *= 1e-6
        vals['charge'] = val_q

    vals['magnetic_field'] = _num(t,
        r'(?:magnetic\s+field|B)\s*=\s*([\d.e\-]+)',
        r'([\d.e\-]+)\s*T\b',
    )

    vals['separation'] = _num(t,
        r'([\d.]+)\s*m\s+apart',
        r'apart\s+(?:by\s+)?([\d.]+)\s*m',
        r'separation\s+(?:of\s+|=\s*)?([\d.]+)',
        r'placed\s+([\d.]+)\s*m\b',
        r'distance\s+(?:between|of)\s+([\d.]+)\s*m',
    )

    # ── Launch angle ──────────────────────────────────────────────────────────
    ang_m = re.search(
        r'(?:at|angle\s+of|angle)\s+([\d.]+)\s*degree|'
        r'([\d.]+)\s*°\s*(?:to\s+(?:the\s+)?horizontal|above|below)',
        t, re.I
    )
    if ang_m:
        vals['launch_angle'] = float(ang_m.group(1) or ang_m.group(2))

    # ── Lines per mm (diffraction) ─────────────────────────────────────────────
    lpm_m = re.search(r'([\d.]+)\s*lines?/mm', t, re.I)
    if lpm_m:
        vals['lines_per_mm'] = float(lpm_m.group(1))

    # ── Diffraction order ─────────────────────────────────────────────────────
    ord_m = re.search(r'(\d+)(?:st|nd|rd|th)\s+order', t, re.I)
    if ord_m:
        vals['diffraction_order'] = int(ord_m.group(1))

    # Speed in km/h → convert to m/s if needed (common for kinematics)
    kmh_m = re.search(r'([\d.]+)\s*km/h\b', t)
    if kmh_m:
        v_kmh = float(kmh_m.group(1))
        # If no speed/velocity set yet in m/s, use km/h converted
        if vals['speed'] is None and vals['initial_velocity'] is None:
            vals['speed'] = _km_h_to_m_s(v_kmh)
        vals['_speed_kmh'] = v_kmh   # preserve original for display

    return vals


# ─────────────────────────────────────────────────────────────────────────────
# Question-type detector
# ─────────────────────────────────────────────────────────────────────────────

def _detect_question_type(text: str) -> str:
    """Return the type of physics problem to solve."""
    I = re.IGNORECASE

    # Ordered from most specific to most general
    if re.search(r'\bsnell|\brefract|\brefractive\s+index\b|n₁|n1\s*sin|critical\s+angle', text, I):
        return 'snell'
    if re.search(r'\blens\b|\bmirror\b|focal\s+length|1/v\s*\+\s*1/u|object\s+distance|image\s+distance', text, I):
        return 'lens'
    if re.search(r'\bohm\b|resistance|resistor|V\s*=\s*IR|series\s+circuit|parallel\s+circuit', text, I):
        return 'ohm'
    if re.search(r'\bvoltage\b|\bcurrent\b', text, I) and re.search(r'\bpower\b|\bwatt', text, I):
        return 'electric_power'
    if re.search(r'\bphoton\b|\bE\s*=\s*hf|\bPlanck\b', text, I):
        return 'photon'
    # ── Specific types that must come before generic energy/kinematics ─────────
    if re.search(r'\bescape\s+velocity\b|\bescape\s+speed\b', text, I):
        return 'escape_velocity'
    if re.search(r'\bde\s+broglie\b|\bmatter\s+wave\b|\bwavelength\s+of\s+(?:an?\s+)?electron\b'
                 r'|\bwavelength\s+of\s+(?:a\s+)?(?:proton|neutron|particle)\b', text, I):
        return 'de_broglie'
    if re.search(r'\bmoment\s+of\s+inertia\b|\brotational\s+inertia\b', text, I):
        return 'moment_of_inertia'
    if re.search(r'\bcoulomb\b|\belectric\s+force\b|\bforce\s+between\s+(?:two\s+)?charges\b'
                 r'|\btwo\s+charges?\b|\bcharges?\b.*\bplaced\b|\bF\s*=\s*kq'
                 r'|\b[µμ]C\b|\bmicro\s*C\b', text, I):
        return 'coulomb'
    if re.search(r'\bmagnetic\s+force\b|\bLorentz\s+force\b|\bF\s*=\s*qvB\b'
                 r'|\bcharge.*moving.*(?:magnetic|field)\b|\bparticle.*magnetic\s+field\b', text, I):
        return 'magnetic_force'
    if re.search(r'\bdiffraction\s+grating\b|\blines\s+per\s+mm\b|\bgrating\s+spacing\b', text, I):
        return 'diffraction'
    if re.search(r'\belastic\s+potential\b|\bspring\s+PE\b|\bspring\b.*\benergy\b'
                 r'|\bcompressed?\s+spring\b|\benergy\s+(?:stored\s+)?in\s+(?:a\s+)?spring\b', text, I):
        return 'spring_energy'
    if re.search(r'\bgravitational\s+potential\s+energy\s+between\b'
                 r'|\battraction\s+between\s+two\s+masses\b', text, I):
        return 'gravitational_pe'
    if re.search(r'\bprojectile\b|\bthrown\b.*\d+\s*degree'
                 r'|\bfired\s+at\s+(?:\d+\s*degrees?|an?\s+angle)\b'
                 r'|\blaunched\s+at\s+(?:\d+\s*degrees?|an?\s+angle)\b'
                 r'|\brange\s+of\s+(?:a\s+)?(?:projectile|ball|object)\b', text, I):
        return 'projectile'
    # ── Generic energy / dynamics ──────────────────────────────────────────────
    if re.search(r'\bkinetic\s+energy\b|\bKE\b|\b½mv²|half\s+mv\s+squared', text, I):
        return 'kinetic_energy'
    if re.search(r'\bpotential\s+energy\b|\bGPE\b|\bPE\b|\bmgh\b|\bgravitational\s+(?:potential\s+)?energy', text, I):
        return 'potential_energy'
    if re.search(r'\bwork\s+done\b|\bwork\s*=|\bW\s*=\s*Fd\b|\benergy\s+transferred'
                 r'|\bforce\s+(?:of\s+[\d.]+\s*N\s+)?applied\s+over\s+a?\s*distance', text, I):
        return 'work'
    if re.search(r'\bpower\b|\bwatts?\b', text, I) and re.search(r'\bwork\b|\btime\b|\bforce\b|\bvelocity\b', text, I):
        return 'power'
    if re.search(r'\bmomentum\b|\bimpulse\b|\bp\s*=\s*mv', text, I):
        return 'momentum'
    if re.search(r'\bcentripetal\b|\bcircular\s+motion\b|\borbiting\b|\borbital\b', text, I):
        return 'circular'
    if re.search(r'\bgravitation|\borbital\s+speed|\bNewton.s\s+law\s+of\s+gravit|\bGm', text, I):
        return 'gravitation'
    if re.search(r'\bwavelength\b|\bfrequency\b|\bwave\s+speed\b|\bperiod\b|\bv\s*=\s*f', text, I):
        return 'waves'
    if re.search(r'\bheat\b|\bspecific\s+heat|\bthermal\s+energy\b|Q\s*=\s*mc|[ΔδΔ]T|delta\s*T', text, I):
        return 'heat'
    if re.search(r'\bpressure\b|\bforce\s+per\s+unit\s+area\b|P\s*=\s*F/A', text, I):
        return 'pressure'
    if re.search(r'\bdensity\b|\bρ\b|\brho\b', text, I):
        return 'density'
    if re.search(r'\bfluid\s+pressure\b|\bhydrostatic\b|\bdepth\b|ρgh', text, I):
        return 'fluid_pressure'
    if re.search(r'F\s*=\s*ma|\bnewton.s\s+second\b|\bnet\s+force\b', text, I):
        return 'force'
    if re.search(r'\bacceleration\b', text, I) and re.search(r'\bforce\b|\bmass\b', text, I):
        return 'force'
    if re.search(r'\bweight\b', text, I) and re.search(r'\bmass\b|g\s*=|\bgravit', text, I):
        return 'weight'
    if re.search(r'\bfriction\b|\bcoefficient\b|\bmu\s*=|\bμ\s*=', text, I):
        return 'friction'
    if re.search(r'\bkinematic|\bsuvat\b|\bdisplacement\b|\baccelerat|\bvelocity\b|\bspeed\b'
                 r'|\bthrown\b|\bdecelerat|\bmax(?:imum)?\s+height\b|\bfrom\s+rest\b', text, I):
        return 'kinematics'

    return 'unknown'


# ─────────────────────────────────────────────────────────────────────────────
# Formula solvers
# ─────────────────────────────────────────────────────────────────────────────

def _solve_kinematics(v: dict, text: str) -> tuple[bool, str]:
    t = text.lower()
    u  = v.get('initial_velocity')
    vf = v.get('final_velocity')
    a  = v.get('acceleration')
    s  = v.get('displacement')
    t_ = v.get('time')

    # Infer a = -g for vertical throw / projectile with no explicit acceleration
    if a is None and re.search(r'\bthrown\b|\bprojectile\b|\bfired\s+(?:up|vert)|\blaunched\s+(?:up|vert)'
                               r'|\bdropped\b|\bfalls?\b|\bfalling\b', t):
        a = -g   # upward positive convention

    # Determine what is being asked
    asking_velocity     = bool(re.search(r'find\s+(?:the\s+)?(?:final\s+)?velocity|what\s+is\s+(?:the\s+)?(?:final\s+)?velocity|speed\s+after', t))
    asking_distance     = bool(re.search(r'how\s+far|find\s+(?:the\s+)?distance|distance\s+travel|displacement', t))
    asking_time         = bool(re.search(r'how\s+long|time\s+taken|when\s+does|find\s+(?:the\s+)?time', t))
    asking_acceleration = bool(re.search(r'find\s+(?:the\s+)?acceleration|what\s+is\s+(?:the\s+)?acceleration', t))

    # Use speed as initial_velocity if not set
    if u is None:
        u = v.get('speed')

    results = []

    # v = u + at
    if u is not None and a is not None and t_ is not None:
        vf_calc = u + a * t_
        results.append(f"v = u + at = {u} + {a}×{t_} = {vf_calc:.4g} m/s")

    # s = ut + ½at²
    if u is not None and a is not None and t_ is not None:
        s_calc = u * t_ + 0.5 * a * t_**2
        results.append(f"s = ut + ½at² = {u}×{t_} + ½×{a}×{t_}² = {s_calc:.4g} m")

    # v² = u² + 2as
    if u is not None and a is not None and s is not None:
        v2 = u**2 + 2 * a * s
        if v2 >= 0:
            vf_calc = math.sqrt(v2)
            results.append(f"v² = u² + 2as  →  v = √({u}² + 2×{a}×{s}) = {vf_calc:.4g} m/s")

    # Solve for time: s = ut + ½at² → quadratic in t
    if s is not None and u is not None and a is not None and t_ is None:
        # at²/2 + ut - s = 0
        A_, B_, C_ = 0.5 * a, u, -s
        disc = B_**2 - 4 * A_ * C_
        if disc >= 0 and A_ != 0:
            t1 = (-B_ + math.sqrt(disc)) / (2 * A_)
            t2 = (-B_ - math.sqrt(disc)) / (2 * A_)
            t_pos = [t for t in [t1, t2] if t >= 0]
            if t_pos:
                results.append(f"Time: t = {min(t_pos):.4g} s")
        elif A_ == 0 and B_ != 0:
            results.append(f"Time: t = s/u = {s}/{u} = {s/u:.4g} s")

    # Simple d = vt (constant speed, no acceleration)
    if s is None and (u is not None or v.get('speed') is not None) and t_ is not None and a is None:
        spd = u if u is not None else v.get('speed')
        s_calc = spd * t_
        results.append(f"s = v×t = {spd}×{t_} = {s_calc:.4g} m")

    # d = vt → t
    if s is not None and (u is not None or v.get('speed') is not None) and t_ is None and a is None:
        spd = u if u is not None else v.get('speed')
        t_calc = s / spd
        results.append(f"t = s/v = {s}/{spd} = {t_calc:.4g} s")

    # a = (v - u) / t
    if u is not None and vf is not None and t_ is not None and a is None:
        a_calc = (vf - u) / t_
        results.append(f"a = (v - u)/t = ({vf} - {u})/{t_} = {a_calc:.4g} m/s²")

    # t = (v - u) / a  (u, vf, a known — e.g. thrown upward, max height)
    if u is not None and vf is not None and a is not None and t_ is None:
        if a != 0:
            t_calc = (vf - u) / a
            if t_calc >= 0:
                results.append(f"t = (v - u)/a = ({vf} - {u})/{a:.4g} = {t_calc:.4g} s")
            # s = (v² - u²) / (2a)
            s_calc = (vf**2 - u**2) / (2 * a)
            results.append(f"s = (v² - u²)/(2a) = ({vf}² - {u}²)/(2×{a:.4g}) = {s_calc:.4g} m")

    if results:
        return True, "\n".join(results)
    return False, "Not enough information to solve kinematics problem. Provide at least 3 of: u, v, a, s, t."


def _solve_force(v: dict, text: str) -> tuple[bool, str]:
    m = v.get('mass')
    a = v.get('acceleration')
    F = v.get('force')

    if m is not None and a is not None:
        F_calc = m * a
        return True, f"F = ma = {m} × {a} = {F_calc:.4g} N"
    if m is not None and F is not None:
        a_calc = F / m
        return True, f"a = F/m = {F}/{m} = {a_calc:.4g} m/s²"
    if F is not None and a is not None:
        m_calc = F / a
        return True, f"m = F/a = {F}/{a} = {m_calc:.4g} kg"
    return False, "Provide any two of: force (N), mass (kg), acceleration (m/s²)."


def _solve_weight(v: dict, text: str) -> tuple[bool, str]:
    m = v.get('mass')
    if m is not None:
        W = m * g
        return True, f"W = mg = {m} × {g} = {W:.4g} N"
    return False, "Provide mass in kg to calculate weight."


def _solve_kinetic_energy(v: dict, text: str) -> tuple[bool, str]:
    m  = v.get('mass')
    vf = v.get('final_velocity') or v.get('speed') or v.get('initial_velocity')
    KE = v.get('kinetic_energy')

    if m is not None and vf is not None:
        ke = 0.5 * m * vf**2
        return True, f"KE = ½mv² = ½ × {m} × {vf}² = {ke:.4g} J"
    if KE is not None and m is not None:
        vf_calc = math.sqrt(2 * KE / m)
        return True, f"v = √(2KE/m) = √(2×{KE}/{m}) = {vf_calc:.4g} m/s"
    if KE is not None and vf is not None:
        m_calc = 2 * KE / vf**2
        return True, f"m = 2KE/v² = 2×{KE}/{vf}² = {m_calc:.4g} kg"
    return False, "Provide mass (kg) and velocity (m/s) to calculate kinetic energy."


def _solve_potential_energy(v: dict, text: str) -> tuple[bool, str]:
    m  = v.get('mass')
    hv = v.get('height')
    PE = v.get('potential_energy')

    if m is not None and hv is not None:
        pe = m * g * hv
        return True, f"PE = mgh = {m} × {g} × {hv} = {pe:.4g} J"
    if PE is not None and m is not None:
        h_calc = PE / (m * g)
        return True, f"h = PE/(mg) = {PE}/({m}×{g}) = {h_calc:.4g} m"
    if PE is not None and hv is not None:
        m_calc = PE / (g * hv)
        return True, f"m = PE/(gh) = {PE}/({g}×{hv}) = {m_calc:.4g} kg"
    return False, "Provide mass (kg) and height (m) to calculate potential energy."


def _solve_work(v: dict, text: str) -> tuple[bool, str]:
    F  = v.get('force')
    d  = v.get('displacement')
    W  = v.get('work')
    # Check for angle
    theta_m = re.search(r'angle\s+(?:of\s+|=\s*)?([\d.]+)\s*°', text.lower())
    cos_theta = math.cos(math.radians(float(theta_m.group(1)))) if theta_m else 1.0

    if F is not None and d is not None:
        w = F * d * cos_theta
        angle_str = f"×cos({theta_m.group(1)}°)" if theta_m else ""
        return True, f"W = Fd{angle_str} = {F} × {d}{angle_str} = {w:.4g} J"
    if W is not None and d is not None:
        F_calc = W / (d * cos_theta)
        return True, f"F = W/d = {W}/{d} = {F_calc:.4g} N"
    if W is not None and F is not None:
        d_calc = W / (F * cos_theta)
        return True, f"d = W/F = {W}/{F} = {d_calc:.4g} m"
    return False, "Provide force (N) and distance (m) to calculate work."


def _solve_power(v: dict, text: str) -> tuple[bool, str]:
    W  = v.get('work')
    t_ = v.get('time')
    P  = v.get('power')
    F  = v.get('force')
    vf = v.get('speed') or v.get('final_velocity') or v.get('initial_velocity')

    if W is not None and t_ is not None:
        p = W / t_
        return True, f"P = W/t = {W}/{t_} = {p:.4g} W"
    if F is not None and vf is not None:
        p = F * vf
        return True, f"P = Fv = {F} × {vf} = {p:.4g} W"
    if P is not None and t_ is not None:
        W_calc = P * t_
        return True, f"W = Pt = {P} × {t_} = {W_calc:.4g} J"
    return False, "Provide work (J) and time (s), or force (N) and velocity (m/s) to calculate power."


def _solve_momentum(v: dict, text: str) -> tuple[bool, str]:
    m  = v.get('mass')
    vf = v.get('final_velocity') or v.get('speed') or v.get('initial_velocity')
    p  = v.get('potential_energy')   # reuse field, but check context
    F  = v.get('force')
    t_ = v.get('time')

    if m is not None and vf is not None:
        mom = m * vf
        return True, f"p = mv = {m} × {vf} = {mom:.4g} kg·m/s"
    if F is not None and t_ is not None:
        imp = F * t_
        return True, f"Impulse = FΔt = {F} × {t_} = {imp:.4g} N·s (= change in momentum)"
    return False, "Provide mass (kg) and velocity (m/s) to calculate momentum."


def _solve_waves(v: dict, text: str) -> tuple[bool, str]:
    f   = v.get('frequency')
    lam = v.get('wavelength')
    T   = v.get('period')
    ws  = v.get('wave_speed')

    results = []
    # wave speed = f × λ  (or solve for the missing one)
    if f is not None and lam is not None:
        wave_speed = f * lam
        results.append(f"v = fλ = {f:.4g} Hz × {lam:.4g} m = {wave_speed:.4g} m/s")
    elif ws is not None and f is not None:
        lam_calc = ws / f
        results.append(f"λ = v/f = {ws:.4g}/{f:.4g} = {lam_calc:.4g} m")
    elif ws is not None and lam is not None:
        f_calc = ws / lam
        results.append(f"f = v/λ = {ws:.4g}/{lam:.4g} = {f_calc:.4g} Hz")
    # period
    if f is not None:
        T_calc = 1 / f
        results.append(f"T = 1/f = 1/{f:.4g} = {T_calc:.4g} s")
    elif T is not None:
        f_calc = 1 / T
        results.append(f"f = 1/T = 1/{T} = {f_calc:.4g} Hz")
    if results:
        return True, "\n".join(results)
    return False, "Provide frequency (Hz) and wavelength (m), or wave speed (m/s) with either, to solve wave problems."


def _solve_photon(v: dict, text: str) -> tuple[bool, str]:
    f   = v.get('frequency')
    lam = v.get('wavelength')

    if f is not None:
        E = h * f
        return True, f"E = hf = {h:.4e} × {f:.4g} = {E:.4e} J  ({E/e:.4g} eV)"
    if lam is not None:
        E = h * c / lam
        return True, f"E = hc/λ = ({h:.4e} × {c:.4e})/{lam:.4e} = {E:.4e} J  ({E/e:.4g} eV)"
    return False, "Provide frequency (Hz) or wavelength (m) to calculate photon energy."


def _solve_ohm(v: dict, text: str) -> tuple[bool, str]:
    V_ = v.get('voltage')
    I  = v.get('current')
    R_ = v.get('resistance')

    results = []
    if V_ is not None and I is not None:
        R_calc = V_ / I
        results.append(f"R = V/I = {V_}/{I} = {R_calc:.4g} Ω")
    if V_ is not None and R_ is not None:
        I_calc = V_ / R_
        results.append(f"I = V/R = {V_}/{R_} = {I_calc:.4g} A")
    if I is not None and R_ is not None:
        V_calc = I * R_
        results.append(f"V = IR = {I} × {R_} = {V_calc:.4g} V")
    if results:
        return True, "\n".join(results)
    return False, "Provide any two of: voltage (V), current (A), resistance (Ω) to apply Ohm's law."


def _solve_electric_power(v: dict, text: str) -> tuple[bool, str]:
    V_ = v.get('voltage')
    I  = v.get('current')
    R_ = v.get('resistance')
    P_ = v.get('power')

    results = []
    if I is not None and V_ is not None:
        p = I * V_
        results.append(f"P = IV = {I} × {V_} = {p:.4g} W")
    if I is not None and R_ is not None:
        p = I**2 * R_
        results.append(f"P = I²R = {I}² × {R_} = {p:.4g} W")
    if V_ is not None and R_ is not None:
        p = V_**2 / R_
        results.append(f"P = V²/R = {V_}²/{R_} = {p:.4g} W")
    if results:
        return True, "\n".join(results)
    return False, "Provide voltage (V) and current (A), or current and resistance (Ω), to calculate power."


def _solve_heat(v: dict, text: str) -> tuple[bool, str]:
    m  = v.get('mass')
    c_ = v.get('specific_heat')
    dT = v.get('temperature_change')
    Q  = v.get('heat')

    if m is not None and c_ is not None and dT is not None:
        q = m * c_ * dT
        return True, f"Q = mcΔT = {m} × {c_} × {dT} = {q:.4g} J"
    if Q is not None and c_ is not None and dT is not None:
        m_calc = Q / (c_ * dT)
        return True, f"m = Q/(cΔT) = {Q}/({c_}×{dT}) = {m_calc:.4g} kg"
    if Q is not None and m is not None and dT is not None:
        c_calc = Q / (m * dT)
        return True, f"c = Q/(mΔT) = {Q}/({m}×{dT}) = {c_calc:.4g} J/(kg·K)"
    if Q is not None and m is not None and c_ is not None:
        dT_calc = Q / (m * c_)
        return True, f"ΔT = Q/(mc) = {Q}/({m}×{c_}) = {dT_calc:.4g} K"
    return False, "Provide mass (kg), specific heat capacity (J/kg·K), and ΔT (K) to calculate heat."


def _solve_pressure(v: dict, text: str) -> tuple[bool, str]:
    F  = v.get('force')
    A  = v.get('area')
    P_ = v.get('pressure')

    if F is not None and A is not None:
        p = F / A
        return True, f"P = F/A = {F}/{A} = {p:.4g} Pa"
    if P_ is not None and A is not None:
        F_calc = P_ * A
        return True, f"F = P×A = {P_} × {A} = {F_calc:.4g} N"
    return False, "Provide force (N) and area (m²) to calculate pressure."


def _solve_density(v: dict, text: str) -> tuple[bool, str]:
    m  = v.get('mass')
    vol = v.get('volume')
    rho = v.get('density')

    if m is not None and vol is not None:
        d = m / vol
        return True, f"ρ = m/V = {m}/{vol} = {d:.4g} kg/m³"
    if rho is not None and vol is not None:
        m_calc = rho * vol
        return True, f"m = ρV = {rho} × {vol} = {m_calc:.4g} kg"
    if rho is not None and m is not None:
        v_calc = m / rho
        return True, f"V = m/ρ = {m}/{rho} = {v_calc:.4g} m³"
    return False, "Provide mass (kg) and volume (m³) to calculate density."


def _solve_fluid_pressure(v: dict, text: str) -> tuple[bool, str]:
    rho = v.get('density')
    hv  = v.get('height')   # used as depth
    P_  = v.get('pressure')

    depth_m = re.search(r'depth\s+(?:of\s+|=\s*)?([\d.]+)\s*m', text.lower())
    depth = float(depth_m.group(1)) if depth_m else hv

    if rho is not None and depth is not None:
        p = rho * g * depth
        return True, f"P = ρgh = {rho} × {g} × {depth} = {p:.4g} Pa"
    return False, "Provide density (kg/m³) and depth (m) for fluid pressure."


def _solve_snell(v: dict, text: str) -> tuple[bool, str]:
    n1 = v.get('refractive_index_1')
    n2 = v.get('refractive_index_2')
    theta1 = v.get('angle')

    # Parse n1=X / n2=X patterns (including "n1=1" style without subscripts)
    n1_m = re.search(r'n[_₁1]\s*=\s*([\d.]+)', text, re.I)
    n2_m = re.search(r'n[_₂2]\s*=\s*([\d.]+)', text, re.I)
    if n1_m:
        n1 = float(n1_m.group(1))
    if n2_m:
        n2 = float(n2_m.group(1))

    # Fallback: plain "n = X n = Y" style
    if n1 is None or n2 is None:
        ri_vals = re.findall(r'(?<![0-9])n\s*=\s*([\d.]+)', text, re.I)
        if len(ri_vals) >= 2:
            n1, n2 = float(ri_vals[0]), float(ri_vals[1])
        elif len(ri_vals) == 1 and n2 is None:
            n2 = float(ri_vals[0])

    if n1 is None:
        n1 = 1.0   # default: air

    # Parse angle: "angle1=30", "angle=30", "30°"
    ang_m = re.search(r'angle[_₁1]?\s*=\s*([\d.]+)', text, re.I) or re.search(r'θ[_₁1]?\s*=\s*([\d.]+)', text, re.I)
    if ang_m:
        theta1 = float(ang_m.group(1))
    if theta1 is None:
        angles = re.findall(r'([\d.]+)\s*°', text)
        if angles:
            theta1 = float(angles[0])

    if n1 is not None and n2 is not None and theta1 is not None:
        sin_theta2 = n1 * math.sin(math.radians(theta1)) / n2
        if abs(sin_theta2) <= 1:
            theta2 = math.degrees(math.asin(sin_theta2))
            return True, (
                f"Snell's law: n₁sin(θ₁) = n₂sin(θ₂)\n"
                f"{n1} × sin({theta1}°) = {n2} × sin(θ₂)\n"
                f"sin(θ₂) = {sin_theta2:.4f}\n"
                f"θ₂ = {theta2:.4g}°"
            )
        else:
            return True, (
                f"Total internal reflection! sin(θ₂) = {sin_theta2:.4f} > 1"
            )
    return False, "Provide both refractive indices and angle of incidence."


def _solve_lens(v: dict, text: str) -> tuple[bool, str]:
    f_  = v.get('focal_length')
    u_  = v.get('object_distance')
    vi_ = v.get('image_distance')

    if f_ is not None and u_ is not None:
        inv_v = 1/f_ - 1/u_
        if inv_v != 0:
            vi_calc = 1 / inv_v
            m_val = vi_calc / u_
            return True, (
                f"Lens formula: 1/f = 1/v + 1/u\n"
                f"1/{f_} = 1/v + 1/{u_}\n"
                f"1/v = {1/f_:.4g} - {1/u_:.4g} = {inv_v:.4g}\n"
                f"v = {vi_calc:.4g} cm\n"
                f"Magnification m = v/u = {m_val:.4g}"
            )
    if f_ is not None and vi_ is not None:
        inv_u = 1/f_ - 1/vi_
        if inv_u != 0:
            u_calc = 1 / inv_u
            return True, (
                f"u = {u_calc:.4g} cm"
            )
    return False, "Provide focal length and object distance (or image distance) to solve lens problem."


def _solve_circular(v: dict, text: str) -> tuple[bool, str]:
    m  = v.get('mass')
    r  = v.get('radius')
    vf = v.get('speed') or v.get('final_velocity') or v.get('initial_velocity')
    F  = v.get('force')

    results = []
    if m is not None and vf is not None and r is not None:
        Fc = m * vf**2 / r
        ac = vf**2 / r
        results.append(f"Centripetal force: F = mv²/r = {m}×{vf}²/{r} = {Fc:.4g} N")
        results.append(f"Centripetal acceleration: a = v²/r = {vf}²/{r} = {ac:.4g} m/s²")
    if vf is not None and r is not None:
        omega = vf / r
        T = 2 * math.pi * r / vf
        results.append(f"Angular velocity: ω = v/r = {vf}/{r} = {omega:.4g} rad/s")
        results.append(f"Period: T = 2πr/v = {T:.4g} s")
    if results:
        return True, "\n".join(results)
    return False, "Provide mass (kg), speed (m/s), and radius (m) for circular motion."


def _solve_gravitation(v: dict, text: str) -> tuple[bool, str]:
    # Extract all numeric values followed by 'kg' as masses
    masses = re.findall(r'([\d.e\+\-]+)\s*kg\b', text, re.I)
    # Also try m1= / m2= format
    m1_m = re.search(r'm[₁1]\s*=\s*([\d.e\+\-]+)', text, re.I)
    m2_m = re.search(r'm[₂2]\s*=\s*([\d.e\+\-]+)', text, re.I)
    if m1_m and m2_m:
        masses = [m1_m.group(1), m2_m.group(1)]
    r_m = re.search(r'(?:distance|radius|separation|apart)\s+(?:of\s+|=\s*)?([\d.e\+\-]+)\s*(?:m|km)?', text, re.I)
    # Also accept plain "100m" patterns if distance keyword present
    if r_m is None:
        r_m = re.search(r',\s*([\d.e\+\-]+)\s*m\b', text, re.I)

    if len(masses) >= 2 and r_m:
        m1, m2 = float(masses[0]), float(masses[1])
        r_ = float(r_m.group(1))
        # If km → convert
        if re.search(r'[\d.]+\s*km', text, re.I):
            r_ *= 1000
        F_grav = G * m1 * m2 / r_**2
        return True, (
            f"F = Gm₁m₂/r² = {G:.4e} × {m1:.4g} × {m2:.4g} / {r_:.4g}²"
            f" = {F_grav:.4e} N"
        )
    return False, "Provide two masses (kg) and separation distance (m)."


def _solve_friction(v: dict, text: str) -> tuple[bool, str]:
    mu  = v.get('friction_coefficient')
    m   = v.get('mass')
    N_  = v.get('force')    # sometimes normal force is given directly

    # Fallback: parse mu from text if not in vals
    if mu is None:
        mu_m = re.search(r'(?:μ|mu|coefficient(?:\s+of\s+friction)?)\s*=?\s*([\d.]+)', text, re.I)
        if mu_m:
            mu = float(mu_m.group(1))

    if mu is not None and m is not None:
        N  = m * g
        Ff = mu * N
        return True, (
            f"Normal force N = mg = {m}×{g} = {N:.4g} N\n"
            f"Friction force f = μN = {mu}×{N:.4g} = {Ff:.4g} N"
        )
    if mu is not None and N_ is not None:
        Ff = mu * N_
        return True, f"Friction force f = μN = {mu} × {N_} = {Ff:.4g} N"
    return False, "Provide coefficient of friction (μ) and mass (kg) or normal force (N)."


# ─────────────────────────────────────────────────────────────────────────────
# New formula solvers (T001 gap-fill)
# ─────────────────────────────────────────────────────────────────────────────

def _solve_escape_velocity(v: dict, text: str) -> tuple[bool, str]:
    t = text.lower()
    g_  = _num(t, r'g\s*=\s*([\d.e\-]+)', r'([\d.e\-]+)\s*m/s[²2]') or g
    R_  = v.get('radius') or _num(t,
        r'R\s*=\s*([\d.e\+\-]+)', r'radius\s+(?:of\s+|=\s*)?([\d.e\+\-]+)',
        r'([\d.e\+\-]+)\s*m\b',
    )
    M_  = _num(t, r'M\s*=\s*([\d.e\+\-]+)', r'mass\s+of\s+(?:the\s+)?(?:planet|earth|moon)\s+(?:=\s*)?([\d.e\+\-]+)')

    if R_ is not None:
        if M_ is not None:
            ve = math.sqrt(2 * G * M_ / R_)
            return True, (
                f"Escape velocity: v_e = √(2GM/R)\n"
                f"  = √(2 × {G:.4e} × {M_:.4e} / {R_:.4e})\n"
                f"  = {ve:.4g} m/s  ({ve/1000:.4g} km/s)"
            )
        ve = math.sqrt(2 * g_ * R_)
        return True, (
            f"Escape velocity: v_e = √(2gR)\n"
            f"  = √(2 × {g_} × {R_:.4e})\n"
            f"  = {ve:.4g} m/s  ({ve/1000:.4g} km/s)"
        )
    return False, "Provide planet radius R (m) and optionally mass M (kg) or surface gravity g (m/s²)."


def _solve_spring_energy(v: dict, text: str) -> tuple[bool, str]:
    k_ = v.get('spring_constant')
    x_ = v.get('spring_compression')
    if k_ is not None and x_ is not None:
        E = 0.5 * k_ * x_**2
        return True, (
            f"Elastic PE: E = ½kx²\n"
            f"  = ½ × {k_} × {x_}²\n"
            f"  = {E:.4g} J"
        )
    if k_ is not None:
        return False, f"Spring constant k = {k_} N/m found. Provide compression/extension x (m)."
    return False, "Provide spring constant k (N/m) and compression/extension x (m)."


def _solve_de_broglie(v: dict, text: str) -> tuple[bool, str]:
    t = text.lower()
    mass_  = v.get('mass')
    speed_ = v.get('speed') or v.get('initial_velocity') or v.get('final_velocity')

    # Determine particle type if mass not given
    if mass_ is None:
        if re.search(r'\belectron\b', t):
            mass_ = m_e
            particle = "electron"
        elif re.search(r'\bproton\b', t):
            mass_ = m_p
            particle = "proton"
        else:
            particle = "particle"
    else:
        particle = "particle"

    if mass_ is not None and speed_ is not None:
        lam = h / (mass_ * speed_)
        p   = mass_ * speed_
        return True, (
            f"de Broglie wavelength: λ = h/(mv)\n"
            f"  h = {h:.4e} J·s,  m = {mass_:.4e} kg ({particle}),  v = {speed_:.4e} m/s\n"
            f"  p = mv = {p:.4e} kg·m/s\n"
            f"  λ = {h:.4e} / {p:.4e} = {lam:.4e} m"
        )
    return False, "Provide particle mass (kg) and velocity (m/s) — or name 'electron'/'proton' and velocity."


def _solve_moment_of_inertia(v: dict, text: str) -> tuple[bool, str]:
    t = text.lower()
    M_ = v.get('mass')
    R_ = v.get('radius')
    L_ = _num(t, r'length\s+(?:of\s+|=\s*)?([\d.]+)', r'L\s*=\s*([\d.]+)')

    results = []
    if re.search(r'\bsolid\s+sphere\b', t) and M_ and R_:
        I = 0.4 * M_ * R_**2
        results.append(f"Solid sphere: I = (2/5)MR² = 0.4 × {M_} × {R_}² = {I:.4g} kg·m²")
    if re.search(r'\bhollow\s+sphere\b|\bspherical\s+shell\b|\bthin\s+shell\b', t) and M_ and R_:
        I = (2/3) * M_ * R_**2
        results.append(f"Hollow sphere (thin shell): I = (2/3)MR² = {I:.4g} kg·m²")
    if re.search(r'\bsolid\s+(?:cylinder|disc|disk)\b|\bdisc\b|\bdisk\b', t) and M_ and R_:
        I = 0.5 * M_ * R_**2
        results.append(f"Solid cylinder/disc: I = (1/2)MR² = 0.5 × {M_} × {R_}² = {I:.4g} kg·m²")
    if re.search(r'\bhollow\s+cylinder\b|\bthin\s+(?:ring|hoop)\b|\bring\b', t) and M_ and R_:
        I = M_ * R_**2
        results.append(f"Thin ring/hollow cylinder: I = MR² = {M_} × {R_}² = {I:.4g} kg·m²")
    if re.search(r'\brod\b|\bstick\b|\bbar\b', t) and M_ and L_:
        I_cm = (1/12) * M_ * L_**2
        I_end = (1/3) * M_ * L_**2
        results.append(f"Rod about centre: I = (1/12)ML² = {I_cm:.4g} kg·m²")
        results.append(f"Rod about end:    I = (1/3)ML²  = {I_end:.4g} kg·m²")

    # Generic sphere fallback (no solid/hollow keyword)
    if not results and M_ and R_:
        I_solid = 0.4 * M_ * R_**2
        I_hol   = (2/3) * M_ * R_**2
        results.append(f"Solid sphere:  I = (2/5)MR² = {I_solid:.4g} kg·m²")
        results.append(f"Hollow sphere: I = (2/3)MR² = {I_hol:.4g} kg·m²")

    if results:
        return True, "\n".join(results)
    return False, "Provide mass M (kg) and radius R (m) [sphere/cylinder] or length L (m) [rod]."


def _solve_coulomb(v: dict, text: str) -> tuple[bool, str]:
    t = text.lower()
    # Extract two charge values with sign context — handle both µC and microC
    UC_RE = r'[µμ]C\b|micro\s*C\b'
    plus_vals  = re.findall(r'\+([\d.e\-]+)\s*(?:[µμ]C\b|micro\s*C\b)', text, re.I)
    minus_vals = re.findall(r'[-−]([\d.e\-]+)\s*(?:[µμ]C\b|micro\s*C\b)', text, re.I)
    all_vals   = re.findall(r'([\d.e\-]+)\s*(?:[µμ]C\b|micro\s*C\b)', text, re.I)
    # Fall back to signed all_vals to get sign from ±
    if not all_vals:
        all_vals = re.findall(r'([+\-]?[\d.e]+)\s*(?:[µμ]C\b|micro\s*C\b)', text, re.I)

    q1 = q2 = None
    if len(all_vals) >= 2:
        q1 = float(all_vals[0]) * 1e-6
        q2 = float(all_vals[1]) * 1e-6
    elif v.get('charge1') is not None:
        q1 = v['charge1']
        q2 = v.get('charge2')

    r_ = v.get('separation') or _num(t,
        r'([\d.]+)\s*m\s+apart', r'apart\s+(?:by\s+)?([\d.]+)',
        r'distance\s+of\s+([\d.]+)\s*m', r'([\d.]+)\s*m\b',
    )

    if q1 is not None and q2 is not None and r_ is not None:
        F = k_e * abs(q1) * abs(q2) / r_**2
        nature = "attractive" if (q1 * q2 < 0) else "repulsive"
        return True, (
            f"Coulomb's Law: F = k|q₁||q₂|/r²\n"
            f"  k = {k_e:.4e} N·m²/C²\n"
            f"  q₁ = {q1:.4e} C,  q₂ = {q2:.4e} C,  r = {r_} m\n"
            f"  F = {k_e:.4e} × {abs(q1):.4e} × {abs(q2):.4e} / {r_}²\n"
            f"  F = {F:.4e} N  ({nature})"
        )
    return False, "Provide two charges (C or µC) and separation distance (m)."


def _solve_gravitational_pe(v: dict, text: str) -> tuple[bool, str]:
    t = text.lower()
    masses = re.findall(r'([\d.e\+\-]+)\s*kg\b', text, re.I)
    m1_m = re.search(r'm[₁1]\s*=\s*([\d.e\+\-]+)', text, re.I)
    m2_m = re.search(r'm[₂2]\s*=\s*([\d.e\+\-]+)', text, re.I)
    if m1_m and m2_m:
        masses = [m1_m.group(1), m2_m.group(1)]
    # Handle "two X kg masses" — only one kg value but word "two" means both masses = X
    if len(masses) == 1 and re.search(r'\btwo\b', t):
        masses = [masses[0], masses[0]]
    r_ = v.get('separation') or _num(t,
        r'([\d.]+)\s*m\s+apart', r'apart\s+by\s+([\d.]+)',
        r'distance\s+(?:of\s+|=\s*)?([\d.]+)\s*m', r'at\s+([\d.]+)\s*m\b',
    )
    if len(masses) >= 2 and r_:
        m1, m2 = float(masses[0]), float(masses[1])
        U = -G * m1 * m2 / r_
        return True, (
            f"Gravitational PE: U = −Gm₁m₂/r\n"
            f"  G = {G:.4e} N·m²/kg²\n"
            f"  U = −{G:.4e} × {m1} × {m2} / {r_}\n"
            f"  U = {U:.4e} J"
        )
    return False, "Provide two masses (kg) and separation distance (m)."


def _solve_diffraction(v: dict, text: str) -> tuple[bool, str]:
    t = text.lower()
    lines_mm = v.get('lines_per_mm')
    order    = v.get('diffraction_order', 1)
    lam      = v.get('wavelength')
    angle    = v.get('angle')

    # Grating spacing d = 1/N (in mm), convert to m
    d = (1.0 / lines_mm) * 1e-3 if lines_mm else None

    # Wavelength from nm if not found
    nm_m = re.search(r'([\d.]+)\s*nm\b', t)
    if nm_m and lam is None:
        lam = float(nm_m.group(1)) * 1e-9

    if d is not None and lam is not None:
        sin_val = order * lam / d
        if abs(sin_val) <= 1:
            theta = math.degrees(math.asin(sin_val))
            return True, (
                f"Diffraction grating: d·sin θ = mλ\n"
                f"  d = 1/{lines_mm:.0f} mm = {d:.4e} m  (grating spacing)\n"
                f"  m = {order} (order),  λ = {lam:.4e} m\n"
                f"  sin θ = mλ/d = {order}×{lam:.4e}/{d:.4e} = {sin_val:.4f}\n"
                f"  θ = arcsin({sin_val:.4f}) = {theta:.2f}°"
            )
        return False, f"sin θ = {sin_val:.4f} > 1 — this order does not exist for given wavelength and grating."
    if d is not None and angle is not None:
        lam_calc = d * math.sin(math.radians(angle)) / order
        return True, (
            f"Diffraction grating: d·sin θ = mλ\n"
            f"  d = {d:.4e} m,  θ = {angle}°,  m = {order}\n"
            f"  λ = d·sin θ / m = {lam_calc:.4e} m"
        )
    return False, "Provide grating (lines/mm), wavelength (nm), and order number."


def _solve_projectile(v: dict, text: str) -> tuple[bool, str]:
    t = text.lower()
    u_     = v.get('initial_velocity') or v.get('speed')
    theta_ = v.get('launch_angle')

    if u_ is None or theta_ is None:
        return False, "Provide initial velocity (m/s) and launch angle (degrees) for projectile motion."

    theta_r = math.radians(theta_)
    ux = u_ * math.cos(theta_r)
    uy = u_ * math.sin(theta_r)

    T  = 2 * uy / g                           # time of flight
    H  = uy**2 / (2 * g)                      # max height
    Rg = u_**2 * math.sin(2 * theta_r) / g   # horizontal range

    return True, (
        f"Projectile Motion: u = {u_} m/s at θ = {theta_}°\n"
        f"  uₓ = u·cos θ = {u_}·cos({theta_}°) = {ux:.4g} m/s\n"
        f"  u_y = u·sin θ = {u_}·sin({theta_}°) = {uy:.4g} m/s\n\n"
        f"  Time of flight: T = 2u_y/g = 2×{uy:.4g}/{g} = {T:.4g} s\n"
        f"  Maximum height: H = u_y²/(2g) = {uy:.4g}²/(2×{g}) = {H:.4g} m\n"
        f"  Horizontal range: R = u²·sin2θ/g = {u_}²·sin({2*theta_}°)/{g} = {Rg:.4g} m"
    )


def _solve_magnetic_force(v: dict, text: str) -> tuple[bool, str]:
    t = text.lower()
    q_ = v.get('charge') or _num(t,
        r'q\s*=\s*([\d.e\-]+)\s*[µμ]C',
        r'([\d.e\-]+)\s*[µμ]C\b',
    )
    if q_ and re.search(r'[µμ]C', t) and q_ > 1e-3:
        q_ *= 1e-6

    spd = v.get('speed') or v.get('initial_velocity') or _num(t,
        r'v\s*=\s*([\d.e\+\-]+)',
        r'moving\s+at\s+([\d.e\+\-]+)',
        r'velocity\s+(?:of\s+|=\s*)?([\d.e\+\-]+)',
    )
    B_  = v.get('magnetic_field')
    ang_m = re.search(r'angle\s+(?:of\s+|=\s*)?([\d.]+)\s*°', t)
    sin_theta = math.sin(math.radians(float(ang_m.group(1)))) if ang_m else 1.0  # perpendicular default

    if q_ is not None and spd is not None and B_ is not None:
        F = abs(q_) * spd * B_ * sin_theta
        angle_str = f"·sin({ang_m.group(1)}°)" if ang_m else " (perpendicular, sinθ = 1)"
        return True, (
            f"Magnetic Force: F = qvB·sinθ\n"
            f"  q = {q_:.4e} C,  v = {spd:.4e} m/s,  B = {B_:.4e} T{angle_str}\n"
            f"  F = {abs(q_):.4e} × {spd:.4e} × {B_:.4e}{' × sin(' + ang_m.group(1) + '°)' if ang_m else ''}\n"
            f"  F = {F:.4e} N"
        )
    return False, "Provide charge q (C or µC), velocity v (m/s), and magnetic field B (T)."


# ─────────────────────────────────────────────────────────────────────────────
# Public interface
# ─────────────────────────────────────────────────────────────────────────────

_SOLVERS = {
    'kinematics':         _solve_kinematics,
    'projectile':         _solve_projectile,
    'force':              _solve_force,
    'weight':             _solve_weight,
    'kinetic_energy':     _solve_kinetic_energy,
    'potential_energy':   _solve_potential_energy,
    'spring_energy':      _solve_spring_energy,
    'work':               _solve_work,
    'power':              _solve_power,
    'momentum':           _solve_momentum,
    'waves':              _solve_waves,
    'photon':             _solve_photon,
    'de_broglie':         _solve_de_broglie,
    'ohm':                _solve_ohm,
    'electric_power':     _solve_electric_power,
    'coulomb':            _solve_coulomb,
    'magnetic_force':     _solve_magnetic_force,
    'heat':               _solve_heat,
    'pressure':           _solve_pressure,
    'density':            _solve_density,
    'fluid_pressure':     _solve_fluid_pressure,
    'snell':              _solve_snell,
    'lens':               _solve_lens,
    'circular':           _solve_circular,
    'gravitation':        _solve_gravitation,
    'gravitational_pe':   _solve_gravitational_pe,
    'escape_velocity':    _solve_escape_velocity,
    'moment_of_inertia':  _solve_moment_of_inertia,
    'diffraction':        _solve_diffraction,
    'friction':           _solve_friction,
}


class PhysicsEngine:
    """Deterministic physics formula solver."""

    CONSTANTS = {
        'g':   f"{g} m/s² (standard gravity)",
        'G':   f"{G:.4e} N·m²/kg² (gravitational constant)",
        'h':   f"{h:.4e} J·s (Planck constant)",
        'c':   f"{c:.4e} m/s (speed of light)",
        'R':   f"{R} J/(mol·K) (ideal gas constant)",
        'e':   f"{e:.4e} C (elementary charge)",
        'k_B': f"{k_B:.4e} J/K (Boltzmann constant)",
    }

    def solve(self, text: str) -> tuple[bool, str, str]:
        """
        Attempt to solve a physics problem from natural language.
        Returns (success, result_string, formula_type).
        """
        qtype  = _detect_question_type(text)
        vals   = _extract(text)
        solver = _SOLVERS.get(qtype)

        if solver is None:
            return False, "Could not identify the physics formula to apply.", "unknown"

        try:
            success, result = solver(vals, text)
            return success, result, qtype
        except Exception as exc:
            return False, f"Calculation error: {exc}", qtype

    def is_physics_question(self, text: str) -> bool:
        """Quick check — used by the router."""
        return _detect_question_type(text) != 'unknown'
