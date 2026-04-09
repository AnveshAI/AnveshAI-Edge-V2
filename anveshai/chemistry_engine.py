"""
Chemistry Engine — deterministic solver for quantitative chemistry problems.

Supported calculations:
  · Molar mass         : formula → M (g/mol) using full periodic table
  · Moles              : n = m/M,  m = nM,  M = m/n
  · Molarity           : C = n/V,  n = CV,  V = n/C
  · Dilution           : C₁V₁ = C₂V₂
  · pH & pOH           : pH = −log[H⁺],  pOH = −log[OH⁻],  pH + pOH = 14
  · Weak acid/base     : Ka, Kb approximation (√(Ka·C))
  · Boyle's law        : P₁V₁ = P₂V₂
  · Charles's law      : V₁/T₁ = V₂/T₂
  · Gay-Lussac's law   : P₁/T₁ = P₂/T₂
  · Ideal gas law      : PV = nRT
  · Combined gas law   : P₁V₁/T₁ = P₂V₂/T₂
  · Heat / calorimetry : q = mcΔT,  q = nΔH
  · Percent composition: % = (part_mass / molar_mass) × 100
  · Nuclear half-life  : N = N₀ × (½)^(t/t½),  activity decay
  · Electrochemistry   : ΔG = −nFE,  Nernst equation (conceptual)

All formulas solved purely in Python — no external dependencies.
"""

from __future__ import annotations

import re
import math
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Periodic table — atomic masses (g/mol), all 118 elements
# ─────────────────────────────────────────────────────────────────────────────

ATOMIC_MASS: dict[str, float] = {
    'H': 1.008,   'He': 4.003,  'Li': 6.941,  'Be': 9.012,
    'B': 10.81,   'C': 12.011,  'N': 14.007,  'O': 15.999,
    'F': 18.998,  'Ne': 20.180, 'Na': 22.990, 'Mg': 24.305,
    'Al': 26.982, 'Si': 28.086, 'P': 30.974,  'S': 32.065,
    'Cl': 35.453, 'Ar': 39.948, 'K': 39.098,  'Ca': 40.078,
    'Sc': 44.956, 'Ti': 47.867, 'V': 50.942,  'Cr': 51.996,
    'Mn': 54.938, 'Fe': 55.845, 'Co': 58.933, 'Ni': 58.693,
    'Cu': 63.546, 'Zn': 65.38,  'Ga': 69.723, 'Ge': 72.640,
    'As': 74.922, 'Se': 78.960, 'Br': 79.904, 'Kr': 83.798,
    'Rb': 85.468, 'Sr': 87.620, 'Y': 88.906,  'Zr': 91.224,
    'Nb': 92.906, 'Mo': 95.960, 'Tc': 98.0,   'Ru': 101.07,
    'Rh': 102.91, 'Pd': 106.42, 'Ag': 107.87, 'Cd': 112.41,
    'In': 114.82, 'Sn': 118.71, 'Sb': 121.76, 'Te': 127.60,
    'I': 126.90,  'Xe': 131.29, 'Cs': 132.91, 'Ba': 137.33,
    'La': 138.91, 'Ce': 140.12, 'Pr': 140.91, 'Nd': 144.24,
    'Pm': 145.0,  'Sm': 150.36, 'Eu': 151.96, 'Gd': 157.25,
    'Tb': 158.93, 'Dy': 162.50, 'Ho': 164.93, 'Er': 167.26,
    'Tm': 168.93, 'Yb': 173.05, 'Lu': 174.97, 'Hf': 178.49,
    'Ta': 180.95, 'W': 183.84,  'Re': 186.21, 'Os': 190.23,
    'Ir': 192.22, 'Pt': 195.08, 'Au': 196.97, 'Hg': 200.59,
    'Tl': 204.38, 'Pb': 207.20, 'Bi': 208.98, 'Po': 209.0,
    'At': 210.0,  'Rn': 222.0,  'Fr': 223.0,  'Ra': 226.0,
    'Ac': 227.0,  'Th': 232.04, 'Pa': 231.04, 'U': 238.03,
    'Np': 237.0,  'Pu': 244.0,  'Am': 243.0,  'Cm': 247.0,
    'Bk': 247.0,  'Cf': 251.0,  'Es': 252.0,  'Fm': 257.0,
    'Md': 258.0,  'No': 259.0,  'Lr': 262.0,  'Rf': 267.0,
    'Db': 268.0,  'Sg': 271.0,  'Bh': 274.0,  'Hs': 269.0,
    'Mt': 278.0,  'Ds': 281.0,  'Rg': 282.0,  'Cn': 285.0,
    'Nh': 286.0,  'Fl': 289.0,  'Mc': 290.0,  'Lv': 293.0,
    'Ts': 294.0,  'Og': 294.0,
}

# Gas constant
R_gas = 8.314   # J/(mol·K)
F_const = 96485  # Faraday's constant (C/mol)


# ─────────────────────────────────────────────────────────────────────────────
# Molar mass calculation from chemical formula string
# ─────────────────────────────────────────────────────────────────────────────

def _parse_formula(formula: str) -> dict[str, float]:
    """
    Parse a chemical formula like H2O, Ca(OH)2, Fe2(SO4)3 into {element: count}.
    Returns {} if formula cannot be parsed.
    """
    def _parse_segment(seg: str, mult: int = 1) -> dict[str, float]:
        counts: dict[str, float] = {}
        i = 0
        while i < len(seg):
            if seg[i] == '(':
                # Find matching closing parenthesis
                depth, j = 1, i + 1
                while j < len(seg) and depth > 0:
                    if seg[j] == '(':   depth += 1
                    elif seg[j] == ')': depth -= 1
                    j += 1
                inner = seg[i+1:j-1]
                # Read subscript after ')'
                k = j
                while k < len(seg) and seg[k].isdigit():
                    k += 1
                sub = int(seg[j:k]) if j < k else 1
                inner_counts = _parse_segment(inner, mult * sub)
                for el, cnt in inner_counts.items():
                    counts[el] = counts.get(el, 0) + cnt
                i = k
            elif seg[i].isupper():
                # Read element symbol
                j = i + 1
                while j < len(seg) and seg[j].islower():
                    j += 1
                el = seg[i:j]
                # Read subscript
                k = j
                while k < len(seg) and seg[k].isdigit():
                    k += 1
                sub = int(seg[j:k]) if j < k else 1
                counts[el] = counts.get(el, 0) + sub * mult
                i = k
            else:
                i += 1
        return counts

    # Strip charge notation like 2+ or 3- or ²⁺
    formula = re.sub(r'[\d\+\-²³⁴⁵⁶⁷⁸⁹⁺⁻]+$', '', formula).strip()
    return _parse_segment(formula)


def calc_molar_mass(formula: str) -> tuple[bool, float, str]:
    """
    Calculate molar mass from a chemical formula.
    Returns (success, molar_mass, breakdown_string).
    """
    counts = _parse_formula(formula)
    if not counts:
        return False, 0.0, f"Cannot parse formula '{formula}'"

    total = 0.0
    breakdown = []
    for el, cnt in counts.items():
        mass = ATOMIC_MASS.get(el)
        if mass is None:
            return False, 0.0, f"Unknown element '{el}' in formula"
        contrib = mass * cnt
        total += contrib
        breakdown.append(f"{el}: {cnt} × {mass:.4g} = {contrib:.4g}")

    detail = "\n  ".join(breakdown)
    return True, total, f"Molar mass of {formula}:\n  {detail}\n  Total M = {total:.4f} g/mol"


# ─────────────────────────────────────────────────────────────────────────────
# Number extractor
# ─────────────────────────────────────────────────────────────────────────────

def _n(text: str, *patterns: str) -> Optional[float]:
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                continue
    return None


def _extract_formula_from_text(text: str) -> Optional[str]:
    """Find a chemical formula in text (e.g., H2O, NaCl, Ca(OH)2)."""
    # Match standard chemical formulas — starts with uppercase letter
    m = re.search(
        r'\b([A-Z][a-z]?\d*(?:\([A-Z][a-z]?\d*(?:\([A-Z][a-z]?\d*\)\d*)*\)\d*|[A-Z][a-z]?\d*)*\d*)\b',
        text
    )
    if m and len(m.group(1)) >= 2:
        cand = m.group(1)
        # Sanity check: must contain at least one valid element
        if any(el in cand for el in ['H', 'C', 'N', 'O', 'Na', 'Ca', 'Fe', 'Cl', 'S', 'P', 'K']):
            return cand
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Problem-type detector
# ─────────────────────────────────────────────────────────────────────────────

def _detect_chem_type(text: str) -> str:
    # Use re.IGNORECASE throughout so mixed-case symbols (Ka, pH, Kb, etc.) always match.
    flags = re.IGNORECASE

    if re.search(r'molar\s+mass|molecular\s+weight|atomic\s+mass|formula\s+mass|relative\s+molecular', text, flags):
        return 'molar_mass'
    # weak_acid / weak_base BEFORE molarity to avoid "concentration" stealing the match
    if re.search(r'\bdegree\s+of\s+dissociation\b|\bfraction\s+dissociated\b', text, flags):
        return 'dissociation_degree'
    if re.search(r'\bKa\b|\bKb\b|\bweak\s+acid\b|\bweak\s+base\b', text, flags):
        return 'weak_acid'
    # pH / pOH BEFORE molarity
    if re.search(r'\bpH\b|\bpOH\b|\b\[H\+?\]|\[OH-?\]|\bacid\s+concentration|\bbase\s+concentration', text, flags):
        return 'ph'
    if re.search(r'\bhow\s+many\s+moles\b|\bn\s*=\s*m/M|\bmoles?\s+of\b', text, flags):
        return 'moles'
    if re.search(r'\bmoles?\b.*\bmass\b|\bmass\b.*\bmoles?\b', text, flags):
        return 'moles'
    if re.search(r'dilut|\bC[₁1]V[₁1]\b|\bC1V1\b', text, flags):
        return 'dilution'
    if re.search(r'molarity|mol(?:es?)?\s*/\s*L', text, flags):
        return 'molarity'
    if re.search(r'\bconcentration\b', text, flags) and not re.search(r'\bweak\b|\bKa\b|\bKb\b|\bpH\b', text, flags):
        return 'molarity'
    if re.search(r'combined\s+gas\s+law', text, flags):
        return 'combined_gas'
    if re.search(r"boyle'?s?\s+law|P[₁1]V[₁1]\s*=\s*P[₂2]V[₂2]|P1V1\s*=\s*P2V2", text, flags):
        return 'boyle'
    if re.search(r"charles'?s?\s+law|V[₁1]/T[₁1]\s*=\s*V[₂2]/T[₂2]|V1/T1", text, flags):
        return 'charles'
    if re.search(r'gay.?lussac|P[₁1]/T[₁1]|P1/T1', text, flags):
        return 'gay_lussac'
    if re.search(r'ideal\s+gas\s+law|\bPV\s*=\s*nRT\b', text, flags):
        return 'ideal_gas'
    # Ideal gas thermodynamics — must come BEFORE calorimetry
    if re.search(
        r'monatomic|diatomic|triatomic'
        r'|molar\s+heat\s+capacity'
        r'|\bCv\b|\bCp\b|\bcv\s*=|\bcp\s*='
        r'|\binternal\s+energy\b'
        r'|\bΔU\b|\bdelta\s+u\b'
        r'|\bconstant\s+(?:pressure|volume)\b'
        r'|\bisobaric\b|\bisochoric\b',
        text, flags
    ):
        return 'ideal_gas_thermo'
    # Gas name + moles/heat/energy → ideal gas thermo
    if re.search(r'\b(?:argon|helium|neon|krypton|xenon|nitrogen|oxygen|hydrogen)\b', text, flags) and \
       re.search(r'\bmol(?:es?)?\b', text, flags) and \
       re.search(r'\b(?:heat|energy|temperature|J\b|joule)', text, flags):
        return 'ideal_gas_thermo'
    if re.search(r'\bheat\b|\bcalorim|\bq\s*=\s*mc|\bspecific\s+heat\b|\benthalpy\b', text, flags):
        return 'calorimetry'
    if re.search(r'percent\s+composition|mass\s+percent|%\s+by\s+mass|percentage\s+composition', text, flags):
        return 'percent_composition'
    if re.search(r'half.?life|\bradioact|\bdecay|\bN\s*=\s*N[₀0]', text, flags):
        return 'half_life'
    if re.search(r'electroch|nernst|cell\s+potential|\bE°?\s*=|\bΔG\b|\bfaraday', text, flags):
        return 'electrochemistry'
    if re.search(r'stoichiom|molar\s+ratio|limiting\s+reagent|theoretical\s+yield', text, flags):
        return 'stoichiometry'
    if re.search(r'\bempir(?:ical)?\s+formula\b|%.*C.*%.*H|%.*composition.*formula|from\s+percent(?:age)?', text, flags):
        return 'empirical_formula'
    if re.search(r'\boxidation\s+state\b|\boxidation\s+number\b|\bvalence\s+state\b', text, flags):
        return 'oxidation_state'
    if re.search(r'\bnormality\b|\bnormal\s+solution\b|\bN\s*=\s*M\b|\bequivalents?\b', text, flags):
        return 'normality'
    if re.search(r'\b(?:moles?\s+(?:in|of|at)|number\s+of\s+moles?\s+(?:in|at))\b.*\bSTP\b'
                 r'|\bSTP\b.*\bmoles?\b|\b22\.4\s*[Ll]\b', text, flags):
        return 'moles_stp'
    if re.search(r'\bbalance\s+(?:the\s+)?equation\b|\bbalanced?\s+(?:chemical\s+)?equation\b', text, flags):
        return 'balance_equation'
    if re.search(r'\bgrams?\s+(?:of\s+\w+\s+)?(?:are\s+)?produced\b|\bhow\s+many\s+grams?\b.*\bproduced\b'
                 r'|\b\d+\s+mol\s+\w+\b.*\bgrams?\b|\btheoretical\s+mass\b', text, flags):
        return 'stoichiometry_mass'
    return 'unknown'


# ─────────────────────────────────────────────────────────────────────────────
# Solvers
# ─────────────────────────────────────────────────────────────────────────────

def _solve_molar_mass(text: str) -> tuple[bool, str]:
    # Extract formula from text
    formula = _extract_formula_from_text(text)
    # Also look for explicit "of X" or "for X"
    m = re.search(r'(?:of|for|:)\s*([A-Z][a-zA-Z0-9()]+)', text)
    if m:
        formula = m.group(1)

    if not formula:
        return False, "Could not find a chemical formula in the question."

    return calc_molar_mass(formula)[:2][0], (calc_molar_mass(formula)[2] if calc_molar_mass(formula)[0] else calc_molar_mass(formula)[2])


def _solve_moles(text: str) -> tuple[bool, str]:
    t = text.lower()
    mass_g = _n(t,
        r'([\d.]+)\s*g\b',
        r'mass\s+(?:of\s+|=\s*)?([\d.]+)',
        r'm\s*=\s*([\d.]+)',
    )
    M = _n(t,
        r'molar\s+mass\s+(?:of\s+|=\s*)?([\d.]+)',
        r'M\s*=\s*([\d.]+)',
        r'([\d.]+)\s*g/mol',
    )
    n = _n(t,
        r'n\s*=\s*([\d.]+)',
        r'([\d.]+)\s*mol(?:es?)?\b',
    )

    # Try to compute molar mass from formula if not given directly
    formula = _extract_formula_from_text(text)
    if formula and M is None:
        ok, M_calc, _ = calc_molar_mass(formula)
        if ok:
            M = M_calc

    results = []
    if mass_g is not None and M is not None:
        n_calc = mass_g / M
        results.append(f"n = m/M = {mass_g} g ÷ {M:.4g} g/mol = {n_calc:.4g} mol")
    if n is not None and M is not None and mass_g is None:
        m_calc = n * M
        results.append(f"m = nM = {n} mol × {M:.4g} g/mol = {m_calc:.4g} g")
    if n is not None and mass_g is not None and M is None:
        M_calc = mass_g / n
        results.append(f"M = m/n = {mass_g}/{n} = {M_calc:.4g} g/mol")

    if results:
        return True, "\n".join(results)
    return False, "Provide mass (g), molar mass (g/mol), or formula to calculate moles."


def _solve_molarity(text: str) -> tuple[bool, str]:
    t = text.lower()
    n  = _n(t, r'([\d.]+)\s*mol(?:es?)?\b', r'n\s*=\s*([\d.]+)')
    V  = _n(t,
        r'([\d.]+)\s*(?:L|liters?|litres?)\b',
        r'volume\s+(?:of\s+|=\s*)?([\d.]+)',
        r'V\s*=\s*([\d.]+)',
    )
    C  = _n(t,
        r'concentration\s+(?:of\s+|=\s*)?([\d.]+)',
        r'([\d.]+)\s*(?:M|mol/L|mol L-1|M\b)',
        r'C\s*=\s*([\d.]+)',
        r'molarity\s+(?:of\s+|=\s*)?([\d.]+)',
    )

    # Convert mL → L
    vml_m = re.search(r'([\d.]+)\s*mL\b', t, re.I)
    if vml_m and V is None:
        V = float(vml_m.group(1)) / 1000

    results = []
    if n is not None and V is not None:
        C_calc = n / V
        results.append(f"C = n/V = {n} mol ÷ {V} L = {C_calc:.4g} mol/L")
    if C is not None and V is not None and n is None:
        n_calc = C * V
        results.append(f"n = CV = {C} M × {V} L = {n_calc:.4g} mol")
    if C is not None and n is not None and V is None:
        V_calc = n / C
        results.append(f"V = n/C = {n}/{C} = {V_calc:.4g} L")

    if results:
        return True, "\n".join(results)
    return False, "Provide moles (mol) and volume (L), or concentration (M) and volume to solve molarity."


def _solve_dilution(text: str) -> tuple[bool, str]:
    t = text.lower()
    nums = re.findall(r'([\d.]+)\s*(?:M|mol/L|mL|L|liters?)?', t)
    floats = [float(x) for x in nums if x]

    # Try explicit variable extraction
    C1 = _n(t, r'C[₁1]\s*=\s*([\d.]+)', r'initial\s+concentration\s+(?:of\s+|=\s*)?([\d.]+)')
    V1 = _n(t, r'V[₁1]\s*=\s*([\d.]+)', r'initial\s+volume\s+(?:of\s+|=\s*)?([\d.]+)')
    C2 = _n(t, r'C[₂2]\s*=\s*([\d.]+)', r'final\s+concentration\s+(?:of\s+|=\s*)?([\d.]+)')
    V2 = _n(t, r'V[₂2]\s*=\s*([\d.]+)', r'final\s+volume\s+(?:of\s+|=\s*)?([\d.]+)')

    known = [x for x in [C1, V1, C2, V2] if x is not None]

    if len(known) == 3 and C1 and V1 and C2 and V2 is None:
        V2_calc = C1 * V1 / C2
        return True, (
            f"C₁V₁ = C₂V₂\n"
            f"{C1} × {V1} = {C2} × V₂\n"
            f"V₂ = {C1*V1}/{C2} = {V2_calc:.4g} L (or mL, same units as V₁)"
        )
    if len(known) == 3 and C1 and V1 and V2 and C2 is None:
        C2_calc = C1 * V1 / V2
        return True, (
            f"C₁V₁ = C₂V₂\n"
            f"{C1} × {V1} = C₂ × {V2}\n"
            f"C₂ = {C1*V1}/{V2} = {C2_calc:.4g} M"
        )
    return False, ("Provide any three of C₁, V₁, C₂, V₂ for dilution calculation.\n"
                   "Format: 'C1 = 2M, V1 = 50mL, C2 = 0.5M, find V2'")


def _solve_ph(text: str) -> tuple[bool, str]:
    t = text.lower()

    # [H+] concentration directly given
    h_conc = _n(t,
        r'\[H\+?\]\s*=\s*([\d.e\-]+)',
        r'\[H3O\+?\]\s*=\s*([\d.e\-]+)',
        r'H\+?\s+concentration\s+(?:of\s+|=\s*)?([\d.e\-]+)',
        r'([\d.e\-]+)\s*mol/L\s+(?:HCl|H2SO4|HNO3|HBr|HI)',
    )
    oh_conc = _n(t,
        r'\[OH-?\]\s*=\s*([\d.e\-]+)',
        r'OH-?\s+concentration\s+(?:of\s+|=\s*)?([\d.e\-]+)',
    )
    pH_given = _n(t, r'\bpH\s*=\s*([\d.]+)')
    pOH_given = _n(t, r'\bpOH\s*=\s*([\d.]+)')

    # Strong acid: concentration given (e.g., "0.01 M HCl")
    acid_conc = _n(t,
        r'([\d.e\-]+)\s*[Mm](?:ol/L)?\s+(?:HCl|HBr|HI|HNO3|H2SO4|HClO4)',
        r'([\d.]+)\s*[Mm]\s+(?:strong\s+acid|acid)',
    )
    base_conc = _n(t,
        r'([\d.e\-]+)\s*[Mm](?:ol/L)?\s+(?:NaOH|KOH|Ca\(OH\)2|Mg\(OH\)2|LiOH)',
        r'([\d.]+)\s*[Mm]\s+(?:strong\s+base|base)',
    )

    results = []

    if h_conc is not None:
        pH_calc = -math.log10(h_conc)
        pOH_calc = 14 - pH_calc
        results.append(f"pH = −log[H⁺] = −log({h_conc:.4e}) = {pH_calc:.4f}")
        results.append(f"pOH = 14 − pH = 14 − {pH_calc:.4f} = {pOH_calc:.4f}")

    elif acid_conc is not None:
        # Check if H2SO4 (diprotic)
        if 'h2so4' in t:
            h_conc = 2 * acid_conc
            note = f"H₂SO₄ is diprotic → [H⁺] = 2 × {acid_conc:.4g} = {h_conc:.4g} M"
        else:
            h_conc = acid_conc
            note = f"Strong acid (fully dissociates) → [H⁺] = {acid_conc:.4g} M"
        pH_calc = -math.log10(h_conc)
        pOH_calc = 14 - pH_calc
        results.append(note)
        results.append(f"pH = −log[H⁺] = −log({h_conc:.4g}) = {pH_calc:.4f}")
        results.append(f"pOH = 14 − pH = {pOH_calc:.4f}")

    elif base_conc is not None:
        oh_conc = base_conc
        pOH_calc = -math.log10(oh_conc)
        pH_calc = 14 - pOH_calc
        results.append(f"Strong base (fully dissociates) → [OH⁻] = {base_conc:.4g} M")
        results.append(f"pOH = −log[OH⁻] = −log({oh_conc:.4g}) = {pOH_calc:.4f}")
        results.append(f"pH = 14 − pOH = {pH_calc:.4f}")

    elif oh_conc is not None:
        pOH_calc = -math.log10(oh_conc)
        pH_calc = 14 - pOH_calc
        results.append(f"pOH = −log[OH⁻] = −log({oh_conc:.4e}) = {pOH_calc:.4f}")
        results.append(f"pH = 14 − pOH = {pH_calc:.4f}")

    elif pH_given is not None:
        h_conc = 10**(-pH_given)
        pOH_calc = 14 - pH_given
        oh_conc = 10**(-pOH_calc)
        results.append(f"[H⁺] = 10^(−pH) = 10^(−{pH_given}) = {h_conc:.4e} mol/L")
        results.append(f"pOH = 14 − {pH_given} = {pOH_calc:.4f}")
        results.append(f"[OH⁻] = 10^(−pOH) = {oh_conc:.4e} mol/L")

    elif pOH_given is not None:
        pH_calc = 14 - pOH_given
        h_conc = 10**(-pH_calc)
        results.append(f"pH = 14 − pOH = 14 − {pOH_given} = {pH_calc:.4f}")
        results.append(f"[H⁺] = 10^(−{pH_calc:.4f}) = {h_conc:.4e} mol/L")

    if results:
        return True, "\n".join(results)
    return False, ("Provide [H⁺], [OH⁻], or concentration of strong acid/base, or pH/pOH.\n"
                   "Example: 'What is the pH of 0.01 M HCl?'")


def _extract_ka(text: str) -> Optional[float]:
    """Robustly extract Ka value from text, handling 'Ka=1e-4', 'Ka of X is 1.8e-5'."""
    SCI = r'[\d.]+(?:[eE][+\-]?\d+)?'
    m = re.search(rf'Ka\s*=\s*({SCI})', text, re.I)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    m = re.search(rf'Ka\b[^0-9]+({SCI})', text, re.I)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    return None


def _extract_kb(text: str) -> Optional[float]:
    SCI = r'[\d.]+(?:[eE][+\-]?\d+)?'
    m = re.search(rf'Kb\s*=\s*({SCI})', text, re.I) or re.search(rf'Kb\b[^0-9]+({SCI})', text, re.I)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    return None


def _solve_weak_acid(text: str) -> tuple[bool, str]:
    t = text.lower()
    Ka = _extract_ka(text)
    Kb = _extract_kb(text)
    C  = _n(t,
        r'([\d.]+)\s*M\b',
        r'concentration\s+(?:of\s+|=\s*)?([\d.]+)',
        r'C\s*=\s*([\d.]+)',
    )

    if Ka is not None and C is not None:
        h_conc = math.sqrt(Ka * C)
        pH_calc = -math.log10(h_conc)
        percent = (h_conc / C) * 100
        return True, (
            f"Weak acid approximation: [H⁺] ≈ √(Ka × C)\n"
            f"[H⁺] ≈ √({Ka:.4e} × {C}) = {h_conc:.4e} mol/L\n"
            f"pH ≈ {pH_calc:.4f}\n"
            f"Percent dissociation ≈ {percent:.4f}%"
        )
    if Kb is not None and C is not None:
        oh_conc = math.sqrt(Kb * C)
        pOH_calc = -math.log10(oh_conc)
        pH_calc = 14 - pOH_calc
        return True, (
            f"Weak base approximation: [OH⁻] ≈ √(Kb × C)\n"
            f"[OH⁻] ≈ √({Kb:.4e} × {C}) = {oh_conc:.4e} mol/L\n"
            f"pOH ≈ {pOH_calc:.4f}\n"
            f"pH ≈ {pH_calc:.4f}"
        )
    return False, "Provide Ka (or Kb) and concentration (M) for weak acid/base pH calculation."


def _solve_boyle(text: str) -> tuple[bool, str]:
    t = text.lower()
    P1 = _n(t, r'P[₁1]\s*=\s*([\d.]+)', r'initial\s+pressure\s+(?:of\s+|=\s*)?([\d.]+)')
    V1 = _n(t, r'V[₁1]\s*=\s*([\d.]+)', r'initial\s+volume\s+(?:of\s+|=\s*)?([\d.]+)')
    P2 = _n(t, r'P[₂2]\s*=\s*([\d.]+)', r'final\s+pressure\s+(?:of\s+|=\s*)?([\d.]+)')
    V2 = _n(t, r'V[₂2]\s*=\s*([\d.]+)', r'final\s+volume\s+(?:of\s+|=\s*)?([\d.]+)')

    if P1 and V1 and P2 and not V2:
        V2_calc = P1 * V1 / P2
        return True, f"Boyle's Law: P₁V₁ = P₂V₂\n{P1}×{V1} = {P2}×V₂\nV₂ = {V2_calc:.4g} (same units as V₁)"
    if P1 and V1 and V2 and not P2:
        P2_calc = P1 * V1 / V2
        return True, f"Boyle's Law: P₁V₁ = P₂V₂\n{P1}×{V1} = P₂×{V2}\nP₂ = {P2_calc:.4g} (same units as P₁)"
    return False, "Provide any three of P₁, V₁, P₂, V₂ for Boyle's Law."


def _solve_charles(text: str) -> tuple[bool, str]:
    t = text.lower()
    V1 = _n(t, r'V[₁1]\s*=\s*([\d.]+)', r'initial\s+volume\s+(?:of\s+|=\s*)?([\d.]+)')
    T1 = _n(t, r'T[₁1]\s*=\s*([\d.]+)', r'initial\s+temperature\s+(?:of\s+|=\s*)?([\d.]+)')
    V2 = _n(t, r'V[₂2]\s*=\s*([\d.]+)', r'final\s+volume\s+(?:of\s+|=\s*)?([\d.]+)')
    T2 = _n(t, r'T[₂2]\s*=\s*([\d.]+)', r'final\s+temperature\s+(?:of\s+|=\s*)?([\d.]+)')

    # Convert Celsius to Kelvin if temperature looks like Celsius (< 400 and "°C" mentioned)
    if T1 and T1 < 400 and re.search(r'celsius|°C|degree\s+C', text, re.I):
        T1 += 273.15
    if T2 and T2 < 400 and re.search(r'celsius|°C|degree\s+C', text, re.I):
        T2 += 273.15

    if V1 and T1 and T2 and not V2:
        V2_calc = V1 * T2 / T1
        return True, f"Charles's Law: V₁/T₁ = V₂/T₂\nV₂ = V₁×T₂/T₁ = {V1}×{T2}/{T1} = {V2_calc:.4g}"
    if V1 and T1 and V2 and not T2:
        T2_calc = T1 * V2 / V1
        return True, f"Charles's Law: V₁/T₁ = V₂/T₂\nT₂ = T₁×V₂/V₁ = {T1}×{V2}/{V1} = {T2_calc:.4g} K"
    return False, "Provide any three of V₁, T₁, V₂, T₂ for Charles's Law (temperatures in Kelvin)."


def _solve_gay_lussac(text: str) -> tuple[bool, str]:
    t = text.lower()
    P1 = _n(t, r'P[₁1]\s*=\s*([\d.]+)', r'initial\s+pressure\s+(?:of\s+|=\s*)?([\d.]+)')
    T1 = _n(t, r'T[₁1]\s*=\s*([\d.]+)', r'initial\s+temperature\s+(?:of\s+|=\s*)?([\d.]+)')
    P2 = _n(t, r'P[₂2]\s*=\s*([\d.]+)', r'final\s+pressure\s+(?:of\s+|=\s*)?([\d.]+)')
    T2 = _n(t, r'T[₂2]\s*=\s*([\d.]+)', r'final\s+temperature\s+(?:of\s+|=\s*)?([\d.]+)')

    if P1 and T1 and T2 and not P2:
        P2_calc = P1 * T2 / T1
        return True, f"Gay-Lussac's Law: P₁/T₁ = P₂/T₂\nP₂ = {P1}×{T2}/{T1} = {P2_calc:.4g}"
    if P1 and T1 and P2 and not T2:
        T2_calc = T1 * P2 / P1
        return True, f"Gay-Lussac's Law: P₁/T₁ = P₂/T₂\nT₂ = {T1}×{P2}/{P1} = {T2_calc:.4g} K"
    return False, "Provide any three of P₁, T₁, P₂, T₂ for Gay-Lussac's Law."


def _solve_ideal_gas(text: str) -> tuple[bool, str]:
    t = text.lower()
    P_ = _n(t,
        r'pressure\s+(?:of\s+|=\s*)?([\d.]+)\s*(?:Pa|kPa|atm)',
        r'P\s*=\s*([\d.]+)',
        r'([\d.]+)\s*(?:Pa|atm|kPa)\b',
    )
    V_ = _n(t,
        r'volume\s+(?:of\s+|=\s*)?([\d.]+)\s*(?:m3|m³|L)',
        r'V\s*=\s*([\d.]+)',
        r'([\d.]+)\s*(?:m[³3]|L)\b',
    )
    n_ = _n(t,
        r'n\s*=\s*([\d.]+)',
        r'([\d.]+)\s*mol(?:es?)?\b',
    )
    T_ = _n(t,
        r'temperature\s+(?:of\s+|=\s*)?([\d.]+)',
        r'T\s*=\s*([\d.]+)',
        r'([\d.]+)\s*[Kk](?:elvin)?\b',
    )
    # Celsius → Kelvin
    if T_ and T_ < 400 and re.search(r'°C|celsius', text, re.I):
        T_ += 273.15

    # Convert units
    # kPa → Pa
    kpa_m = re.search(r'([\d.]+)\s*kPa\b', t)
    if kpa_m and P_ == float(kpa_m.group(1)):
        P_ *= 1000
    # atm → Pa
    atm_m = re.search(r'([\d.]+)\s*atm\b', t)
    if atm_m and P_ == float(atm_m.group(1)):
        P_ = float(atm_m.group(1)) * 101325
    # L → m³
    L_m = re.search(r'([\d.]+)\s*L\b', t)
    if L_m and V_ == float(L_m.group(1)):
        V_ = float(L_m.group(1)) * 1e-3

    R = R_gas
    knowns = [x for x in [P_, V_, n_, T_] if x is not None]

    if len(knowns) == 3:
        if P_ is None:
            P_calc = n_ * R * T_ / V_
            return True, f"PV=nRT → P = nRT/V = {n_}×{R}×{T_}/{V_} = {P_calc:.4g} Pa"
        if V_ is None:
            V_calc = n_ * R * T_ / P_
            return True, f"PV=nRT → V = nRT/P = {n_}×{R}×{T_}/{P_} = {V_calc:.4g} m³"
        if n_ is None:
            n_calc = P_ * V_ / (R * T_)
            return True, f"PV=nRT → n = PV/RT = {P_}×{V_}/({R}×{T_}) = {n_calc:.4g} mol"
        if T_ is None:
            T_calc = P_ * V_ / (n_ * R)
            return True, f"PV=nRT → T = PV/nR = {P_}×{V_}/({n_}×{R}) = {T_calc:.4g} K"

    return False, ("PV = nRT (R = 8.314 J/mol·K). Provide any three of:\n"
                   "  P (Pa), V (m³), n (mol), T (K)")


def _solve_calorimetry(text: str) -> tuple[bool, str]:
    t = text.lower()
    m_ = _n(t,
        r'm\s*=\s*([\d.]+)',
        r'mass\s+(?:of\s+|=\s*)?([\d.]+)\s*(?:g|kg)',
        r'([\d.]+)\s*g\b',
    )
    c_ = _n(t,
        r'c\s*=\s*([\d.]+)',
        r'specific\s+heat\s+(?:capacity\s+)?(?:of\s+|=\s*)?([\d.]+)',
        r'([\d.]+)\s*J/\(?g',
    )
    # ΔT patterns — search original text (case-sensitive Unicode) and lowercased
    dT = _n(text,
        r'[ΔδΔ]T\s*=\s*([\d.]+)',
        r'delta\s*T\s*=\s*([\d.]+)',
    )
    if dT is None:
        dT = _n(t, r'temperature\s+(?:change|rise|drop|increase|decrease)\s+(?:of\s+|=\s*)?([\d.]+)')
    if dT is None:
        # Last two numbers after commas pattern, e.g. "specific heat 4.18, ΔT 20"
        m_dt = re.search(r'[ΔδΔ]T[,\s]+(\d+(?:\.\d+)?)', text, re.I)
        if m_dt:
            dT = float(m_dt.group(1))
    if dT is None:
        rng = re.search(r'from\s+([\d.]+)\s*(?:°C|K)?\s+to\s+([\d.]+)', t)
        if rng:
            dT = abs(float(rng.group(2)) - float(rng.group(1)))
    q_ = _n(t, r'q\s*=\s*([\d.]+)', r'heat\s+(?:of\s+|=\s*)?([\d.]+)\s*(?:J|kJ)')

    if m_ and c_ and dT:
        q_calc = m_ * c_ * dT
        return True, f"q = mcΔT = {m_} × {c_} × {dT} = {q_calc:.4g} J"
    if q_ and c_ and dT:
        m_calc = q_ / (c_ * dT)
        return True, f"m = q/(cΔT) = {q_}/({c_}×{dT}) = {m_calc:.4g} g"
    if q_ and m_ and dT:
        c_calc = q_ / (m_ * dT)
        return True, f"c = q/(mΔT) = {q_}/({m_}×{dT}) = {c_calc:.4g} J/(g·°C)"
    if q_ and m_ and c_:
        dT_calc = q_ / (m_ * c_)
        return True, f"ΔT = q/(mc) = {q_}/({m_}×{c_}) = {dT_calc:.4g} °C"
    return False, "Provide mass (g), specific heat (J/g·°C), and ΔT, or any three of q, m, c, ΔT."


def _solve_percent_composition(text: str) -> tuple[bool, str]:
    formula = _extract_formula_from_text(text)
    # Find the element of interest
    el_m = re.search(r'percent(?:age)?\s+(?:composition\s+)?(?:of\s+|by\s+mass\s+of\s+)?([A-Z][a-z]?)\b', text, re.I)
    target_el = el_m.group(1).capitalize() if el_m else None

    if not formula:
        return False, "Could not find a chemical formula in the question."

    ok, M, breakdown = calc_molar_mass(formula)
    if not ok:
        return False, breakdown

    counts = _parse_formula(formula)
    results = [breakdown]
    results.append(f"\nPercent composition of {formula} (M = {M:.4f} g/mol):")
    for el, cnt in counts.items():
        el_mass = ATOMIC_MASS.get(el, 0) * cnt
        pct = el_mass / M * 100
        results.append(f"  {el}: ({el_mass:.4g}/{M:.4f}) × 100 = {pct:.4f}%")

    return True, "\n".join(results)


def _solve_half_life(text: str) -> tuple[bool, str]:
    t = text.lower()

    # ── k → t½  (first-order: t½ = 0.693/k) ──────────────────────────────────
    k_m = re.search(r'\bk\s*=\s*([\d.e\+\-]+)\s*(?:s[-⁻]1|s\^-?1|per\s+s|s-1)', text, re.I)
    if k_m and re.search(r'half.?life|t[½_]', text, re.I) and re.search(r'first.?order', text, re.I):
        k_val = float(k_m.group(1))
        t_half_calc = 0.693147 / k_val
        return True, (
            f"First-order half-life: t₁/₂ = ln2 / k = 0.6931 / k\n"
            f"  k = {k_val:.4e} s⁻¹\n"
            f"  t₁/₂ = 0.6931 / {k_val:.4e} = {t_half_calc:.4g} s"
        )

    t_half = _n(t,
        r'half.?life\s+(?:of\s+|=\s*)?([\d.]+)',
        r't[½_]?\s*=\s*([\d.]+)',
    )
    t_elapsed = _n(t,
        r'(?:after|in|elapsed)\s+([\d.]+)\s*(?:s|days?|years?|hours?|min)',
        r'time\s+(?:of\s+|=\s*)?([\d.]+)',
    )
    N0 = _n(t,
        r'N[₀0]\s*=\s*([\d.]+)',
        r'initial\s+(?:amount|quantity|mass)\s+(?:of\s+|=\s*)?([\d.]+)',
    )

    if t_half and t_elapsed:
        ratio = t_elapsed / t_half
        fraction = (0.5)**ratio
        results = [f"t₁/₂ = {t_half},  t = {t_elapsed}"]
        results.append(f"Number of half-lives = t/t₁/₂ = {ratio:.4g}")
        results.append(f"Fraction remaining = (½)^{ratio:.4g} = {fraction:.4g}")
        if N0:
            N_remaining = N0 * fraction
            results.append(f"N = N₀ × (½)^n = {N0} × {fraction:.4g} = {N_remaining:.4g}")
        return True, "\n".join(results)
    return False, ("Provide half-life (t₁/₂) and elapsed time (t) to calculate decay.\n"
                   "Optionally add initial amount (N₀) to find remaining amount.\n"
                   "For first-order: provide rate constant k (s⁻¹) to find t₁/₂.")


def _solve_electrochemistry(text: str) -> tuple[bool, str]:
    t = text.lower()
    E_ = _n(t, r'E°?\s*=\s*([\d.]+)', r'cell\s+potential\s+(?:of\s+|=\s*)?([\d.]+)')
    n_ = _n(t, r'\bn\s*=\s*(\d+)', r'(\d+)\s+electrons?')
    dG = _n(t, r'ΔG\s*=\s*(-?[\d.]+)', r'delta\s*G\s*=\s*(-?[\d.]+)')

    results = []
    if E_ is not None and n_ is not None:
        dG_calc = -n_ * F_const * E_
        results.append(f"ΔG = −nFE = −{n_} × {F_const:.4g} × {E_} = {dG_calc:.4g} J/mol")
        if dG_calc < 0:
            results.append("ΔG < 0 → Reaction is spontaneous (favours products).")
        else:
            results.append("ΔG > 0 → Reaction is non-spontaneous.")
    if dG is not None and n_ is not None:
        E_calc = -dG / (n_ * F_const)
        results.append(f"E = −ΔG/(nF) = −{dG}/({n_}×{F_const:.4g}) = {E_calc:.4g} V")
    if results:
        return True, "\n".join(results)
    return False, "Provide cell potential E° (V) and number of electrons n for ΔG = −nFE."


def _solve_combined_gas(text: str) -> tuple[bool, str]:
    t = text.lower()
    P1 = _n(t, r'P[₁1]\s*=\s*([\d.]+)', r'initial\s+pressure\s+(?:of\s+|=\s*)?([\d.]+)')
    V1 = _n(t, r'V[₁1]\s*=\s*([\d.]+)', r'initial\s+volume\s+(?:of\s+|=\s*)?([\d.]+)')
    T1 = _n(t, r'T[₁1]\s*=\s*([\d.]+)', r'initial\s+temperature\s+(?:of\s+|=\s*)?([\d.]+)')
    P2 = _n(t, r'P[₂2]\s*=\s*([\d.]+)', r'final\s+pressure\s+(?:of\s+|=\s*)?([\d.]+)')
    V2 = _n(t, r'V[₂2]\s*=\s*([\d.]+)', r'final\s+volume\s+(?:of\s+|=\s*)?([\d.]+)')
    T2 = _n(t, r'T[₂2]\s*=\s*([\d.]+)', r'final\s+temperature\s+(?:of\s+|=\s*)?([\d.]+)')

    if T1 and T1 < 400 and re.search(r'celsius|°C|degree\s*C', text, re.I):
        T1 += 273.15
    if T2 and T2 < 400 and re.search(r'celsius|°C|degree\s*C', text, re.I):
        T2 += 273.15

    known = {k: v for k, v in {'P1': P1, 'V1': V1, 'T1': T1, 'P2': P2, 'V2': V2, 'T2': T2}.items() if v is not None}
    if len(known) == 5:
        if P2 is None and P1 and V1 and T1 and V2 and T2:
            P2_calc = P1 * V1 * T2 / (T1 * V2)
            return True, f"Combined Gas Law: P₁V₁/T₁ = P₂V₂/T₂\nP₂ = P₁V₁T₂/(T₁V₂) = {P1}×{V1}×{T2}/({T1}×{V2}) = {P2_calc:.4g}"
        if V2 is None and P1 and V1 and T1 and P2 and T2:
            V2_calc = P1 * V1 * T2 / (T1 * P2)
            return True, f"Combined Gas Law: P₁V₁/T₁ = P₂V₂/T₂\nV₂ = P₁V₁T₂/(T₁P₂) = {P1}×{V1}×{T2}/({T1}×{P2}) = {V2_calc:.4g}"
        if T2 is None and P1 and V1 and T1 and P2 and V2:
            T2_calc = T1 * P2 * V2 / (P1 * V1)
            return True, f"Combined Gas Law: P₁V₁/T₁ = P₂V₂/T₂\nT₂ = T₁P₂V₂/(P₁V₁) = {T1}×{P2}×{V2}/({P1}×{V1}) = {T2_calc:.4g} K"
    return False, ("Combined Gas Law: P₁V₁/T₁ = P₂V₂/T₂\n"
                   "Provide any five of P₁, V₁, T₁, P₂, V₂, T₂ (temperatures in Kelvin).\n"
                   "Example: 'P1=1atm, V1=2L, T1=300K, P2=2atm, T2=400K, find V2'")


def _solve_stoichiometry(text: str) -> tuple[bool, str]:
    return False, (
        "Stoichiometry requires a balanced equation. Please provide:\n"
        "  1. The balanced chemical equation\n"
        "  2. The given mass or moles of a reactant/product\n"
        "  3. The substance you want to find\n"
        "Example: '2H₂ + O₂ → 2H₂O; given 4g H₂, find mass of H₂O'"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Public interface
# ─────────────────────────────────────────────────────────────────────────────

def _solve_ideal_gas_thermo(text: str) -> tuple[bool, str]:
    """
    Ideal gas thermodynamics solver.

    Supported queries
    -----------------
    * Find final temperature and ΔU when q joules are added to n mol
      of a monatomic/diatomic ideal gas at constant pressure or volume.
    * Cv, Cp values for mono/diatomic gases.

    Atomicity table
    ---------------
    Monatomic (Ar, He, Ne, Kr, Xe, Rn)  → Cv = 3R/2,  Cp = 5R/2
    Diatomic  (H₂, N₂, O₂, F₂, Cl₂,
               CO, NO, HCl)             → Cv = 5R/2,  Cp = 7R/2
    Default (unknown)                   → Cv = 3R/2 (monatomic fallback)
    """
    flags = re.IGNORECASE
    t = text

    # ── 1. Identify R value (allow user-supplied; default NIST) ───────────────
    r_match = re.search(r'\bR\s*=\s*([\d.]+)', t, flags)
    R = float(r_match.group(1)) if r_match else 8.314  # J·mol⁻¹·K⁻¹

    # ── 2. Atomicity ─────────────────────────────────────────────────────────
    MONATOMIC = r'\b(?:argon|Ar|helium|He|neon|Ne|krypton|Kr|xenon|Xe|radon|Rn)\b'
    DIATOMIC  = r'\b(?:hydrogen|H2|nitrogen|N2|oxygen|O2|fluorine|F2|chlorine|Cl2'  \
                r'|carbon\s+monoxide|CO|nitric\s+oxide|NO|HCl|HBr)\b'

    if re.search(MONATOMIC, t, flags) or re.search(r'\bmonatomic\b', t, flags):
        atomicity   = 'monatomic'
        Cv_factor   = 1.5   # 3/2
        Cp_factor   = 2.5   # 5/2
    elif re.search(DIATOMIC, t, flags) or re.search(r'\bdiatomic\b', t, flags):
        atomicity   = 'diatomic'
        Cv_factor   = 2.5   # 5/2
        Cp_factor   = 3.5   # 7/2
    else:
        atomicity   = 'monatomic (assumed)'
        Cv_factor   = 1.5
        Cp_factor   = 2.5

    Cv = Cv_factor * R   # J·mol⁻¹·K⁻¹
    Cp = Cp_factor * R

    # ── 3. Extract q (heat added, J) ─────────────────────────────────────────
    q = _n(t.lower(),
        r'([\d.]+)\s*(?:j|joules?)\s+(?:of\s+)?(?:energy|heat)',
        r'([\d.]+)\s*(?:j|joules?)',
        r'q\s*=\s*([\d.]+)',
    )
    if q is None:
        return False, "Could not extract the heat energy (J) from the question."

    # ── 4. Extract n (moles) ─────────────────────────────────────────────────
    n = _n(t.lower(),
        r'([\d.]+)\s*mol(?:es?)?\b',
        r'n\s*=\s*([\d.]+)',
    )
    if n is None:
        return False, "Could not extract the number of moles from the question."

    # ── 5. Extract initial temperature (K) ───────────────────────────────────
    T_i = _n(t.lower(),
        r'([\d.]+)\s*k\b',
        r'at\s+([\d.]+)',
        r'T(?:_?i(?:nitial)?)?\s*=\s*([\d.]+)',
    )
    if T_i is None:
        T_i = 298.0  # assume room temperature if not given

    # ── 6. Determine process (constant pressure vs constant volume) ───────────
    if re.search(r'constant\s+volume|isochoric', t, flags):
        process = 'constant volume'
        dT = q / (n * Cv)
        dU = q  # ΔU = q at constant volume
    else:
        # Default: constant pressure (when atm or pressure given, or unspecified)
        process = 'constant pressure'
        dT = q / (n * Cp)
        dU = n * Cv * dT  # ΔU = nCvΔT always

    T_f = T_i + dT

    lines = [
        f"Ideal Gas Thermodynamics  [{atomicity}, {process}]",
        f"  R  = {R} J·mol⁻¹·K⁻¹",
        f"  Cv = ({Cv_factor} × R) = {Cv:.4f} J·mol⁻¹·K⁻¹",
        f"  Cp = ({Cp_factor} × R) = {Cp:.4f} J·mol⁻¹·K⁻¹",
        "",
        f"  q  = {q} J   |   n = {n} mol   |   T_i = {T_i} K",
        "",
    ]

    if process == 'constant pressure':
        lines += [
            f"  At constant pressure:",
            f"    ΔT = q / (n·Cp) = {q} / ({n} × {Cp:.4f})",
            f"       = {dT:.4f} K",
            f"    T_final = T_i + ΔT = {T_i} + {dT:.4f} = {T_f:.4f} K",
            "",
            f"    ΔU = n·Cv·ΔT = {n} × {Cv:.4f} × {dT:.4f}",
            f"       = {dU:.4f} J",
        ]
    else:
        lines += [
            f"  At constant volume:",
            f"    ΔT = q / (n·Cv) = {q} / ({n} × {Cv:.4f})",
            f"       = {dT:.4f} K",
            f"    T_final = T_i + ΔT = {T_i} + {dT:.4f} = {T_f:.4f} K",
            "",
            f"    ΔU = q = {dU:.4f} J  (all heat goes to internal energy at const V)",
        ]

    lines += [
        "",
        f"  ✓  Final temperature : {T_f:.2f} K",
        f"  ✓  ΔU (internal energy change) : {dU:.2f} J",
    ]

    return True, "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# New solvers (T001 gap-fill)
# ─────────────────────────────────────────────────────────────────────────────

def _solve_empirical_formula(text: str) -> tuple[bool, str]:
    """Determine empirical formula from % composition."""
    t = text.lower()
    # Extract element-percentage pairs e.g. "C=40%, H=6.67%, O=53.33%"
    pairs = re.findall(r'\b([A-Z][a-z]?)\s*[=:]\s*([\d.]+)\s*%', text)
    if not pairs:
        # Also try "carbon 40%, hydrogen 6.67%"
        name_map = {
            'carbon': 'C', 'hydrogen': 'H', 'oxygen': 'O', 'nitrogen': 'N',
            'sulfur': 'S', 'chlorine': 'Cl', 'sodium': 'Na', 'potassium': 'K',
        }
        for name, sym in name_map.items():
            m = re.search(rf'{name}[^%]*([\d.]+)\s*%', t)
            if m:
                pairs.append((sym, m.group(1)))
    if len(pairs) < 2:
        return False, "Provide % composition as 'C=40%, H=6.67%, O=53.33%'."

    # Moles = % / atomic_mass
    mole_ratios = []
    lines = ["Empirical Formula Calculation:"]
    for sym, pct_str in pairs:
        pct = float(pct_str)
        AM  = ATOMIC_MASS.get(sym.capitalize())
        if AM is None:
            return False, f"Unknown element symbol '{sym}'."
        ratio = pct / AM
        mole_ratios.append((sym.capitalize(), pct, AM, ratio))
        lines.append(f"  {sym}: {pct}% / {AM:.4g} g/mol = {ratio:.4f} mol")

    # Divide by smallest
    smallest = min(r for _, _, _, r in mole_ratios)
    lines.append(f"\n  Divide by smallest ({smallest:.4f}):")
    norm = []
    for sym, pct, AM, ratio in mole_ratios:
        n = ratio / smallest
        lines.append(f"  {sym}: {n:.3f}")
        norm.append((sym, n))

    # Round to nearest integer (multiply up if needed)
    def _best_int(x: float) -> int:
        for mult in range(1, 9):
            val = x * mult
            if abs(val - round(val)) < 0.1:
                return round(val) * mult  # not right, should be round(val)
        return round(x)

    # Smarter: find multiplier such that all n*mult ≈ integer
    best_mult = 1
    for mult in range(1, 9):
        if all(abs(n * mult - round(n * mult)) < 0.15 for _, n in norm):
            best_mult = mult
            break

    formula = ""
    for sym, n in norm:
        count = round(n * best_mult)
        formula += sym + (str(count) if count > 1 else "")
    lines.append(f"\n  Empirical formula: {formula}")
    return True, "\n".join(lines)


def _solve_oxidation_state(text: str) -> tuple[bool, str]:
    """Determine oxidation state of a target element in a compound."""
    # Known fixed oxidation states
    FIXED = {
        'H': +1, 'O': -2, 'F': -1,
        'Na': +1, 'K': +1, 'Li': +1, 'Rb': +1, 'Cs': +1,
        'Mg': +2, 'Ca': +2, 'Ba': +2, 'Sr': +2,
        'Al': +3, 'Ag': +1, 'Zn': +2,
        'Cl': -1, 'Br': -1, 'I': -1,
    }
    # Find compound
    compound_m = re.search(r'(?:in|of)\s+([A-Z][A-Za-z0-9()]+)', text)
    if not compound_m:
        return False, "Specify compound, e.g. 'oxidation state of Mn in KMnO4'."
    compound = compound_m.group(1)
    counts = _parse_formula(compound)
    if not counts:
        return False, f"Cannot parse compound formula '{compound}'."

    # Determine overall charge (assumed 0 unless ionic, e.g. SO4^2-)
    overall_charge = 0
    charge_m = re.search(r'\^?([+-]?\d+)[+-]', compound)
    if charge_m:
        overall_charge = int(charge_m.group(1))

    # Find unknown element
    target_m = re.search(r'(?:of|state\s+of)\s+([A-Z][a-z]?)\s+in', text, re.I)
    if not target_m:
        # Try to infer: the element NOT in FIXED
        unknowns = [el for el in counts if el not in FIXED]
        if len(unknowns) != 1:
            return False, "Specify which element's oxidation state to find, e.g. 'oxidation state of Mn in KMnO4'."
        target = unknowns[0]
    else:
        target = target_m.group(1).capitalize()
        if target not in counts:
            return False, f"Element {target} not found in {compound}."

    # Sum of all other elements × count × fixed OS
    sum_known = 0
    lines = [f"Oxidation state of {target} in {compound}:"]
    lines.append(f"  Total charge of compound = {overall_charge}")
    for el, cnt in counts.items():
        if el == target:
            continue
        os = FIXED.get(el)
        if os is None:
            return False, f"Cannot determine fixed oxidation state of {el} — too complex."
        contrib = os * cnt
        sum_known += contrib
        lines.append(f"  {el}: OS = {os:+d}, count = {cnt}, contribution = {contrib:+d}")
    target_count = counts[target]
    target_os = (overall_charge - sum_known) / target_count
    lines.append(f"\n  {target}: {target_count} × OS({target}) + {sum_known} = {overall_charge}")
    lines.append(f"  OS({target}) = ({overall_charge} − {sum_known}) / {target_count} = {target_os:+.4g}")
    return True, "\n".join(lines)


def _solve_normality(text: str) -> tuple[bool, str]:
    """Normality = Molarity × n-factor."""
    t = text.lower()
    M_ = _n(t,
        r'([\d.]+)\s*M\b',
        r'molarity\s+(?:of\s+|=\s*)?([\d.]+)',
        r'concentration\s+(?:of\s+|=\s*)?([\d.]+)',
    )
    # n-factor table for common acids/bases
    N_FACTORS = {
        'h2so4': 2, 'h3po4': 3, 'hcl': 1, 'hno3': 1,
        'naoh': 1, 'ca(oh)2': 2, 'koh': 1, 'h2c2o4': 2,
        'na2co3': 2, 'k2cr2o7': 6, 'kmno4': 5,
    }
    n_factor = 1
    n_name   = "unknown (assumed 1)"
    for formula, nf in N_FACTORS.items():
        if formula in t:
            n_factor = nf
            n_name   = formula.upper()
            break
    # Also accept explicit "n-factor = X"
    nf_m = re.search(r'n.?factor\s*=\s*(\d+)', t)
    if nf_m:
        n_factor = int(nf_m.group(1))
        n_name   = f"given ({n_factor})"

    if M_ is not None:
        N_ = M_ * n_factor
        return True, (
            f"Normality: N = M × n-factor\n"
            f"  M = {M_} mol/L,  n-factor = {n_factor} ({n_name})\n"
            f"  N = {M_} × {n_factor} = {N_:.4g} N"
        )
    return False, "Provide molarity (M) and compound name (e.g. H2SO4) or explicit n-factor."


def _solve_moles_stp(text: str) -> tuple[bool, str]:
    """Moles of gas at STP: n = V / 22.4 L/mol."""
    t = text.lower()
    V_ = _n(t,
        r'([\d.]+)\s*L\b',
        r'([\d.]+)\s*litres?\b',
        r'volume\s+(?:of\s+|=\s*)?([\d.]+)',
        r'([\d.]+)\s*(?:ml|millilitres?)',
    )
    # Convert mL → L
    ml_m = re.search(r'([\d.]+)\s*m[Ll]\b', t)
    if ml_m and V_ == float(ml_m.group(1)):
        V_ = float(ml_m.group(1)) / 1000

    if V_ is not None:
        n = V_ / 22.4
        return True, (
            f"Moles at STP: n = V / 22.4 L·mol⁻¹\n"
            f"  V = {V_} L\n"
            f"  n = {V_} / 22.4 = {n:.4g} mol"
        )
    return False, "Provide volume in litres (L) to find moles at STP."


def _solve_balance_equation(text: str) -> tuple[bool, str]:
    """Balance simple common equations by look-up or inspection."""
    t = text.lower()
    KNOWN = {
        'h2 + o2':    ('2H₂ + O₂ → 2H₂O',   'Combustion of hydrogen'),
        'ch4 + o2':   ('CH₄ + 2O₂ → CO₂ + 2H₂O', 'Combustion of methane'),
        'c + o2':     ('C + O₂ → CO₂',        'Combustion of carbon'),
        'n2 + h2':    ('N₂ + 3H₂ → 2NH₃',    'Haber process'),
        'na + cl2':   ('2Na + Cl₂ → 2NaCl',   'Formation of NaCl'),
        'na + h2o':   ('2Na + 2H₂O → 2NaOH + H₂', 'Sodium with water'),
        'fe + o2':    ('4Fe + 3O₂ → 2Fe₂O₃',  'Rusting of iron'),
        'al + o2':    ('4Al + 3O₂ → 2Al₂O₃',  'Aluminium oxide'),
        'ca + h2o':   ('Ca + 2H₂O → Ca(OH)₂ + H₂', 'Calcium with water'),
        'hcl + naoh': ('HCl + NaOH → NaCl + H₂O', 'Acid-base neutralisation'),
        'h2so4 + naoh': ('H₂SO₄ + 2NaOH → Na₂SO₄ + 2H₂O', 'Acid-base neutralisation'),
        'c3h8 + o2':  ('C₃H₈ + 5O₂ → 3CO₂ + 4H₂O', 'Combustion of propane'),
        'c2h5oh + o2':('C₂H₅OH + 3O₂ → 2CO₂ + 3H₂O', 'Combustion of ethanol'),
        'kmno4 + hcl':('2KMnO₄ + 16HCl → 2KCl + 2MnCl₂ + 5Cl₂ + 8H₂O', 'KMnO₄ with HCl'),
    }
    # Normalise input to find key
    reactants = re.split(r'→|->', t)
    if reactants:
        lhs = reactants[0].strip()
        lhs_clean = re.sub(r'\s+', ' ', re.sub(r'[^\w\s]', '', lhs)).lower()
        for key, (balanced, name) in KNOWN.items():
            # Check if key components appear
            parts = key.split(' + ')
            if all(p.replace(' ', '') in lhs_clean.replace(' ', '') for p in parts):
                return True, (
                    f"Balanced equation ({name}):\n"
                    f"  {balanced}\n\n"
                    f"Check: count atoms on each side to verify coefficients."
                )
    return False, (
        "I can balance common equations by inspection.\n"
        "Try: H2 + O2, CH4 + O2, N2 + H2, Na + Cl2, Na + H2O, Fe + O2.\n"
        "For complex equations, use the algebraic method."
    )


def _solve_stoichiometry_mass(text: str) -> tuple[bool, str]:
    """How many grams of product from given moles of reactant."""
    t = text.lower()
    # Extract moles of reactant
    n_reactant = _n(t,
        r'([\d.]+)\s*mol\b',
        r'n\s*=\s*([\d.]+)',
    )
    # Find product formula in text
    prod_m = re.search(r'(?:grams?\s+of\s+|produce[sd]?\s+)([A-Z][a-zA-Z0-9()]+)', text)
    reactant_m = re.search(r'([\d.]+)\s*mol\s+([A-Z][a-z]?[a-zA-Z0-9()]*)', text)

    if n_reactant is not None and prod_m:
        product_formula = prod_m.group(1)
        ok, M_prod, _ = calc_molar_mass(product_formula)
        if ok:
            # Assume 1:1 molar ratio for simple single-displacement reactions
            # Try to detect ratio from "2 mol Na → 1 mol NaCl" patterns
            ratio = 1.0
            ratio_m = re.search(r'(\d+)\s*mol\s+\w+.*?(\d+)\s*mol\s+\w+', t)
            if ratio_m:
                ratio = float(ratio_m.group(2)) / float(ratio_m.group(1))
            n_product = n_reactant * ratio
            mass = n_product * M_prod
            return True, (
                f"Stoichiometry:\n"
                f"  n(reactant) = {n_reactant} mol\n"
                f"  Molar ratio (product:reactant) ≈ {ratio:.2g}\n"
                f"  n(product) = {n_product:.4g} mol\n"
                f"  M({product_formula}) = {M_prod:.4f} g/mol\n"
                f"  mass = n × M = {n_product:.4g} × {M_prod:.4f} = {mass:.4g} g"
            )
    return False, "Provide moles of reactant and product formula. E.g. '2 mol Na → NaCl'."


def _solve_dissociation_degree(text: str) -> tuple[bool, str]:
    """Degree of dissociation α from Ka and concentration."""
    t = text.lower()
    Ka = _extract_ka(text)
    C_ = _n(t,
        r'([\d.]+)\s*M\b',
        r'concentration\s+(?:of\s+|=\s*)?([\d.]+)',
        r'C\s*=\s*([\d.]+)',
    )
    if Ka is not None and C_ is not None:
        # Ka ≈ Cα² / (1-α) ≈ Cα² for small α
        # Quadratic: Cα² + Kaα - Ka = 0
        A_ = C_
        B_ = Ka
        C_coef = -Ka
        disc = B_**2 - 4 * A_ * C_coef
        alpha = (-B_ + math.sqrt(disc)) / (2 * A_)
        h_conc = alpha * C_
        pH = -math.log10(h_conc)
        return True, (
            f"Degree of dissociation: HA ⇌ H⁺ + A⁻\n"
            f"  Ka = {Ka:.4e},  C = {C_} M\n"
            f"  Quadratic: Cα² + Kaα − Ka = 0\n"
            f"  α = {alpha:.4f}  ({alpha*100:.4f}%)\n"
            f"  [H⁺] = αC = {alpha:.4f} × {C_} = {h_conc:.4e} M\n"
            f"  pH = −log[H⁺] = {pH:.4f}"
        )
    return False, "Provide Ka and concentration C (M) to calculate degree of dissociation."


_CHEM_SOLVERS = {
    'molar_mass':          _solve_molar_mass,
    'moles':               _solve_moles,
    'molarity':            _solve_molarity,
    'dilution':            _solve_dilution,
    'ph':                  _solve_ph,
    'weak_acid':           _solve_weak_acid,
    'dissociation_degree': _solve_dissociation_degree,
    'boyle':               _solve_boyle,
    'charles':             _solve_charles,
    'gay_lussac':          _solve_gay_lussac,
    'ideal_gas':           _solve_ideal_gas,
    'ideal_gas_thermo':    _solve_ideal_gas_thermo,
    'combined_gas':        _solve_combined_gas,
    'calorimetry':         _solve_calorimetry,
    'percent_composition': _solve_percent_composition,
    'half_life':           _solve_half_life,
    'electrochemistry':    _solve_electrochemistry,
    'stoichiometry':       _solve_stoichiometry,
    'empirical_formula':   _solve_empirical_formula,
    'oxidation_state':     _solve_oxidation_state,
    'normality':           _solve_normality,
    'moles_stp':           _solve_moles_stp,
    'balance_equation':    _solve_balance_equation,
    'stoichiometry_mass':  _solve_stoichiometry_mass,
}


class ChemistryEngine:
    """Deterministic chemistry formula solver."""

    # Expose molar mass for external use
    @staticmethod
    def molar_mass(formula: str) -> tuple[bool, float, str]:
        return calc_molar_mass(formula)

    def solve(self, text: str) -> tuple[bool, str, str]:
        """
        Attempt to solve a chemistry problem from natural language.
        Returns (success, result_string, problem_type).
        """
        ctype  = _detect_chem_type(text)
        solver = _CHEM_SOLVERS.get(ctype)

        if solver is None:
            return False, "Could not identify the chemistry calculation to perform.", "unknown"

        try:
            success, result = solver(text)
            return success, result, ctype
        except Exception as exc:
            return False, f"Calculation error: {exc}", ctype

    def is_chemistry_question(self, text: str) -> bool:
        return _detect_chem_type(text) != 'unknown'
