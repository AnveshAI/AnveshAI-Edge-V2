"""
Hint Engine — three-level progressive hints for JEE problems.

Levels:
    1 — Conceptual hint  (what topic / principle applies)
    2 — Formula hint     (which equation to use)
    3 — First-step hint  (what to do first with the numbers)
"""

import re
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Hint rules: (intent, keyword_pattern) → (concept, formula, first_step)
# ─────────────────────────────────────────────────────────────────────────────
_RULES = [
    # Physics — Kinematics
    (r'velocity|acceleration|distance.*time|time.*distance|uniform\s+acceler',
     "Kinematics",
     "Use kinematic equations: v=u+at, s=ut+½at², v²=u²+2as",
     "Identify which three variables are known (u, v, a, s, t) and pick the matching equation"),
    # Projectile
    (r'projectile|range.*launch|angle.*throw|thrown.*angle',
     "Projectile motion (2-D kinematics)",
     "Horizontal: x = u·cosθ·t ; Vertical: y = u·sinθ·t − ½gt²\nRange R = u²sin2θ/g",
     "Decompose initial velocity into horizontal (u cosθ) and vertical (u sinθ) components"),
    # Energy
    (r'kinetic\s+energy|potential\s+energy|work.*done|energy.*conserv',
     "Work & Energy theorem",
     "KE = ½mv²,  PE = mgh,  W = F·d·cosθ,  W_net = ΔKE",
     "Identify all energy forms present and check if non-conservative forces do work"),
    # Momentum
    (r'momentum|impulse|collision|conserv.*momentum',
     "Conservation of Momentum",
     "p = mv,  J = FΔt = Δp,  p_initial = p_final (isolated system)",
     "Define the system; check if external forces are zero; then apply p_before = p_after"),
    # Gravitation
    (r'gravitational|escape\s+velocity|orbital|satellite',
     "Gravitation",
     "F = Gm₁m₂/r²,  g = GM/R²,  vₑ = √(2gR),  v_orb = √(GM/r)",
     "Identify masses, separation, and whether surface or orbital radius is given"),
    # SHM
    (r'simple\s+harmonic|oscillat|spring.*mass|pendulum',
     "Simple Harmonic Motion (SHM)",
     "x = A sin(ωt+φ),  ω = 2πf,  T = 2π√(m/k) (spring),  T = 2π√(L/g) (pendulum)",
     "Find the restoring force constant; then ω = √(k/m) gives period and frequency"),
    # Electricity
    (r"ohm'?s?\s*law|resistance|current.*voltage|voltage.*current",
     "Ohm's Law & Circuits",
     "V = IR,  P = IV = I²R = V²/R,  R_series = ΣRᵢ,  1/R_parallel = Σ(1/Rᵢ)",
     "Simplify the circuit by combining series/parallel resistors first"),
    # Coulomb / Electrostatics
    (r'coulomb|electric\s+force|charge.*force',
     "Coulomb's Law",
     "F = kq₁q₂/r²  (k = 8.987×10⁹ N·m²/C²)",
     "Convert charges to Coulombs (1 µC = 10⁻⁶ C), then plug in F = kq₁q₂/r²"),
    # Waves / Sound
    (r'\bwave\b|frequency|wavelength|sound\s+speed',
     "Wave mechanics",
     "v = fλ,  Beat frequency = |f₁−f₂|",
     "Identify which two of {v, f, λ} are given, then find the third"),
    # Optics
    (r"snell'?s?\s*law|refraction|lens|mirror\s+formula",
     "Optics",
     "n₁sinθ₁ = n₂sinθ₂ (Snell),  1/f = 1/v + 1/u (mirror),  1/f = 1/v − 1/u (lens)",
     "Determine sign convention (real-is-positive or Cartesian) before substituting"),
    # Thermodynamics
    (r'heat\s+engine|carnot|efficiency.*heat|thermal\s+efficiency',
     "Thermodynamics",
     "η_Carnot = 1 − T_cold/T_hot  ;  ΔU = Q − W",
     "Express temperatures in Kelvin (K = °C + 273), then apply the efficiency formula"),
    # Modern Physics
    (r'de\s+broglie|photoelectric|photon\s+energy|work\s+function',
     "Modern Physics (Quantum)",
     "E = hf = hc/λ  (h = 6.626×10⁻³⁴),  λ = h/(mv),  KE_max = hf − φ",
     "Find frequency f = v/λ or use E = hf; then apply the photoelectric condition"),
    # Half-life
    (r'half.?life|radioactive|decay\s+constant',
     "Radioactive Decay",
     "N = N₀(½)^(t/t½),  t½ = 0.693/λ",
     "Find the number of half-lives n = t/t½, then remaining fraction = (½)ⁿ"),
    # Chemistry — pH
    (r'\bpH\b|acid.*concentration|strong\s+acid|strong\s+base',
     "Acid-Base Chemistry",
     "Strong acid: pH = −log[H⁺]  ;  Weak acid: [H⁺] ≈ √(Ka·C)",
     "Determine if the acid is strong (fully dissociated) or weak (partial), then use the appropriate equation"),
    # Gas Laws
    (r"boyle'?s?|charles'?s?|ideal\s+gas|pv\s*=\s*nrt",
     "Gas Laws",
     "Boyle: P₁V₁=P₂V₂  ;  Charles: V₁/T₁=V₂/T₂  ;  PV=nRT  (R=0.082 L·atm/mol·K)",
     "Identify which variable is constant (T or P), then apply the matching gas law"),
    # Moles & Stoichiometry
    (r'moles?|molar\s+mass|stoichiometr',
     "Mole Concept",
     "n = m/M  ;  n = V/22.4 (STP)  ;  M = Σ(atomic masses × subscripts)",
     "Find molar mass of the compound, then use n = m/M to convert grams to moles"),
    # Electrochemistry
    (r'cell\s+potential|reduction\s+potential|nernst|faraday.*electrolysis',
     "Electrochemistry",
     "E_cell = E_cathode − E_anode  ;  ΔG° = −nFE°  ;  Nernst: E = E° − (RT/nF)lnQ",
     "Identify oxidation and reduction half-reactions, look up standard reduction potentials"),
    # Calculus — integration
    (r'integrat|∫|definite\s+integral',
     "Calculus — Integration",
     "∫xⁿ dx = xⁿ⁺¹/(n+1)+C  ;  ∫eˣ=eˣ  ;  ∫sin=−cos  ;  ∫1/x=ln|x|",
     "Apply the power rule for polynomials; for products consider integration by parts (∫u dv = uv − ∫v du)"),
    # Calculus — differentiation
    (r'differentiat|derivative|d/dx',
     "Calculus — Differentiation",
     "d/dx(xⁿ)=nxⁿ⁻¹  ;  product rule  ;  chain rule  ;  quotient rule",
     "Identify the outermost function and apply chain rule outward-in for composite functions"),
    # Probability
    (r'probability|random\s+variable|binomial\s+distribution|bayes',
     "Probability",
     "P(A) = n(A)/n(S)  ;  P(A∪B) = P(A)+P(B)−P(A∩B)  ;  Bayes theorem",
     "List all outcomes in the sample space, count favourable outcomes, then apply the formula"),
    # Vectors
    (r'vector|dot\s+product|cross\s+product|magnitude.*direction',
     "Vectors",
     "A·B = |A||B|cosθ  ;  |A×B| = |A||B|sinθ  ;  |A| = √(x²+y²+z²)",
     "Resolve all vectors into components (î, ĵ, k̂), then compute the required product component-wise"),
]


def get_hints(question: str, level: int = 1) -> str:
    """
    Return progressive hints for a problem.
    level 1 = concept only
    level 2 = concept + formula
    level 3 = concept + formula + first step
    """
    q_lower = question.lower()

    matched = None
    for pattern, concept, formula, first_step in _RULES:
        if re.search(pattern, q_lower, re.IGNORECASE):
            matched = (concept, formula, first_step)
            break

    if matched is None:
        concept    = "General problem-solving"
        formula    = "Identify the relevant physical law or mathematical principle"
        first_step = "List all given quantities with units, identify what is asked, then select the right equation"
    else:
        concept, formula, first_step = matched

    lines = [f"\n  Hint Level {level}/3 for your question:"]
    lines.append(f"\n  💡 Concept : {concept}")
    if level >= 2:
        lines.append(f"\n  📐 Formula : {formula}")
    if level >= 3:
        lines.append(f"\n  🔑 First step: {first_step}")
    lines.append("\n  (Type /hint2 or /hint3 for deeper hints)")
    return "\n".join(lines)
