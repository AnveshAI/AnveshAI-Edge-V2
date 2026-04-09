"""
Formula Sheet — JEE Advanced quick reference.
/formulas [physics|chemistry|math]
"""

from typing import Optional

FORMULAS = {
    "physics": {
        "Kinematics": [
            "v = u + at",
            "s = ut + ½at²",
            "v² = u² + 2as",
            "s_n = u + a(2n−1)/2  (nth second)",
            "Range  R = u²sin2θ/g",
            "H_max  = u²sin²θ/(2g)",
            "Time of flight T = 2u sinθ/g",
        ],
        "Newton's Laws & Friction": [
            "F = ma",
            "f_s ≤ μₛN  (static)",
            "f_k = μₖN  (kinetic)",
            "Impulse J = FΔt = Δp",
        ],
        "Work, Energy & Power": [
            "W = F·d·cosθ",
            "KE = ½mv²",
            "PE (gravity) = mgh",
            "PE (spring) = ½kx²",
            "Power P = W/t = Fv",
            "Work-Energy theorem: W_net = ΔKE",
        ],
        "Circular & Gravitation": [
            "F_c = mv²/r = mω²r",
            "F_g = Gm₁m₂/r²   (G = 6.674×10⁻¹¹)",
            "g = GM/R²",
            "Escape velocity vₑ = √(2gR) = √(2GM/R)",
            "Orbital speed v = √(GM/r)",
            "Gravitational PE = −Gm₁m₂/r",
        ],
        "Rotational Motion": [
            "τ = Iα = r × F",
            "L = Iω",
            "KE_rot = ½Iω²",
            "Solid sphere: I = 2/5 MR²",
            "Hollow sphere: I = 2/3 MR²",
            "Disk/cylinder: I = ½MR²",
            "Rod (centre): I = ML²/12",
            "Rod (end): I = ML²/3",
        ],
        "Oscillations (SHM)": [
            "x = A sin(ωt + φ)",
            "ω = 2π/T = 2πf",
            "Simple pendulum: T = 2π√(L/g)",
            "Spring-mass: T = 2π√(m/k)",
            "v_max = Aω  at equilibrium",
            "E_total = ½kA² = ½mω²A²",
        ],
        "Waves": [
            "v = fλ",
            "Intensity ∝ A²",
            "Standing wave: λₙ = 2L/n",
            "Beat frequency = |f₁ − f₂|",
        ],
        "Thermodynamics": [
            "Q = mcΔT  (calorimetry)",
            "ΔU = Q − W  (1st law)",
            "Isothermal: PV = const",
            "Adiabatic: PVᵞ = const  (γ = Cp/Cv)",
            "η_Carnot = 1 − T_cold/T_hot",
            "PV = nRT  (ideal gas, R = 8.314 J/mol·K)",
        ],
        "Electrostatics": [
            "F = kq₁q₂/r²  (k = 8.987×10⁹ N·m²/C²)",
            "E = kq/r²  (point charge)",
            "V = kq/r",
            "C = Q/V  (capacitance)",
            "Energy in capacitor = ½CV²",
            "Parallel plate: C = ε₀A/d",
        ],
        "Current Electricity": [
            "V = IR  (Ohm's law)",
            "P = IV = I²R = V²/R",
            "R_series = R₁ + R₂ + …",
            "1/R_parallel = 1/R₁ + 1/R₂ + …",
            "Kirchhoff KVL, KCL",
        ],
        "Magnetism": [
            "F = qvB sinθ  (Lorentz)",
            "F = BIL sinθ  (force on wire)",
            "B (solenoid) = μ₀nI",
            "B (wire) = μ₀I/(2πr)",
            "Magnetic flux Φ = BA cosθ",
            "EMF = −dΦ/dt  (Faraday)",
        ],
        "Modern Physics": [
            "E = hf = hc/λ  (h = 6.626×10⁻³⁴ J·s)",
            "E = mc²",
            "λ_de Broglie = h/(mv) = h/p",
            "KE_max = hf − φ  (photoelectric)",
            "rₙ = n²a₀  (Bohr, a₀ = 0.529 Å)",
            "Eₙ = −13.6/n² eV  (Hydrogen)",
            "t₁/₂ = ln2/λ = 0.693/λ",
        ],
        "Optics": [
            "n₁ sinθ₁ = n₂ sinθ₂  (Snell's law)",
            "Mirror: 1/f = 1/v + 1/u",
            "Lens: 1/f = 1/v − 1/u",
            "Lens maker: 1/f = (n−1)(1/R₁ − 1/R₂)",
            "Resolving power ∝ 1/λ",
        ],
    },
    "chemistry": {
        "Atomic Structure": [
            "E_n = −13.6/n² eV (H atom)",
            "rₙ = 0.529n² Å (Bohr radius)",
            "λ = h/(mv)  (de Broglie)",
            "Δx·Δp ≥ h/(4π)  (Heisenberg)",
        ],
        "Moles & Stoichiometry": [
            "n = m/M  (moles)",
            "n = N/Nₐ  (Nₐ = 6.022×10²³)",
            "n (STP) = V/22.4 L",
            "Molarity M = n/V(L)",
            "Normality N = M × n-factor",
            "% by mass = (solute/solution) × 100",
        ],
        "Gas Laws": [
            "Boyle:    P₁V₁ = P₂V₂  (const T)",
            "Charles:  V₁/T₁ = V₂/T₂  (const P)",
            "Gay-Lussac: P₁/T₁ = P₂/T₂  (const V)",
            "Ideal:    PV = nRT  (R = 0.082 L·atm/mol·K = 8.314 J/mol·K)",
            "Graham:   r₁/r₂ = √(M₂/M₁)",
        ],
        "Thermochemistry": [
            "ΔH = H_products − H_reactants",
            "q = mcΔT  (calorimetry)",
            "Hess's law: ΔH = ΣΔH_f(products) − ΣΔH_f(reactants)",
            "Bond energy: ΔH = Σ(bonds broken) − Σ(bonds formed)",
        ],
        "Chemical Equilibrium": [
            "Kc = [products]/[reactants]  (mol/L)",
            "Kp = Kc(RT)^Δn",
            "Le Chatelier's principle",
            "Q < K → forward  ;  Q > K → reverse",
        ],
        "Ionic Equilibrium (Acids/Bases)": [
            "pH = −log[H⁺]",
            "pOH = −log[OH⁻]",
            "pH + pOH = 14  (25°C)",
            "Weak acid: [H⁺] ≈ √(Ka·C)",
            "Weak base: [OH⁻] ≈ √(Kb·C)",
            "Buffer (Henderson-Hasselbalch): pH = pKa + log([A⁻]/[HA])",
            "Kw = Ka × Kb = 10⁻¹⁴  (25°C)",
        ],
        "Electrochemistry": [
            "E_cell = E_cathode − E_anode",
            "ΔG° = −nFE°  (F = 96485 C/mol)",
            "Nernst: E = E° − (RT/nF)lnQ",
            "Faraday: m = (M × I × t)/(n × F)",
        ],
        "Kinetics": [
            "Rate = k[A]^m[B]^n",
            "1st order: [A] = [A]₀e^(−kt)",
            "t₁/₂ (1st) = 0.693/k",
            "t₁/₂ (2nd) = 1/(k[A]₀)",
            "Arrhenius: k = Ae^(−Ea/RT)",
        ],
        "Coordination Chemistry": [
            "CFT: Crystal Field Splitting Δ",
            "Strong-field: low spin  ;  Weak-field: high spin",
            "EAN rule = 18-electron rule",
        ],
    },
    "math": {
        "Algebra": [
            "Quadratic: x = (−b ± √(b²−4ac))/(2a)",
            "Sum of roots = −b/a ; Product = c/a",
            "AM ≥ GM ≥ HM",
            "Binomial: (a+b)ⁿ = ΣC(n,k)aⁿ⁻ᵏbᵏ",
        ],
        "Sequences & Series": [
            "AP: aₙ = a + (n−1)d ; S = n(2a+(n−1)d)/2",
            "GP: aₙ = arⁿ⁻¹ ; S = a(1−rⁿ)/(1−r)",
            "S_∞ (GP, |r|<1) = a/(1−r)",
            "ΣnΣ = n(n+1)/2  ;  Σn² = n(n+1)(2n+1)/6",
        ],
        "Trigonometry": [
            "sin²θ + cos²θ = 1",
            "tan²θ + 1 = sec²θ",
            "1 + cot²θ = csc²θ",
            "sin(A±B) = sinA cosB ± cosA sinB",
            "cos(A±B) = cosA cosB ∓ sinA sinB",
            "Sine rule: a/sinA = b/sinB = c/sinC",
            "Cosine rule: a² = b² + c² − 2bc cosA",
        ],
        "Calculus — Differentiation": [
            "d/dx(xⁿ) = nxⁿ⁻¹",
            "d/dx(eˣ) = eˣ",
            "d/dx(ln x) = 1/x",
            "Product rule: (uv)' = u'v + uv'",
            "Chain rule: d/dx[f(g(x))] = f'(g)·g'",
            "L'Hôpital: lim f/g = lim f'/g'  (0/0 or ∞/∞)",
        ],
        "Calculus — Integration": [
            "∫xⁿ dx = xⁿ⁺¹/(n+1) + C",
            "∫eˣ dx = eˣ + C",
            "∫sin x dx = −cos x + C",
            "∫cos x dx = sin x + C",
            "∫(1/x) dx = ln|x| + C",
            "Integration by parts: ∫uv' = uv − ∫u'v",
        ],
        "Vectors & 3-D": [
            "Dot product: A·B = |A||B|cosθ",
            "Cross product: |A×B| = |A||B|sinθ",
            "Unit vector: â = A/|A|",
            "Section formula: P = (m·B + n·A)/(m+n)",
        ],
        "Probability": [
            "P(A) = n(A)/n(S)",
            "P(A∪B) = P(A)+P(B)−P(A∩B)",
            "Bayes: P(A|B) = P(B|A)P(A)/P(B)",
            "Binomial: P(X=k) = C(n,k)pᵏ(1−p)ⁿ⁻ᵏ",
            "Mean = np ; Variance = np(1−p)",
        ],
        "Matrices & Determinants": [
            "det[[a,b],[c,d]] = ad − bc",
            "A⁻¹ = adj(A)/det(A)",
            "Cayley-Hamilton: A satisfies its characteristic eq.",
            "Rank ≤ min(m, n)",
        ],
        "Complex Numbers": [
            "i² = −1  ;  i³ = −i  ;  i⁴ = 1",
            "|z| = √(a²+b²)",
            "arg(z) = arctan(b/a)",
            "De Moivre: (cosθ+i sinθ)ⁿ = cos(nθ)+i sin(nθ)",
        ],
    },
}


def get_formula_sheet(subject: Optional[str] = None) -> str:
    """Return formatted formula sheet for given subject or all subjects."""
    subjects = (
        [subject.lower()] if subject and subject.lower() in FORMULAS
        else list(FORMULAS.keys())
    )

    lines: list = []
    for subj in subjects:
        data = FORMULAS[subj]
        lines.append(f"\n{'═'*60}")
        lines.append(f"  {subj.upper()} — FORMULA SHEET")
        lines.append(f"{'═'*60}")
        for section, formulas in data.items():
            lines.append(f"\n  ── {section} ──")
            for f in formulas:
                lines.append(f"    {f}")

    lines.append(f"\n{'═'*60}")
    lines.append("  Use: /formulas physics  |  /formulas chemistry  |  /formulas math")
    lines.append(f"{'═'*60}")
    return "\n".join(lines)
