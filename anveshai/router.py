"""
Intent Router v4.0 — 8-way classifier tuned for JEE Advanced.

Priority order (evaluated top to bottom):
  1. system        : /commands
  2. logic         : formal logical structure (if-then, all-are, either-or)
  3. physics       : physics formula problems  ← moved BEFORE chemistry
  4. chemistry     : quantitative chemistry calculations
  5. advanced_math : symbolic math / word problems
  6. math          : plain arithmetic (numbers only)
  7. knowledge     : factual questions
  8. conversation  : everything else

v4.0 changes:
  - Physics now evaluated BEFORE chemistry to fix thermodynamics overlap
  - Fixed joules regex (case-sensitive J to avoid matching vector i,j,k)
  - Fixed Henry/Tesla/Farad/Newton patterns similarly
  - Added many missing physics keywords (escape velocity, orbital, optics, AC)
  - Added many missing math keywords (dy/dx, area enclosed, trig inverse, etc.)
  - Removed overlapping thermo keywords from chemistry that belong to physics
"""

import re

from preprocess import normalize_input


# ─────────────────────────────────────────────────────────────────────────────
# Conceptual question detection (used before physics/chemistry to avoid
# routing "What is Newton's second law?" → physics instead of knowledge).
# ─────────────────────────────────────────────────────────────────────────────

_CONCEPTUAL_STARTERS = re.compile(
    r'^(?:what\s+is\s+|what\s+are\s+|define\s+|explain\s+|describe\s+)',
    re.I,
)

_NOT_CONCEPTUAL_SIGNALS = re.compile(
    r'\b(?:calculate|compute|find|evaluate|solve|differentiate|integrate|'
    r'derive|determine|factorise|factorize|simplify|expand|'
    r'limit\s+of|limit\s+as|integral\s+of|derivative\s+of|'
    r'sum\s+of|product\s+of)\b'
    r'|\d',
    re.I,
)


def _is_conceptual_question(lowered: str) -> bool:
    """
    Return True for conceptual "what is X?" queries with no numeric/computational
    content.  These should route to knowledge even if they mention physics or
    chemistry concepts.

    Examples that return True:
        "what is newton's second law"
        "define entropy in thermodynamics"
        "what is escape velocity"
        "what is ph"

    Examples that return False (computational / numeric):
        "what is the ph of 0.01 m hcl"   ← has a digit
        "what is the derivative of x^2"   ← has 'derivative of'
        "find the limit of sin(x)/x"      ← starts with 'find'
    """
    if not _CONCEPTUAL_STARTERS.match(lowered):
        return False
    if _NOT_CONCEPTUAL_SIGNALS.search(lowered):
        return False
    return True


# ─────────────────────────────────────────────────────────────────────────────
# 1. Logic — structural patterns (check BEFORE math and knowledge)
# ─────────────────────────────────────────────────────────────────────────────

LOGIC_STRUCTURAL_PATTERNS: list[re.Pattern] = [
    re.compile(r"(?i)^if\s+.+\s+then\s+.+"),
    re.compile(r"(?i)if\s+.+\s+then\s+.+\bif\s+.+\s+then\b"),
    re.compile(r"(?i)^all\s+.+\s+(are|have|is|require|need|must|contain|use|produce|involve|consist)\s+\w"),
    re.compile(r"(?i)\beither\s+.+\s+or\s+.+"),
    re.compile(r"(?i)\btherefore\b"),
    re.compile(r"(?i)\bit\s+follows\s+that\b"),
    re.compile(r"(?i)\bdoes\s+it\s+follow\s+that\b"),
    re.compile(r"(?i)\bwhat\s+(follows|can\s+we\s+conclude|is\s+the\s+conclusion)\b"),
    re.compile(r"(?i)\bmodus\s+(ponens|tollens)\b"),
    re.compile(r"(?i)\b(hypothetical|categorical|disjunctive)\s+syllogism\b"),
    re.compile(r"(?i)\bsyllogism\b"),
    re.compile(r"(?i)\bcontrapositive\b"),
    re.compile(r"(?i)\btruth\s+table\b(?!\s+of\s+(?:nand|nor|xor|and|or|n?and|x?or)\s+gate)"),
    re.compile(r"(?i)\btautology\b"),
    re.compile(r"(?i)\bpropositional\s+logic\b"),
    re.compile(r"(?i)\bvalid\s+argument\b|\binvalid\s+argument\b"),
    re.compile(r"(?i)\bimplies\b.*\btherefore\b"),
    re.compile(r"(?i)\bis\s+(this|the)\s+(argument|reasoning|inference|syllogism)\s+valid\b"),
    # "P implies Q" — only for single-letter propositions or explicit logical questions,
    # NOT for everyday assertions like "the result implies that x=5"
    re.compile(r"(?i)\bif\s+\w+\s+impl(ies|y)\b"),
    re.compile(r"(?i)\bdoes\s+\w+\s+impl(y|ies)\b"),
    re.compile(r"(?i)\bcan\s+we\s+conclude\b"),
    re.compile(r"(?i)\bvalid\b.*\bif\b.*\bimpl"),
]


# ─────────────────────────────────────────────────────────────────────────────
# 2. Physics — physics formula problem keywords (JEE Advanced)
#    Evaluated BEFORE chemistry to fix thermodynamics overlap
# ─────────────────────────────────────────────────────────────────────────────

PHYSICS_KEYWORDS = [
    # Kinematics
    "kinematics", "suvat",
    "initial velocity", "final velocity",
    "starts from rest",
    "uniformly accelerated", "projectile",
    "time of flight", "range of projectile", "maximum height",
    "relative velocity", "relative motion",
    "horizontal range",
    # Dynamics
    "f = ma", "newton's second law", "net force",
    "centripetal force", "centripetal acceleration",
    "friction force", "coefficient of friction",
    "gravitational force", "gravitational constant",
    "normal force", "tension in string", "atwood",
    "escape velocity", "orbital velocity",
    "gravitational potential energy",
    # Collisions & momentum
    "elastic collision", "inelastic collision", "perfectly inelastic",
    "conservation of momentum", "coefficient of restitution",
    "common velocity after",
    "collide elastically", "collide inelastically", "embeds in", "bullet embeds",
    "head-on collision", "head on collision",
    # Energy / work / power
    "kinetic energy", "potential energy", "gravitational potential",
    "work done", "conservation of energy",
    "elastic potential", "mechanical energy",
    "power dissipated", "instantaneous power",
    # SHM
    "simple harmonic motion", "shm", "s.h.m",
    "angular frequency", "angular frequency ω",
    "time period of oscillation", "period of oscillation",
    "amplitude of oscillation", "restoring force",
    "spring constant", "spring-mass system", "spring mass",
    "spring balance",
    "displacement in shm", "velocity in shm", "acceleration in shm",
    "energy in shm",
    "pendulum", "simple pendulum", "physical pendulum",
    "seconds pendulum",
    "oscillates on a spring", "oscillates with amplitude", "oscillates with time period",
    "time period of spring", "time period 2 s", "time period t =", "time period t=",
    "mass oscillates", "block oscillates", "body oscillates",
    "find k, max velocity", "find the spring constant",
    # Rotational Mechanics
    "moment of inertia", "angular momentum",
    "rotational kinetic energy", "angular acceleration",
    "rolling without slipping", "rolling motion",
    "parallel axis theorem", "perpendicular axis theorem",
    "radius of gyration", "torque about",
    "angular impulse", "conservation of angular momentum",
    "moment of force", "couple",
    "angular velocity",
    # Waves / optics
    "wavelength", "wave speed",
    "snell's law", "refractive index", "critical angle",
    "lens formula", "focal length", "object distance", "image distance",
    "total internal reflection",
    "photon energy", "e = hf", "planck",
    "interference", "young's double slit", "fringe width",
    "diffraction", "resolving power",
    "standing waves", "stationary waves", "harmonics", "overtone",
    "beats", "doppler effect",
    "speed of sound", "velocity of sound",
    "convex lens", "concave lens", "convex mirror", "concave mirror",
    "mirror formula", "magnification",
    "angle of minimum deviation", "prism", "angle of deviation",
    "power of lens",
    # Electricity
    "ohm's law", "v = ir",
    "electric power", "electric current",
    "resistance in series", "resistance in parallel",
    "kirchhoff",
    # Electrostatics
    "coulomb's law", "coulomb law",
    "electric field", "electric flux",
    "gauss's law", "gauss law",
    "electric potential", "potential difference",
    "capacitance", "capacitor", "parallel plate capacitor",
    "dielectric", "permittivity",
    "electric dipole", "dipole moment",
    "energy stored in capacitor",
    "charge distribution",
    # Magnetism
    "magnetic field", "magnetic flux",
    "biot-savart", "biot savart",
    "ampere's law", "ampere law",
    "lorentz force", "magnetic force on charge",
    "faraday's law of induction", "faraday law",
    "lenz's law", "lenz law",
    "self-inductance", "self inductance", "mutual inductance",
    "inductance of solenoid", "inductance of coil",
    "magnetic dipole moment", "magnetic dipole",
    "cyclotron", "helical motion",
    "emf induced", "motional emf", "induced emf",
    "electromagnetic induction",
    "ac circuit", "impedance", "reactance", "resonance frequency",
    "transformer", "inductive reactance", "capacitive reactance",
    "rl circuit", "rc circuit", "rlc circuit",
    "time constant", "current gain",
    "q-factor", "quality factor",
    "rms current", "rms voltage", "rms value",
    "power factor",
    # Modern Physics
    "photoelectric effect", "work function",
    "threshold frequency", "threshold wavelength",
    "de broglie", "de-broglie", "matter wave",
    "compton effect", "compton scattering",
    "bohr model", "bohr radius", "bohr's model", "bohr orbit",
    "energy level", "energy of electron",
    "ionization energy", "ionization potential",
    "hydrogen spectrum", "spectral line", "spectral series",
    "balmer series", "lyman series", "paschen series", "brackett series", "pfund series",
    "hydrogen atom", "electron in hydrogen",
    "transition from n=", "makes a transition", "jump from n=",
    "nuclear fission", "nuclear fusion",
    "binding energy", "mass defect", "nuclear binding",
    "radioactivity", "alpha decay", "beta decay", "gamma decay",
    "alpha particle emitted", "alpha particles emitted",
    "nuclear reaction",
    "radioactive", "decay constant",
    "q-value", "q value of",
    "rms speed", "root mean square speed",
    "stopping potential",
    # Semiconductors
    "p-n junction", "pn junction",
    "diode", "transistor",
    "band gap", "depletion layer",
    "forward bias", "reverse bias",
    "rectifier", "logic gate",
    "npn transistor", "pnp transistor",
    "beta of transistor", "alpha of transistor",
    # Thermodynamics (physics context)
    "latent heat", "thermal energy",
    "carnot", "thermodynamic efficiency",
    "heat engine", "coefficient of performance",
    "entropy",
    "isobaric", "isochoric", "adiabatic", "isothermal",
    "monatomic", "diatomic",
    "internal energy", "molar heat capacity",
    "specific heat ratio", "heat capacity ratio",
    "ideal gas", "pv = nrt",
    "work done by gas",
    "rms velocity", "mean kinetic energy of",
    # Fluid
    "fluid pressure", "hydrostatic pressure", "archimedes",
    "buoyancy", "bernoulli",
    # Other
    "circular motion",
    "vertical circle", "swung in a circle",
    "electromagnetic", "electromagnetic wave",
    "and gate", "nand gate", "nor gate", "xor gate",
    "full-wave rectifier", "half-wave rectifier",
    "rolls down incline", "rolls down a slope",
    "lateral shift", "glass slab",
    "minimum speed required",
    "deceleration",
    "braking force", "stopping distance", "brakes applied",
    "angle of repose", "pump of power", "power of pump",
    "centre of mass of", "center of mass of",
    # Additional kinematics / mechanics
    "slides down", "slides on", "frictionless incline", "frictionless surface",
    "frictionless curved", "frictionless plane",
    "bullet of mass", "gun fires", "gun of mass", "recoil velocity",
    "satellite orbits", "satellite at height", "orbital speed",
    "satellite moves", "geostationary", "orbital period",
    "boat moves", "river flows", "cross the river",
    "man of mass", "weighing scale", "lift accelerates", "elevator accelerates",
    "spring releases", "compressed spring", "spring compressed",
    "angular acceleration of a wheel", "wheel speeds up", "rpm to",
    "two bodies collide", "collision between", "rocket ejects",
    "rate of fuel", "thrust is",
    # Rotational extras
    "skater spinning", "extends arms", "moment of inertia changes",
    "conservation of angular", "angular momentum conservation",
    "rolling cylinder", "rolling sphere", "rolling disc",
    "disc and ring", "reaches bottom", "roll down incline",
    "uniform rod rotated", "thin rod rotated", "rod rotated about",
    # Optics extras
    "near point", "far point", "glasses of power", "power of glasses",
    "myopic", "hypermetropic", "presbyopia", "eye defect",
    "person uses glasses", "corrective lens",
    "newton's rings", "diameter of ring",
    "single slit", "double slit",
    "image height", "image is formed", "object is placed",
    # AC / Electrical extras
    "phase difference between voltage", "capacitive circuit", "pure capacitor",
    "pure inductor", "pure resistor",
    "series rlc", "series lcr", "lc circuit",
    "peak voltage", "peak current", "average output",
    "ce amplifier", "common emitter", "common base", "common collector",
    "amplifier circuit", "beta=", "ic=", "vce", "vcc",
    "zener diode", "breakdown voltage", "series resistance",
    # Nuclear / modern extras
    "nuclear radius", "nuclear density", "mass number a",
    "radius of nucleus", "r0=", "r0 =",
    "activity of a sample", "activity decreases", "activity of 1 g",
    "disintegrations per second", "dps", "becquerel",
    "number of beta particles", "beta particles emitted",
    "fraction remaining after", "fraction remaining",
    "thorium series", "uranium series",
    "mean life", "average life",
    "atomic nucleus", "nuclear charge",
    "x-ray photon", "bragg reflection", "bragg's law",
    # Semiconductor extras
    "logic gate", "and gate output", "nor gate", "xnor gate",
    "full wave rectifier", "zener",
    "output of and gate", "output of nand",
    "p-n junction diode", "forward resistance",
    # More mechanics / optics / nuclear
    "pulled up a", "pushed up a", "pulled along the incline",
    "up the incline", "up an incline", "along the incline",
    "at constant speed by force", "constant speed up",
    "elevator moving upward", "elevator moving downward",
    "person stands on scale", "stands on weighing",
    "lift moving", "lift decelerates", "lift accelerating upward",
    "thrown at 30°", "thrown at 45°", "thrown at 60°",
    "needs to clear", "clear a wall", "wall at distance",
    "uniformly charged ring", "charged ring", "ring of charge",
    "point on axis of", "axis of ring",
    "mean free path", "free path",
    "sound level in db", "sound level in decibel", "decibel",
    "intensity i0", "intensity in db",
    "mi of a thick", "mi of thick", "thick ring",
    "inner radius r", "inner radius r and outer",
    "emits alpha", "nucleus emits", "parent nucleus",
    "becomes radon", "becomes lead", "becomes uranium",
    "element after", "daughter nucleus",
    "speed of light", "c=3×10^8",
    "momentum of photon", "energy of photon",
    "escape velocity from", "escape velocity of moon",
    "terminal velocity", "stoke's law",
    "diffraction grating", "lines per mm",
    # Modern Physics extras
    "pair production", "pair annihilation",
    "gamma ray photon", "gamma photon",
    "14c activity", "carbon-14", "radioactive dating",
    "bone sample", "carbon dating", "archaeological",
    "age of a bone", "age of a rock",
    # Prevent "find the minimum frequency" from being stolen from physics
    "minimum frequency of", "threshold frequency for pair",
]

PHYSICS_CONTEXT_PATTERNS = [
    # Kinematics patterns
    re.compile(r'(?i)(?:travels?|moves?|accelerates?)\s+at\s+[\d.]+\s*(?:m/s|km/h|mph)'),
    # Force: number of newtons — capital N only (avoids n=moles in chemistry)
    re.compile(r'[\d.]+\s*N\b(?!\s*a\b)(?!\s*m\b)'),
    # Energy: joules — CASE-SENSITIVE J to avoid vector notation j
    # Exclude kJ/mol and kJ·mol^-1 (chemistry contexts) — only match bare J/kJ
    re.compile(r'[\d.]+\s*J\b(?!\s*/|\s*·mol)|[\d.]+\s*kJ\b(?!\s*/mol|\s*·mol|\s*mol)'),
    # Voltage/Current/Resistance
    re.compile(r'(?i)\b(?:volts?|amperes?|ohms?|watts?)\b'),
    # Frequency / wavelength
    re.compile(r'[\d.]+\s*(?:Hz|kHz|MHz|GHz|nm)\b'),
    # "calculate/find the [physics quantity]"
    re.compile(r'(?i)(?:calculate|find|determine)\s+(?:the\s+)?'
               r'(?:velocity|acceleration|force|energy|momentum|power|pressure|'
               r'density|frequency|wavelength|period|voltage|current|resistance|'
               r'heat|temperature|inductance|capacitance|charge|flux|field|'
               r'amplitude|angular\s+momentum|moment\s+of\s+inertia|torque|'
               r'image\s+distance|focal\s+length|refractive\s+index|'
               r'fringe\s+width|impedance|peak\s+emf|rms\s+current)'),
    # Spring / SHM patterns
    re.compile(r'(?i)(?:spring|oscillat|vibrat|periodic)\s+(?:constant|motion|frequency|period)'),
    # Angle of incidence / refraction
    re.compile(r'(?i)(?:angle\s+of\s+(?:incidence|refraction|reflection|minimum\s+deviation)|critical\s+angle)'),
    # Charge / electron volt — CASE-SENSITIVE eV, MeV, μC etc.
    re.compile(r'[\d.]+\s*(?:eV|MeV|keV|μC|mC|nC)\b'),
    re.compile(r'(?i)[\d.]+\s*(?:coulombs?)\b'),
    # Tesla / Gauss — CASE-SENSITIVE T
    re.compile(r'[\d.]+\s*(?:T\b|tesla|gauss|Wb|weber)'),
    # Henry (inductance) — CASE-SENSITIVE H
    re.compile(r'[\d.]+\s*(?:H\b|mH\b|μH\b|henry|henries)'),
    # Farad (capacitance) — CASE-SENSITIVE F
    re.compile(r'[\d.]+\s*(?:F\b|μF\b|nF\b|pF\b|farad)'),
    # Ohm — uppercase Ω or Ohm
    re.compile(r'(?i)[\d.]+\s*(?:Ω|ohms?)\b'),
    # Specific physics contexts
    re.compile(r'(?i)\b(?:rolls?\s+without\s+slipping|projectile\s+motion|'
               r'elastic\s+collision|inelastic\s+collision|'
               r'escape\s+velocity|orbital\s+velocity|'
               r'peak\s+emf|rms\s+speed|speed\s+of\s+sound)\b'),
]


_CHEMISTRY_PRIORITY_KEYWORDS = [
    "activation energy", "arrhenius", "first order", "second order", "zero order",
    "rate constant", "rate of reaction", "rate law", "order of reaction",
    "hess", "enthalpy of", "heat of formation", "heat of combustion",
    "heat of neutralization", "bond enthalpy", "lattice energy",
    "born-haber", "delta h", "δh", "delta_h°", "delta_hrxn",
    "delta_h=", "delta_h =", "delta_h:-", "delta_h:",
    "delta_g", "delta_g°", "delta g°", "gibbs",
    # Molecular polarity / dipole comparisons → always chemistry
    "dipole moment",
    "nernst", "electrode potential", "electrolysis", "electrochemistry",
    "e°cell", "e0cell", "e°_cell", "ecell", "emf of cell", "emf changes",
    "f=96500", "f =96500", "faraday constant",
    "cell constant", "conductance of", "molar conductivity",
    "unit cell", "packing", "bcc", "fcc", "edge length",
    "nacl type", "kcl type", "cscl type", "rock salt structure",
    "rock salt type", "zinc blende", "wurtzite",
    "colligative", "osmotic pressure", "van't hoff", "freezing point depression",
    "boiling point elevation", "boiling point of solution", "raoult",
    "hybridization", "vsepr", "bond order", "formal charge",
    "bond dissociation energy", "bond dissociation enthalpy",
    "magnetic moment of", "spin-only", "paramagnetic", "diamagnetic",
    "avogadro", "moles of", "molar mass=",
    "molarity", "molality", "normality",
    "decreasing basicity", "increasing basicity", "order of basicity",
    "order of acidity", "decreasing order of acidity",
    "iupac name of", "name of compound",
    "kj/mol", "kj mol",
    # Prevent fluid Bernoulli from stealing math ODE Bernoulli  
    "bernoulli equation:", "dy/dx + y = y", "dy/dx+y=y",
    # Prevent "power of lens" from stealing power series
    "power series expansion of arctan",
]

def _is_physics(text: str, lowered: str) -> bool:
    """Return True if text is a physics formula problem."""
    # Phase -1: Pure abstract math vector problems that look like physics (exclude from physics)
    # NOTE: "moment of force" with actual force vectors IS physics (torque), so NOT blocked here.
    _MATH_VECTOR_OVERRIDES = [
        "find a×b where", "a=2i+j and b=", "a=3i-2j",
    ]
    for kw in _MATH_VECTOR_OVERRIDES:
        if kw in lowered:
            return False
    # Phase 0: STRONG physics overrides — always physics regardless of chemistry keywords
    _PHYSICS_STRONG_OVERRIDES = [
        "bragg reflection", "bragg's law", "bragg diffraction",
        "x-ray diffraction", "x-ray photon", "x ray diffraction",
        "rms speed of", "root mean square speed of",
        "speed of sound in", "velocity of sound in",
        "photoelectric effect", "de broglie",
        "compton effect", "compton scattering",
        "decay constant", "radioactive decay constant",
        # Diffraction grating optics (slash and "per" forms)
        "diffraction grating", "lines/mm", "lines per mm",
        # Thermodynamic processes (ideal gas expansion)
        "isobarically", "isobaric process", "isobaric expansion",
        "isothermally", "isothermal expansion", "isothermal process",
        "adiabatically", "adiabatic expansion", "adiabatic process",
        "moles of ideal gas", "ideal gas expands", "gas expands isobar",
        "ideal diatomic gas", "ideal monoatomic gas",
        # Moment of inertia abbreviation (MI)
        "mi of solid", "mi of hollow", "mi of a solid", "mi of a hollow",
        "mi of uniform", "mi of thin", "mi of a thin", "mi of a uniform",
        # Torque with force vectors
        "moment of force f=", "moment of force f =",
    ]
    for kw in _PHYSICS_STRONG_OVERRIDES:
        if kw in lowered:
            return True
    # Pre-check: if strong math/chemistry signals are present, skip physics
    for kw in _CHEMISTRY_PRIORITY_KEYWORDS:
        if kw in lowered:
            return False  # Let chemistry or advanced_math handler take it
    # Phase 1: Exact keyword match
    for kw in PHYSICS_KEYWORDS:
        if kw in lowered:
            return True
    # Phase 2: Context patterns
    for pat in PHYSICS_CONTEXT_PATTERNS:
        if pat.search(text):
            return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# 3. Chemistry — quantitative chemistry problem keywords (JEE Advanced)
#    Evaluated AFTER physics to avoid thermodynamics overlap
# ─────────────────────────────────────────────────────────────────────────────

CHEMISTRY_STRONG_KEYWORDS = [
    # Unicode / Greek delta forms (ΔH lowercased → δh)
    "δh", "δu", "δg", "δs", "δv", "δhrxn", "δhf",
    "∆h", "∆u", "∆g", "∆s",
    # Van der Waals
    "van der waals", "van der waals equation",
    # Functional groups / organic identification
    "functional group", "functional groups",
    "identify the functional", "identify functional",
    "write balanced", "write the balanced",
    "balanced chemical equation",
    "balance the equation", "balance equation", "balancing equation",
    "balance this equation", "balance chemical",
    "write reaction for", "write the reaction for",
    "preparation of benzaldehyde", "preparation of aniline",
    "chromyl chloride", "etard reaction",
    "distinguish between", "lucas test",
    "aldehyde or ketone", "ketone or aldehyde",
    # Bond dissociation (plural form)
    "bond dissociation energies",
    # Moles / stoichiometry
    "how many moles", "moles of", "number of moles",
    "molar mass of", "molecular weight of", "atomic mass of", "formula mass",
    "stoichiometry", "limiting reagent", "theoretical yield", "molar ratio",
    "percent composition", "percentage composition", "mass percent",
    # Concentration / solutions
    "molarity", "molality", "normality",
    "concentration of solution", "moles per litre", "mol/l",
    "dilution", "c1v1", "c₁v₁",
    # Acids / bases / pH
    "calculate ph", "find ph", "ph of", "what is the ph",
    "poh of", "calculate poh",
    "ka =", "kb =", "weak acid", "weak base", "strong acid", "strong base",
    "buffer solution", "henderson",
    # Gas laws (pure chemistry context)
    "boyle's law", "boyle law", "boyles law",
    "charles's law", "charles law", "charless law",
    "gay-lussac", "gay lussac",
    "combined gas law",
    # Thermochemistry / calorimetry
    "enthalpy", "calorimetry", "heat of reaction", "delta h",
    "q = mcdelta", "q=mcdelta", "specific heat capacity",
    "heat capacity", "q = mc", "mcdeltat",
    "hess's law", "hess law", "bond enthalpy", "lattice energy",
    "born-haber", "heat of formation", "heat of combustion",
    "heat of neutralization", "heat of dissolution",
    "heat of formation",
    # Electrochemistry
    "electrochemistry", "nernst", "cell potential", "standard electrode",
    "faraday", "electrolysis",
    "delta g", "δg", "gibbs energy", "gibbs free energy",
    "e° =", "e0 =",
    "electrode potential", "standard emf", "emf of cell",
    "e°cell", "e0cell",
    # General quantitative chemistry
    "mole fraction", "equivalent weight",
    # Chemical equilibrium
    "equilibrium constant", "kc =", "kp =", "kc=", "kp=",
    " kc ", " kp ", "kc for", "kp for",
    "le chatelier", "degree of dissociation", "extent of reaction",
    "reaction quotient", "qc", "qp",
    "equilibrium concentration", "equilibrium pressure",
    "kw =", "ionic product", "solubility product", "ksp",
    "degree of ionisation", "degree of ionization",
    # Chemical kinetics
    "rate of reaction", "rate constant", "rate law",
    "order of reaction", "first order", "second order", "zero order",
    "activation energy", "arrhenius equation", "arrhenius",
    "integrated rate law",
    "frequency factor", "pre-exponential",
    "temperature coefficient",
    "rate = k[", "rate=k[", "rate =k[",
    "fraction of reactant", "fraction remaining",
    "what fraction remains",
    "t3/4", "3t1/2", "half-life of a first order", "half-life of first order",
    "half-life of second order", "first-order kinetics", "first order kinetics",
    "by what factor does k", "by what factor does the rate",
    "rate doubles", "rate triples",
    # Coordination chemistry
    "coordination compound", "coordination number",
    "effective atomic number", " ean =", " ean=",
    # Solid state
    "unit cell", "packing fraction", "packing efficiency",
    "bcc", "fcc", "simple cubic", "coordination number in crystal",
    "edge length", "radius ratio", "number of atoms per unit cell",
    "density of crystal", "ionic crystal",
    "octahedral void", "tetrahedral void", "close-packed", "close packed",
    "density of nacl", "density of kcl", "density of cscl",
    # Surface chemistry
    "adsorption isotherm", "freundlich", "langmuir",
    "degree of adsorption",
    # Colligative properties
    "vapour pressure", "raoult's law", "raoult law",
    "boiling point elevation", "freezing point depression",
    "osmotic pressure", "van't hoff factor", "van't hoff",
    "molal elevation constant", "molal depression constant",
    "kb =", "kf =",
    "depression in freezing point", "elevation in boiling point",
    "cryoscopic", "ebullioscopic",
    "δtb", "δtf", "deltatb", "deltatf",
    # Organic
    "optical rotation", "degree of unsaturation", "index of hydrogen deficiency",
    "empirical formula", "molecular formula",
    "degree of polymerisation",
    # Chemical bonding / structure
    "hybridization", "vsepr", "molecular orbital", "bond order",
    "formal charge", "resonance structure",
    "paramagnetic", "diamagnetic",
    "sigma bond", "pi bond",
    "magnetic moment of",
    "spin-only formula", "unpaired electrons",
    # Stoichiometry / volumetric
    "volume of co2", "volume of gas produced", "mass of",
    "percentage yield", "theoretical mass",
    "at stp", "at s.t.p",
    "number of molecules", "avogadro",
    "deposit aluminium", "deposit copper", "deposit silver",
    "deposited from", "charge required to deposit",
    "normality", "equivalent",
    "⇌", "⇒", "→ products",
    # Nuclear (chemistry context - radioactive dating etc)
    "radioactive decay constant", "nuclear equation",
    # Organic chemistry
    "iupac name", "structural isomers", "structural isomer",
    "markovnikov", "anti-markovnikov",
    "sn1", "sn2", "sn1 mechanism", "sn2 mechanism",
    "elimination reaction", "e1", "e2",
    "aldol condensation", "aldol reaction",
    "ozonolysis", "ozonolyse",
    "friedel-crafts", "friedel crafts",
    "halogenation", "free radical",
    "asymmetric carbon", "chiral center", "chiral centre",
    "optical isomer", "enantiomer", "diastereomer",
    "degree of unsaturation", "index of hydrogen deficiency",
    "dehydration of ethanol", "dehydration of",
    "alcohol to", "oxidation of alcohol",
    "acidity order", "order of acidity", "increasing order of acidity",
    "boiling point order", "increasing order of boiling",
    "oxidation state of central", "oxidation state in complex",
    "cfse", "crystal field", "high spin", "low spin",
    "strong field", "weak field ligand",
    "geometric isomers", "optical isomers",
    "complex ion", "coordination compound", "coordination complex",
    "predict the product", "major product when",
    "product of reaction", "product when",
    "identify the type of reaction", "identify the reaction",
    "type of halogenation",
    "free radical halogenation",
    # Additional electrochemistry
    "kohlrausch", "molar conductivity",
    "specific conductance", "cell constant",
    "conductance of", "resistance of solution",
    "lambda_m", "lambda0",
    # Stoichiometry extras
    "volume of h2so4", "volume of kmno4",
    "grams of nh3", "grams of o2", "grams of co2",
    "number of atoms in", "number of molecules in",
    "percentage of nitrogen", "percentage by mass",
    "percentage composition of",
    "how many grams of", "how many moles of",
    "how many litres of", "what volume of",
    "amount of silver", "mass of silver",
    "time required to deposit",
    # Solutions extras
    "ppm", "parts per million",
    "mass fraction", "mole fraction of",
    "henry's law", "henry law", "henry's law constant",
    "deposited when", "deposited at", "amount deposited",
    "mass of naoh required", "mass of hcl required", "mass of h2so4 required",
    "number of pi bonds", "number of sigma bonds",
    "asymmetric carbon in", "number of asymmetric",
    # Thermochemistry extras
    "heat released when", "heat evolved when",
    "heat released per gram", "heat released per mole",
    "resonance energy of benzene",
    "heat of vaporization of water",
    "delta_h for", "delta_hrxn",
    "find delta h", "calculate delta h",
    "enthalpy change for",
    # Equilibrium extras
    "find kc for", "find kp for",
    "degree of hydrolysis",
    "hydrolysis of sodium", "hydrolysis of salt",
    "ph at equivalence", "equivalence point",
    "concentration of oh-", "concentration of h+",
    "concentration of oh", "concentration of h",
    "find the concentration of",
    "[oh-]", "[h+]", "[h3o+]",
    # Equilibrium concentration bracket patterns
    "[n2]=", "[h2]=", "[nh3]=", "[no]=", "[co]=", "[pcl5]=",
    "at equilibrium [", "equilibrium [",
    "find kc", " find kp",
    # Solid state extras
    "cscl type structure", "cscl structure", "nacl type structure",
    "radius of cl-", "radius of cs+", "radius of na+",
    " pm ", " angstrom", "angstrom.", "radius ratio in",
    # Electrochemistry extras
    "zn2+ concentration", "[zn2+]", "concentration changes",
    "emf changes from",
    "charge in coulombs", "charge in faraday",
    "coulombs is required", "required to deposit",
    # Colligative extras
    "kb=", "kb =", "kf=", "kf =",
    "delta_tb", "delta_tf", "find delta_tb", "find delta_tf",
    "containing 0.5 mol", "containing 1 mol",
    "solution containing",
    # Thermochemistry extras
    "bond dissociation energy of h2",
    "dissociation energy of h2",
    # Organic extras
    "decreasing basicity", "increasing basicity",
    "major product of bromination",
    "bromination of toluene", "nitration of toluene",
    "in presence of febr3", "in presence of alcl3",
    "type of isomerism", "isomerism between",
    "glucose and fructose", "optical isomerism",
    "cannizzaro reaction", "cannizzaro",
    "kolbe", "kolbe-schmitt", "kolbe reaction",
    "tollens' test", "tollens test", "fehling test",
    "iodoform test", "lucas test with",
    "hydrogen bonding", "hydrogen bond",
    "boiling point than ph3", "higher boiling point than",
    "oxidation state of s in", "oxidation state of n in",
    "oxidation state of cl in", "oxidation state of cr in",
    "oxidation state of mn in",
    # Needed to react / neutralize
    "needed to react with", "needed to neutralize",
    "required to neutralize", "required to react with", "required to titrate",
    " neutralize ", "neutralize 250", "neutralize 100", "neutralize 50",
    "h2so4 needed", "needed to titrate",
    "stoichiometry of", "h2so4 to",
    # Organic chemistry identification
    "identify the primary", "primary, secondary and tertiary",
    "primary secondary and tertiary", "tertiary carbons",
    "primary and secondary carbons", "2-methylbutane", "isopentane",
    "neopentane", "2-methylpropane", "isobutane",
]

CHEMISTRY_FORMULA_PATTERN = re.compile(
    r'\b([A-Z][a-z]?\d+|[A-Z]{2}[a-z]?\d*)'
    r'(?:\([A-Z][a-z]?\d*\)\d*)*'
    r'\s*(?:molecule|formula|compound|solution|gas|liquid|solid)?\s*'
    r'(?:has|have|is|are|contains?|with|of)?\s*'
    r'(?:molar\s+mass|molecular\s+weight|moles?|concentration)'
)


def _is_chemistry(text: str, lowered: str) -> bool:
    """Return True if text is a quantitative chemistry question."""
    for kw in CHEMISTRY_STRONG_KEYWORDS:
        if kw in lowered:
            return True
    if CHEMISTRY_FORMULA_PATTERN.search(text):
        return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# 4. Advanced math — symbolic operations + word problems (JEE Advanced)
# ─────────────────────────────────────────────────────────────────────────────

ADVANCED_MATH_KEYWORDS = [
    # Calculus — differentiation
    "integrate", "integral", "antiderivative", "indefinite integral",
    "definite integral", "∫",
    "differentiate", "derivative", "d/dx", "d/dy", "d/dz", "d/dt",
    "dy/dx", "dx/dy", "d²y/dx²",
    "second derivative", "third derivative", "nth derivative", "partial derivative",
    "partial diff", "gradient of",
    "limit of", "limit as", "lim ", "find the limit",
    "rolle's theorem", "mean value theorem", "lagrange's theorem",
    "maxima", "minima", "local maximum", "local minimum",
    "point of inflection", "concavity",
    "monotonically increasing", "monotonically decreasing",
    "increasing function", "decreasing function",
    "strictly increasing", "strictly decreasing",
    # Calculus — integration / area
    "area enclosed", "area bounded", "area between",
    "area under the curve", "area of the region",
    "volume of revolution", "solid of revolution",
    "rotated about",
    # Differentiation triggers
    "product rule", "chain rule", "quotient rule",
    "implicit differentiation", "logarithmic differentiation",
    "parametric differentiation",
    "tangent to curve", "normal to curve",
    "equation of tangent", "equation of normal",
    # Differential equations
    "differential equation", "ode ", "dsolve", "solve the ode",
    "y'' ", "y' ", "d2y", "d^2y", "pde ", "partial differential",
    # Algebra / equations
    "solve ", "find roots", "zeros of", "find the value of x",
    "quadratic formula", "discriminant",
    # Linear algebra
    "eigenvalue", "eigenvector", "determinant of", "det of",
    "inverse matrix", "matrix inverse", "rank of matrix", "matrix rank",
    "trace of matrix", "matrix trace", "characteristic polynomial",
    "dot product", "cross product",
    "find the inverse of matrix",
    # Series & transforms
    "taylor series", "maclaurin series", "series expansion", "power series",
    "laplace transform", "laplace of", "inverse laplace",
    "fourier transform", "fourier of",
    # Symbolic manipulation
    "simplify ", "simplify(", "factor ", "factorise", "factorize",
    "expand ", "partial fraction",
    # Number theory
    "gcd(", "gcd of", "greatest common divisor", "highest common factor",
    "lcm(", "lcm of", "least common multiple",
    "prime factor", "prime factorization",
    " mod ", "modulo ", "modular inverse",
    "congruence", "find all integers x such that",
    # Statistics
    "mean of", "average of", "median of", "variance of",
    "standard deviation of", "std dev of",
    "coefficient of variation",
    # Combinatorics
    "factorial of", "factorial(", "binomial coefficient",
    "choose ", "ncr", "npr", "permutation",
    "in how many ways", "number of ways", "arrangements",
    "circular permutation", "derangement",
    "multinomial",
    "c(12,", "p(12,", "c(n,", "p(n,",
    # Summations & products
    "sum of ", "summation of", "∑", "∏",
    "sum σ(k", "find the sum σ",
    # Complex numbers
    "complex number", "real part of", "imaginary part of",
    "modulus of", "argument of", "conjugate of",
    "polar form", "euler form", "de moivre",
    "cube roots of unity", "nth roots of unity",
    "locus of", "locus in argand",
    "|z|", "z=", "z̄",
    # Trigonometric identity simplification
    "simplify trig", "trig simplif", "trigonometric simplif",
    # Coordinate Geometry
    "conic section", "parabola", "ellipse", "hyperbola",
    "chord of contact", "pair of tangents",
    "pole and polar", "director circle",
    "pair of lines", "combined equation",
    "asymptotes of hyperbola",
    "eccentricity", "directrix", "focus of", "latus rectum",
    "normal to ellipse", "tangent to parabola", "tangent to ellipse",
    "tangent to hyperbola", "tangent to circle",
    "chord of curve", "chord bisected",
    "radical axis", "coaxial circles",
    "family of circles", "equation of circle",
    "orthogonal circles",
    "straight line", "slope of line", "intercept form",
    "distance from point to line", "angle between lines", "angle between the lines",
    "angle between two lines",
    "concurrent lines", "collinear points",
    "centre and radius", "center and radius",
    "equation of line", "equation of a line", "line passing through",
    "line through", "slope of",
    "length of tangent", "tangent from point", "tangent from",
    "perpendicular distance", "distance from point",
    "foot of perpendicular", "image of point", "reflection of point",
    "area of triangle", "area of a triangle",
    "area of parallelogram",
    # 3D Geometry & Vectors
    "direction cosines", "direction ratios",
    "skew lines", "shortest distance",
    "coplanar lines", "coplanar vectors",
    "plane equation", "equation of plane",
    "distance from plane", "angle between planes",
    "distance between parallel planes", "distance between planes",
    "line of intersection of planes",
    "unit vector", "position vector",
    "scalar triple product", "vector triple product",
    "projection of vector", "component of vector",
    "section formula", "midpoint formula",
    "foot of perpendicular from point",
    "image of point in the plane",
    "angle between vectors",
    "vector perpendicular", "perpendicular to both",
    "|a×b|", "a·b",
    # Probability
    "conditional probability", "bayes theorem", "bayes' theorem",
    "random variable", "probability distribution",
    "expected value", "expectation of",
    "binomial distribution", "poisson distribution",
    "normal distribution",
    "mutually exclusive", "independent events",
    "probability that", "probability of",
    "probability all", "probability both", "probability exactly",
    "find p(", "find p (", "p(x=", "p(x =",
    "cards drawn", "balls drawn", "drawn without replacement",
    "hypergeometric",
    # Sequences & Series
    "arithmetic progression", "geometric progression", "harmonic progression",
    "ap ", "gp ", "hp ",
    "arithmetic mean", "geometric mean", "harmonic mean",
    "am between", "gm between", "hm between",
    "sum of n terms", "nth term of", "common difference",
    "common ratio",
    "sum to infinity", "infinite gp",
    "n terms of ap", "n terms of gp",
    "sum of n terms of ap",
    # Trigonometry
    "trigonometric equation", "general solution",
    "inverse trig", "arcsin", "arccos", "arctan",
    "sin inverse", "cos inverse", "tan inverse",
    "sin^-1", "cos^-1", "tan^-1",
    "sin⁻¹", "cos⁻¹", "tan⁻¹",
    "principal value", "principal solution",
    "trigonometric identities", "pythagorean identities",
    "compound angle", "multiple angle", "sub-multiple angle",
    "half angle formula", "double angle formula",
    "sum to product", "product to sum",
    "sum-to-product", "product-to-sum",
    "sine rule", "cosine rule", "area of triangle using trig",
    "included angle",
    # Mathematical Induction & Binomial
    "mathematical induction", "principle of induction",
    "binomial theorem", "binomial expansion",
    "general term", "middle term in binomial", "middle term",
    "greatest term in (", "greatest term in the expansion",
    "find the greatest term",
    "coefficient of x", "term independent of x",
    "expansion of (", "in the expansion",
    # Word problems
    "area of a rectangle", "area of the rectangle",
    "area of a square", "area of a circle",
    "area of a trapezoid", "area of a parallelogram",
    "perimeter of a", "perimeter of the",
    "volume of a cube", "volume of a cuboid", "volume of a cylinder",
    "volume of a sphere", "volume of a cone",
    "circumference of",
    "surface area of",
    "percent of", "% of", "percentage of",
    "simple interest", "compound interest",
    "average speed", "total distance", "time taken to travel",
    "miles per hour", "km per hour", "kmph",
    "work together", "fills the tank", "rate of work",
    "ratio of", "in the ratio",
    # GP/AP/HP with parentheses or different spacing (e.g. "(GP)", "find sum GP")
    "(gp)", "(ap)", "(hp)", " gp)", " ap)", " hp)",
    "up to n terms", "up to 10 terms", "up to 5 terms",
    "find the sum: ", "find the sum:",
    "sum: 3 +", "sum: 1 ·", "sum: 1·",
    # Inverse trig with notation variations
    "tan^(-1", "sin^(-1", "cos^(-1",
    "tan^{-1", "sin^{-1", "cos^{-1",
    "tan^-1", "sin^-1", "cos^-1",
    "tan⁻¹", "sin⁻¹", "cos⁻¹",
    "tan−1", "sin−1", "cos−1",
    "prove: tan^", "prove: sin^", "prove: cos^",
    # Trig equations with sec/tan compound forms
    "tan(a-b)=", "sec(a+b)=", "tan(a+b)=", "sec(a-b)=",
    "tan(a -b)", "if tan(", "if sec(", "if sin(",
    "find smallest positive values", "smallest positive values of a",
    # Trig proofs / show that
    "show that sin(", "prove that sin(", "show that cos(", "prove that cos(",
    "show that tan(", "prove that tan(", "derive expressions for sin",
    "sin(a+b)·sin", "sin(a+b)*sin", "prove: (1+sin",
    "show that (1+sin", "sin²(a)+cos²(a)",
    "sin^2(a)+cos^2(a)", "prove that sin²",
    # Trig value computation with degrees
    "value of tan(75", "value of sin(75", "value of cos(75",
    "value of tan(15", "value of sin(15", "value of cos(15",
    "using addition formula", "using compound angle",
    "tan(75°)", "tan(15°)", "cos(15°)", "sin(75°)",
    "exact form", "exact value of",
    # Absolute value equations
    "solve: |", "|2x-", "|x-", "|x+", "|3x", "|2x+",
    "solve |", "equation with |",
    # Complex numbers with Unicode / Greek
    "α and β are roots", "α and β", "β are roots of",
    "α²+β²", "α³+β³", "α+β =", "αβ",
    "z₁=", "z₂=", "z₁ and z₂", "z₁·z₂",
    "find z₁", "z₁/z₂",
    # Vector with magnitude notation |a|, |b|, |c|
    "if a+b+c=0", "|a|=", "|b|=", "|c|=",
    "a+b+c=0 and |",
    # Roots of unity generalized
    "roots of unity", "5th roots of", "4th roots of", "nth roots of",
    "unity and their sum", "all nth roots",
    # Coordinate geometry proofs
    "angle in a semicircle", "angle subtended in",
    "prove that the angle in", "semicircle is 90",
    "prove using coordinate",
    # Matrices with [[...]] or det notation
    "det(a)=", "det(2a)", "det(a^t)", "det(a^t",
    "[[1,2]", "[[2,0]", "[[1,0]", "[[0,1]",
    "find ab and ba", "find ab, ba", "verify ab≠",
    "verify ab ≠", "if a is a 3×3 matrix",
    "if a is a 3x3 matrix",
    # Combinatorics additional
    "from 6 men and", "from n men and", "3 men and 2 women",
    "how many teams", "how many groups can",
    "choose a team of", "select a team of",
    # Mathematical proofs
    "prove that √2 is irrational", "prove √2 is",
    "√2 is irrational", "irrational number",
    "proof by induction for", "prove by induction that",
    # Trigonometry: sum-to-product patterns in equations
    "sin(x) + sin(3x)", "sin x + sin 3x",
    "cos(x) + cos(3x)", "find all solutions of sin",
    "find all solutions of cos", "find all solutions of tan",
    "sin(x)+sin(", "cos(x)+cos(",
    # Calculus extras — limits
    "lim(", "lim(x", "evaluate lim", "find lim",
    "lim(n", "lim(theta", "l'hopital", "l'hôpital",
    "0/0 form", "indeterminate form",
    # Calculus extras — maxima/minima
    "maximum and minimum values", "maximum value of f", "minimum value of f",
    "find the maximum value", "find the minimum value",
    "maximum value of the function", "minimum value of the function",
    "find the maximum", "find the minimum",
    "bounded by y=", "area of region bounded", "region bounded by",
    "area of the region", "bounded by the curves",
    "volume of cone", "volume of sphere using integration",
    # Differentiation extras
    "angle between tangents", "angle between the tangents",
    "angle of intersection", "angle of intersection of curves",
    "angle of intersection of circles",
    "tangent to curve y=", "tangent to the curve",
    "normal to curve", "normal to the curve",
    # Geometry extras
    "distance between parallel lines", "distance between the parallel",
    "equation of angle bisectors", "angle bisectors of lines",
    "length of common chord", "common chord of circles",
    "angle of intersection of circles",
    "equation of sphere", "centre and radius of sphere",
    # Vector extras
    "projection of a=", "projection of vector a",
    "component of a perpendicular", "perpendicular component",
    "prove that a and b are perpendicular",
    "work done by force f=", "work done by force",
    # Sequences extras
    "am of two numbers", "gm of two numbers",
    "am and gm", "a.m. of", "g.m. of",
    "the am of", "the gm of",
    # Probability extras
    "poisson distribution has mean", "poisson distribution mean",
    "normal distribution mean", "binomial distribution has",
    "find p(x=0)", "find p(x=1)", "find p(x>",
    "p(x>=", "p(x >=", "p(x=", "p(x =",
    # Linear algebra extras
    "cayley-hamilton theorem", "cayley hamilton",
    "verify cayley", "lu decomposition",
    "nullspace", "kernel of matrix", "null space",
    "orthogonal matrix", "show that a is orthogonal",
    # Number theory extras
    "sieve of eratosthenes", "prime numbers less than",
    "all prime numbers", "find all primes",
    "fermat's little theorem", "fermat little theorem",
    "units digit of", "unit's digit of", "cyclicity of",
    "last two digits of", "last digit of",
    "2^100", "3^2025", "7^100",
    "positive integer solutions", "integer solutions of",
    "always even", "always divisible", "always odd",
    "proof by contradiction",
    # Linear programming extras
    "maximize z =", "minimize z =", "linear programming",
    "subject to constraints", "corner point",
    # Real roots / equation extras
    "real and equal roots", "real equal roots", "real solutions",
    "for what values of k", "discriminant is zero",
    "am-gm inequality", "am-gm",
    # Combinatorics extras
    "6-letter words", "4-letter words", "words can be formed",
    "letters of word", "arrange the letters",
    "6 digit numbers", "4 digit numbers can be",
    # Vector extras
    "scalar projection of", "vector projection of",
    "angles of depression", "angle of depression",
    "rotating about", "rotate about", "rotated about",
    "volume of solid formed", "solid of revolution formed",
    "volume of solid of revolution",
    # Matrix extras
    "matrix ab", "matrix ba", "matrices a and b",
    "find ab, ba", "find ab and ba", "verify ab", "compute ab and ba",
    # Complex number extras
    "complex solutions of", "complex roots of",
    "find all complex", "z^3=", "z^2=",
    "z1=", "z2=", "z1 and z2",
    # Combinatorics extras
    "4-digit numbers", "5-digit numbers", "digit numbers using",
    "digits 1-9", "digits 0 to", "digits 0,",
    "how many 4-digit", "how many 5-digit",
    "number of diagonals", "diagonals in a polygon",
    "triangles that can be formed", "triangles from",
    # Statistics extras
    "karl pearson", "pearson's coefficient", "coefficient of correlation",
    "regression line y on x", "regression equation",
    "quartile deviation", "quartile range",
    # Trigonometry extras  
    "prove that cos(", "prove that sin(",
    "sin(20°)", "cos(36°)", "cos(72°)",
    "circumradius", "inradius", "in-radius",
    "circumradius of triangle", "inradius of triangle",
    "sides 5, 12, 13", "right triangle with",
    "angles are in ratio", "triangle with angles",
    "height of a tower", "height of the tower",
    "angle of elevation of", "angles of elevation",
    # Differential equations extras
    "bernoulli equation", "orthogonal trajectories",
    "solve dy/dx", "solve the equation dy/dx",
    # Maxima/minima word problems - additional
    "inscribed in a circle", "inscribed in a semicircle",
    "inscribed in a sphere", "maximum area that",
    "maximum volume", "minimum surface area", "minimum cost",
    "dimensions of a rectangle", "dimensions of the rectangle",
    "dimensions of a box", "dimensions of a cylinder",
    "dimensions of maximum", "dimensions of minimum",
    "find the point on curve", "point on curve y=",
    "point on line y=", "closest to point", "nearest to point",
    "farmer has", "fencing to enclose",
    # Monotonicity
    "intervals of monotonicity", "monotonicity for f(x)",
    # Mean value theorem
    "lagrange mvt", "apply lagrange", "apply rolle", "verify rolle",
    "apply mvt", "verify lmvt", "apply lmvt", "rolle's theorem",
    "find c in (", "find c that satisfies", "value of c in (",
    # Circles
    "common chord", "angle of intersection of circles",
    # Pair of lines / homogeneous
    "coincident lines", "two coincident", "represents two lines",
    "represents a pair", "combined equation of",
    "homogeneous equation", "homogeneous second degree",
    # Trigonometry
    "as sum or difference", "sum or difference of",
    "product-to-sum", "product to sum formula",
    "in triangle abc", "triangle abc if", "triangle abc with",
    "find all angles", "verify it is a right",
    "triangle is isosceles", "if in a triangle",
    # Matrices
    "inverse of matrix a", "find inverse of matrix",
    "adjoint method", "using adjoint", "find the inverse of",
    # Quadratic / roots
    "alpha and beta are roots", "roots of 2x^2", "roots of x^2",
    "roots of ax^2", "vieta", "vieta's formulas",
    "find all real roots", "real roots of x",
    "equal roots", "real equal roots",
    "condition for roots", "roots in ratio",
    "for what values of k does",
    "range of f(x)", "find the range of",
    "am-gm inequality", "prove am-gm",
    "taylor expansion", "maclaurin expansion",
    "fourier series of", "fourier series",
    "differentiable at x=", "global maximum", "global minimum",
    "wire bent", "wire of length...bent",
    "approximate value at", "differential of y",
    # Sequences
    "arithmetico-geometric", "arithmetico geometric",
    "arithmetic means between", "geometric means between",
    "insert 4", "insert 3", "insert n",
    # Number theory
    "integral solutions of", "always divisible by 6",
    "n^3-n is", "n^2+n+1 is", "always divisible",
    # Statistics
    "find mean, variance", "mean variance and standard",
    "karl pearson", "pearson's correlation",
    "find the regression",
    # Probability
    "two cards are drawn", "3 girls be arranged",
    "boys and girls arranged", "no two girls",
    "committee of", "at least 1 woman", "at least one woman",
    "distribute", "nCr when n=", "verify pascal",
    "probability distribution of white",
    "play cricket", "play football", "play tennis", "play hockey",
    "play both", "only cricket", "only football", "only both",
    "students play", "members play", "people play",
    "who play only", "how many play",
    # Vectors
    "for vectors a=", "find a×b", "a=2i+", "a=3i+",
    "angle between vectors a=",
    "moment of force f=", "moment of force about",
    "moment about point", "torque about point",
    # Complex numbers
    "modulus and argument of z", "polar form of z",
    "if z = (", "find z in the form a+ib",
    "cube roots of unity", "1+ω+ω²=",
    # Conics
    "asymptotes of hyperbola", "equation of hyperbola with",
    "locus of point p equidistant",
    "equation of parabola with", "directrix x=",
    # Distance / angle between
    "angle between planes", "angle between the planes",
    "distance of point p(",
    "angle between lines with direction",
    # Quadratic extras
    "quadratic equation with roots", "equation with roots",
    "with roots 2+sqrt", "with roots alpha", "sum of roots",
    "product of roots",
    # Wire / fence problems
    "wire of length 20", "wire of length 10",
    "bent into a rectangle", "bent into a shape",
    # Chord of circle
    "chord of circle", "chord of the circle",
    "bisected at point", "chord bisected at",
    "common tangents to", "number of common tangents",
    # Sums / series
    "find the sum 1^3", "1^3+3^3", "sum of odd",
    "sum 1^3+", "sum of cubes",
    "pythagorean triples", "pythagorean triple",
    "number of divisors", "divisors of 360",
    "sum of divisors",
    # Induction / modular
    "prove by induction", "mathematical induction",
    "1+2+3+...+n", "sum to n terms by induction",
    "extended euclidean", "x congruent", "congruent to",
    "7x congruent", "5x congruent",
    # Paths / counting
    "paths from a to b", "paths on a grid",
    "5x3 grid", "moving only right or up",
    "integers from 1 to 1000", "integers divisible by",
    "divisible by 3 or 5",
    # Linear algebra extras
    "transforms basis vectors", "basis vectors i=",
    "transformation matrix", "linear transformation",
    # Calculus extras
    "inflection points of f(x)", "inflection point of",
    "points of inflection", "inflection points",
    # Statistics extras
    "standard error of the mean", "standard error",
    "spearman's rank", "rank correlation coefficient",
    "spearman correlation",
    # Sequences extras
    "find the sum 1^2+2^2", "1^2+2^2+3^2",
    "sum of first n squares", "sum of squares of",
    # Miscellaneous math
    "find all pythagorean", "pythagorean",
    "n(n+1)/2 by induction",
    # Polynomial remainder
    "remainder when polynomial", "remainder theorem",
    "polynomial p(x)", "remainder when p(x)",
    "divided by (x+", "divided by (x-",
    "factor theorem", "synthetic division",
    # Roots of cubic
    "alpha, beta, gamma are roots", "alpha beta gamma are roots",
    "roots of x^3-px^2", "roots of x^3",
    "alpha^2+beta^2+gamma^2", "alpha^3+beta^3",
    "vieta's for cubic", "newton's identities",
    # Competition / olympiad math (AIME/AMC-style)
    "relatively prime positive integers",
    "m and n are relatively prime",
    "find m+n",
    "all arrived at the same time",
    "all three people arrived",
    "arrived at the park at the same time",
    "started walking at a constant speed",
    "started running at a constant speed",
    "started bicycling",
    "miles per hour faster than",
    "constant speed along",
    "same straight road",
]


# ─────────────────────────────────────────────────────────────────────────────
# 5. Simple arithmetic
# ─────────────────────────────────────────────────────────────────────────────

MATH_PATTERN = re.compile(
    r"""
    ^           # start of string
    \s*         # optional leading whitespace
    [\d\s\(\)]  # starts with digit, space, or parenthesis
    [\d\s\+\-\*\/\%\^\(\)\.]*  # followed by math characters
    $           # end of string
    """,
    re.VERBOSE,
)

ARITHMETIC_PREFIXES = ("calculate", "compute", "evaluate", "what is")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Knowledge questions
# ─────────────────────────────────────────────────────────────────────────────

KNOWLEDGE_KEYWORDS = [
    "what is", "what are", "who is", "who are", "explain", "define",
    "tell me about", "describe", "how does", "why is", "when was",
    "where is", "history of", "meaning of", "difference between",
    "knowledge", "information about", "learn about", "facts about",
    "how do", "what causes", "why does", "what happened",
]


# ─────────────────────────────────────────────────────────────────────────────
# 7. System commands
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PATTERN = re.compile(r"^/\w+")


# ─────────────────────────────────────────────────────────────────────────────
# Public classifier
# ─────────────────────────────────────────────────────────────────────────────

def classify_intent(user_input: str) -> str:
    """
    Classify user input and return one of:
        'system' | 'logic' | 'physics' | 'chemistry' |
        'advanced_math' | 'math' | 'knowledge' | 'conversation'

    Priority:
        system > logic > conceptual_knowledge > physics > chemistry
               > advanced_math > math > knowledge > conversation

    Input is normalized via preprocess.normalize_input() before matching
    so that Unicode symbols, arrows, superscripts etc. are handled uniformly.
    """
    # ── 0. Normalize (Unicode, arrows, superscripts, delta signs…) ────────────
    text    = normalize_input(user_input)
    lowered = text.lower()

    # ── 1. System commands (/exit, /help, /history, /clear) ───────────────────
    if SYSTEM_PATTERN.match(text):
        return "system"

    # ── 1.5. Pre-logic advanced math override — C(n,r)/nCr must not be captured by "if...then" ──
    _PRE_LOGIC_MATH_PATTERNS = [
        re.compile(r'C\s*\(\s*n\s*,\s*\d+\s*\)\s*=\s*\d+', re.I),
        re.compile(r'\bnC\d+\s*=\s*\d+|\b\d+C\d+\b', re.I),
        re.compile(r'\bC\s*\(\s*\d+\s*,\s*\d+\s*\)', re.I),
        re.compile(r'\^2\+[a-z]\^2\s*=|\bx\^2\+y\^2', re.I),
        re.compile(r'\b\d+x\^2\s*[+\-]\s*\d+xy\s*[+\-]\s*\d+y\^2\s*=\s*0', re.I),
    ]
    for pat in _PRE_LOGIC_MATH_PATTERNS:
        if pat.search(text):
            return "advanced_math"

    # ── 2. Logic — structural syntax patterns (BEFORE everything) ─────────────
    for pat in LOGIC_STRUCTURAL_PATTERNS:
        if pat.search(text):
            return "logic"

    # ── 2.5. Conceptual knowledge override ────────────────────────────────────
    # "What is Newton's second law?", "Define entropy" → knowledge, NOT physics.
    # Runs AFTER logic (tautology, syllogism) but BEFORE physics/chemistry so
    # conceptual questions don't get swallowed by keyword matching.
    if _is_conceptual_question(lowered):
        return "knowledge"

    # ── 3. Physics — formula-based physics problems (BEFORE chemistry) ─────────
    if _is_physics(text, lowered):
        return "physics"

    # ── 4. Chemistry — quantitative chemistry calculations ────────────────────
    if _is_chemistry(text, lowered):
        return "chemistry"

    # ── 5. Advanced math — symbolic operations + word problems ────────────────
    for kw in ADVANCED_MATH_KEYWORDS:
        if kw in lowered:
            return "advanced_math"

    # ── 6. Simple arithmetic ──────────────────────────────────────────────────
    if _is_simple_arithmetic(text, lowered):
        return "math"

    # ── 7. Knowledge questions ────────────────────────────────────────────────
    for kw in KNOWLEDGE_KEYWORDS:
        if kw in lowered:
            return "knowledge"

    # ── 8. Fallback — conversation / LLM ─────────────────────────────────────
    return "conversation"


def _is_simple_arithmetic(text: str, lowered: str) -> bool:
    """True if the input is a plain numeric arithmetic expression."""
    remainder = lowered
    for prefix in ARITHMETIC_PREFIXES:
        if lowered.startswith(prefix):
            remainder = lowered[len(prefix):].strip()
            break

    has_digit    = any(ch.isdigit() for ch in remainder)
    has_operator = any(ch in "+-*/%^" for ch in remainder)
    has_letters  = bool(re.search(r'[a-zA-Z]', remainder))

    if has_digit and has_operator and not has_letters:
        return True

    return bool(MATH_PATTERN.match(text))
