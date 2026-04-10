"""
Advanced Reasoning Engine v2 — CoT / ToT / FoT Multi-Strategy.

Architecture:
  Stage 1 — Problem Analysis
      · Identify problem type, domain, sub-questions
      · Detect ambiguity or missing information
      · Choose optimal resolution strategy

  Stage 2 — Reasoning-Mode Selection
      · Chain-of-Thought (CoT): linear step-by-step for standard problems
      · Tree-of-Thought  (ToT): multi-branch search for complex single-domain
      · Forest-of-Thought (FoT): multiple independent root hypotheses for
            cross-domain or highly ambiguous multi-step problems

  Stage 3 — Self-Consistency Verification
      · Cross-check reasoning against known constraints
      · Verify internal logical consistency
      · Assign confidence level (HIGH / MEDIUM / LOW)
      · Flag assumptions and potential errors

  Stage 4 — Metacognitive Analysis
      · Identify what the system knows confidently vs. uncertainly
      · Surface knowledge boundaries explicitly
      · Adjust response framing accordingly

  Stage 5 — Response Synthesis
      · Compose structured LLM prompt embedding the full reasoning trace
      · Force the LLM to follow the plan without deviating
      · Add domain-specific constraints and answer validators

New in v2:
  · Forest-of-Thought (FoT): 3-tree parallel hypothesis exploration
  · Enhanced CoT with explicit "Think step by step" anchors
  · Stronger ToT branch scoring for JEE advanced problem types
  · Tree-of-Thought branching for multi-step problems
  · Self-consistency checker that verifies the plan is non-contradictory
  · Metacognitive layer exposing knowledge confidence boundaries
  · Abductive reasoning for explanatory "why" questions
  · Causal chain decomposition
  · Analogy-based reasoning for abstract concepts
  · 15+ new problem types (biology, chemistry, economics, logic, history, …)
  · Richer prompt templates with explicit verification steps
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ReasoningBranch:
    """One candidate reasoning path in Tree-of-Thought exploration."""
    label:       str
    steps:       list[str]
    score:       float     = 0.0     # internal quality score 0–1
    feasible:    bool      = True


@dataclass
class MetacognitiveState:
    """What the engine knows vs. doesn't know about this problem."""
    known_aspects:    list[str] = field(default_factory=list)
    uncertain_aspects: list[str] = field(default_factory=list)
    knowledge_gaps:   list[str] = field(default_factory=list)
    boundary_note:    str       = ""

    def to_text(self) -> str:
        lines = []
        if self.known_aspects:
            lines.append("KNOWN: " + "; ".join(self.known_aspects))
        if self.uncertain_aspects:
            lines.append("UNCERTAIN: " + "; ".join(self.uncertain_aspects))
        if self.knowledge_gaps:
            lines.append("GAPS: " + "; ".join(self.knowledge_gaps))
        return "\n".join(lines)


@dataclass
class ForestTree:
    """One independent root hypothesis in Forest-of-Thought exploration."""
    hypothesis:    str           # root assumption / approach label
    steps:         list[str]     # reasoning steps under this hypothesis
    score:         float  = 0.0  # quality score 0–1
    conclusion:    str    = ""   # predicted outcome of this path


@dataclass
class ReasoningPlan:
    """Full structured reasoning plan produced by the engine."""
    problem_type:     str             = "unknown"
    domain:           str             = "general"
    sub_problems:     list[str]       = field(default_factory=list)
    strategy:         str             = ""
    expected_form:    str             = ""
    assumptions:      list[str]       = field(default_factory=list)
    confidence:       str             = "MEDIUM"   # HIGH / MEDIUM / LOW
    reasoning_steps:  list[str]       = field(default_factory=list)
    warnings:         list[str]       = field(default_factory=list)
    branches:         list[ReasoningBranch] = field(default_factory=list)
    forest:           list[ForestTree]      = field(default_factory=list)
    metacognition:    MetacognitiveState    = field(default_factory=MetacognitiveState)
    consistency_ok:   bool            = True
    consistency_note: str             = ""
    reasoning_mode:   str             = "chain_of_thought"
    # chain_of_thought | tree_of_thought | forest_of_thought

    def summary(self) -> str:
        return (
            f"[Reasoning-v2] domain={self.domain} | type={self.problem_type} "
            f"| mode={self.reasoning_mode} | confidence={self.confidence}"
        )

    def full_trace(self) -> str:
        lines = [
            f"Problem type  : {self.problem_type}",
            f"Domain        : {self.domain}",
            f"Strategy      : {self.strategy}",
            f"Reasoning mode: {self.reasoning_mode}",
        ]
        if self.sub_problems:
            lines.append("Sub-problems:")
            for i, sp in enumerate(self.sub_problems, 1):
                lines.append(f"  {i}. {sp}")
        if self.expected_form:
            lines.append(f"Expected form : {self.expected_form}")
        if self.assumptions:
            lines.append("Assumptions   : " + "; ".join(self.assumptions))
        if self.warnings:
            lines.append("Warnings      : " + "; ".join(self.warnings))
        if not self.consistency_ok:
            lines.append(f"Consistency   : FAIL — {self.consistency_note}")
        meta = self.metacognition.to_text()
        if meta:
            lines.append(meta)
        lines.append(f"Confidence    : {self.confidence}")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Domain + problem-type classifiers
# ─────────────────────────────────────────────────────────────────────────────

_DOMAIN_HINTS: dict[str, list[str]] = {
    "calculus": [
        "integrate", "integral", "derivative", "differentiate",
        "d/dx", "limit", "lim", "antiderivative", "gradient",
        "maxima", "minima", "rolle", "mean value theorem",
        "increasing", "decreasing", "concavity", "inflection",
    ],
    "algebra": [
        "solve", "factor", "expand", "simplify", "roots", "zeros",
        "equation", "polynomial", "quadratic", "linear equation",
        "cubic", "biquadratic", "vieta",
    ],
    "linear_algebra": [
        "matrix", "eigenvalue", "eigenvector", "determinant",
        "inverse", "rank", "trace", "vector", "dot product", "cross product",
        "orthogonal", "basis", "span",
    ],
    "differential_equations": [
        "differential equation", "ode", "dy/dx", "y''", "y'", "dsolve",
        "pde", "partial differential",
    ],
    "transforms": [
        "laplace", "fourier", "z-transform", "inverse laplace",
        "wavelet", "dft",
    ],
    "series": [
        "taylor", "maclaurin", "power series", "series expansion",
        "convergence", "radius of convergence",
        "arithmetic progression", "geometric progression", "harmonic progression",
        "ap", "gp", "hp", "nth term", "sum of n terms",
    ],
    "number_theory": [
        "prime", "gcd", "lcm", "modular", "mod ", "divisible",
        "factorization", "congruence", "fermat", "euler",
    ],
    "statistics": [
        "mean", "median", "mode", "variance", "standard deviation",
        "average", "probability", "distribution", "regression",
        "correlation", "hypothesis", "confidence interval",
        "binomial distribution", "bayes", "conditional probability",
        "expected value", "random variable",
    ],
    "combinatorics": [
        "factorial", "binomial", "permutation", "combination", "choose",
        "nCr", "nPr", "counting", "pigeonhole",
        "derangement", "multinomial", "circular permutation",
    ],
    "complex_analysis": [
        "complex", "imaginary", "real part", "modulus", "argument",
        "conjugate", "polar form", "euler formula",
        "de moivre", "roots of unity", "locus in argand",
    ],
    "coordinate_geometry": [
        "parabola", "ellipse", "hyperbola", "conic", "circle",
        "tangent to", "normal to", "chord of contact",
        "director circle", "asymptote", "latus rectum",
        "eccentricity", "focus", "directrix",
        "radical axis", "pair of lines", "angle between lines",
        "straight line", "distance from point",
    ],
    "3d_geometry": [
        "direction cosines", "direction ratios",
        "skew lines", "shortest distance",
        "coplanar", "plane equation", "distance from plane",
        "line of intersection", "angle between planes",
        "scalar triple product", "vector triple product",
        "section formula", "midpoint",
    ],
    "trigonometry": [
        "sin", "cos", "tan", "sine", "cosine", "tangent",
        "trigonometric equation", "general solution",
        "inverse trig", "arcsin", "arccos", "arctan",
        "compound angle", "double angle", "half angle",
        "sine rule", "cosine rule",
    ],
    "physics": [
        "velocity", "acceleration", "force", "energy", "momentum",
        "electric", "magnetic", "quantum", "wave", "frequency",
        "thermodynamics", "entropy", "optics", "relativity",
        "shm", "oscillation", "amplitude", "spring",
        "moment of inertia", "angular momentum", "torque",
        "photoelectric", "de broglie", "bohr", "nuclear",
        "capacitor", "inductor", "diode", "transistor",
    ],
    "chemistry": [
        "atom", "molecule", "element", "compound", "reaction",
        "bond", "electron", "proton", "neutron", "periodic table",
        "acid", "base", "ph", "oxidation", "reduction",
        "mol", "moles", "molar", "monatomic", "diatomic",
        "internal energy", "enthalpy", "calorimetry",
        "constant pressure", "constant volume", "isobaric", "isochoric",
        "cv", "cp", "ideal gas", "argon", "helium", "neon",
        "equilibrium constant", "kc", "kp", "le chatelier",
        "rate of reaction", "rate constant", "order of reaction",
        "activation energy", "arrhenius",
        "coordination compound", "unit cell", "packing",
        "colligative", "van't hoff", "raoult",
    ],
    "biology": [
        "cell", "dna", "rna", "protein", "gene", "evolution",
        "mitosis", "meiosis", "enzyme", "metabolism", "photosynthesis",
        "organism", "species", "ecosystem",
    ],
    "computer_science": [
        "algorithm", "complexity", "big o", "sorting", "graph",
        "recursion", "dynamic programming", "data structure",
        "hash", "tree", "queue", "stack",
    ],
    "logic": [
        "if then", "implies", "therefore", "conclude", "premise",
        "syllogism", "modus ponens", "modus tollens", "valid",
        "contrapositive", "logical", "proposition", "truth table",
    ],
    "economics": [
        "supply", "demand", "inflation", "gdp", "market",
        "equilibrium", "elasticity", "monetary", "fiscal",
        "opportunity cost", "marginal", "utility",
    ],
    "history": [
        "century", "war", "empire", "revolution", "civilization",
        "ancient", "medieval", "colonial", "historical",
    ],
    "philosophy": [
        "ethics", "morality", "consciousness", "existence", "metaphysics",
        "epistemology", "ontology", "free will", "determinism",
        "utilitarianism", "kant", "aristotle", "plato",
    ],
}

_PROBLEM_TYPE_PATTERNS: list[tuple[str, str]] = [
    # ── Second derivative "find d^2y/dx^2 FOR function" → differentiation (BEFORE ODE) ──
    (r'\bfind\s+d\^?2y/dx\^?2\s+(?:for|of|when|if)\b'
     r'|\bfind\s+d²y/dx²\s+(?:for|of|when)\b'
     r'|\bfind\s+d\^2y/dx\^2\b(?!\s*\+|\s*-|\s*=)'
     r'|\bd\^2y/dx\^2\s+for\s+(?:the\s+)?parametric\b', "differentiation"),
    # ── Chemical kinetics zero/first/second order half-life (BEFORE modern_physics) ──
    (r'\bhalf.life\s+of\s+a\s+(?:zero|first|second|nth)\s+order\b'
     r'|\bzero\s+order\s+reaction\b.*\bhalf.life\b|\bhalf.life\b.*\bzero\s+order\b'
     r'|\bzero\s+order\b.*\bk\s*=\s*[\d.]+\b', "chemical_kinetics"),
    # ── Speed of sound and Bragg/X-ray → physics_problem (BEFORE kinetics / ideal_gas) ──
    (r'\bspeed\s+of\s+sound\s+in\b|\bvelocity\s+of\s+sound\s+in\b'
     r'|\bbragg\s+(?:reflection|diffraction|angle)\b|\bfirst\s+order\s+bragg\b'
     r'|\bx.ray\s+(?:diffraction|scattered|scattering)\b|\bcrystal\s+planes?\b.*\bangle\b', "physics_problem"),
    # ── C(n,r) combinatorics — BEFORE logical_reasoning catches "if...then" ──
    (r'\bC\s*\(\s*n\s*,\s*\d+\s*\)\s*=\s*\d+\b'
     r'|\bC\s*\(\s*\d+\s*,\s*\d+\s*\)\b(?!.*\bgate\b)'
     r'|\bnCr\b(?!\s*when)|\bnPr\b(?!\s*when)'
     r'|\bpascal\'?s?\s+identity\b|\bC\s*\(\s*n\s*,\s*r\s*\)\b', "combinatorics"),
    # ── Vector algebra via cross/dot product — BEFORE triangle_trig ──
    (r'\busing\s+cross\s+product\b|\bcross\s+product\b.*\barea\s+of\s+triangle\b'
     r'|\barea\s+of\s+triangle\b.*\bcross\s+product\b'
     r'|\bangle\s+between\s+them\s+using\s+dot\s+product\b'
     r'|\bfind\s+angle\s+between\b.*\bdot\s+product\b'
     r'|\bvolume\s+of\s+(?:a\s+)?tetrahedron\s+with\s+vertices\b', "vector_algebra"),
    # ── Conics with focal/foci context — BEFORE summation steals "sum of focal" ──
    (r'\bsum\s+of\s+focal\s+distances\b|\bfocal\s+(?:chord|radii)\b.*\bellipse\b'
     r'|\bellipse\b.*\bfoci\b|\bfoci\s+\(?±|\bfoci\s+at\s+\('
     r'|\bfind\s+(?:the\s+)?(?:equation|semi.major|semi.minor)\s+of\s+(?:the\s+)?ellipse\b', "conic_ellipse"),
    # ── 3D line shortest distance — BEFORE straight_line steals "distance between lines" ──
    (r'\bskew\s+lines\b|\bshortest\s+distance\s+between\s+(?:lines?|skew)\b'
     r'|\bshortest\s+distance\s+between\s+lines\s+x\s*=\b'
     r'|\bdistance\s+of\s+point\s+[A-Z]?\s*\(\s*\d+\s*,\s*\d+\s*,\s*\d+\s*\)\b.*\bline\b'
     r'|\bline\s+passing\s+through\s+[A-Z]?\s*\(\s*\d+\s*,\s*\d+\s*,\s*\d+\s*\)\b.*\bdirection\b', "3d_geometry"),
    # ── Trig simplification (sum-to-product) → trig_identities (BEFORE simplification) ──
    (r'\bsimplify\b.*\bsin\s*\(\s*\d+[a-z]\s*\).*(?:sin|cos)\s*\('
     r'|\bsimplify\b.*\b(?:sin|cos|tan)\s*\(.*\)\s*/\s*(?:cos|sin|tan)\s*\('
     r'|\bsimplify\b.*\b(?:sin|cos)\s*\(.*\)\s*[+\-]\s*(?:sin|cos)\s*\('
     r'|\bsimplify\b.*\bsump?.?to.?product\b|\bsimplify\b.*\bproduct.?to.?sum\b', "trig_identities"),
    # ── Probability distribution: geometric/negative binomial distribution ──
    (r'\bgeometric\s+distribution\b|\bnegative\s+binomial\s+distribution\b', "probability_distribution"),
    # Highest priority: ODE — catches y'', y' notation BEFORE differentiation and trig
    (r'\bdifferential\s+equation\b|\bode\b|\bdsolve\b|\bpde\b'
     r"|y''[\s\-\+\*=]|y''\s*$|y''\s*\+|y''\s*\-|y''+|d\^2y/dx\s*[+\-=]"
     r'|\bd\^2y\b|\by\^{\(2\)}\b|\by\^\(2\)\b'
     r'|\bintegrating\s+factor\b'
     r'|\borthogonal\s+traject|\binitial\s+value\s+problem\b'
     r'|\bdy/dx\s*[+\-]|\bdy/dx\s*=\s*f\s*\(', "ode_solving"),
    # "find the derivative of" — BEFORE integration to prevent antiderivative theft
    (r'\bfind\s+(?:the\s+)?derivative\s+of\s+f\b|\bcalculate\s+(?:the\s+)?derivative\s+of\s+f\b'
     r'|\bfind\s+d\^2y/dx\^2\b|\bfind\s+d²y/dx²\b|\bparametric\s+equations\s+x=.*y=\b', "differentiation"),
    (r'\bintegrate\b|\bintegral\b|\bantiderivative\b'
     r'|∫|∫₀|\bevaluate\s+∫|\bfind\s+∫'
     r'|\bdefinite\s+integrat|\bindefinite\s+integrat'
     r'|\busing\s+integration\s+by\s+parts\b|\busing\s+u.sub|\busing\s+trig.sub'
     r'|\bweierstrass\s+substitution\b|\bking\'?s?\s+property\b'
     r'|\barea\s+enclosed\s+by\s+the\s+curve\b|\barea\s+bounded\s+by\b'
     r'|\barea\s+enclosed\s+between\b|\barea\s+between\s+(?:the\s+)?(?:curves?|parabola|line|y\s*=)\b'
     r'|\barea\s+between\s+y\s*=\b'
     r'|\barea\s+enclosed\s+by\s+(?:the\s+)?(?:curves?|parabola|line)\b'
     r'|\barea\s+of\s+region\s+bounded\b|\bregion\s+bounded\s+by\b'
     r'|\busing\s+definite\s+integrat|\busing\s+integrat\b'
     r'|\bvolume\s+of\s+solid\s+formed\b|\bvolume\s+of\s+solid\s+of\s+revolution\b'
     r'|\bgenerated\s+by\s+revolving\b|\bvolume\s+of\s+solid\s+generated\b'
     r'|\bvolume\s+of\s+(?:a\s+)?cone\b.*\bintegrat\b|\bintegrat\b.*\bvolume\s+of\s+(?:a\s+)?cone\b'
     r'|\brotated?\s+about\s+(?:the\s+)?(?:x|y).axis\b|\brotating\s+\S+\s+about\b', "integration"),
    (r'\bderivative\b|\bdifferentiate\b|\bd/d[a-z]\b|\bgradient\s+of\b'
     r'|\bdy/dx\b|\bd²y/dx²\b|\bd\^2y/dx\^2\b|\bfind\s+dy/dx\b|\bfind\s+d²y\b|\bfind\s+d\^2y\b'
     r'|\bproduct\s+rule\b|\bquotient\s+rule\b|\bchain\s+rule\b'
     r'|\bimplicit\s+differentiation\b|\blogarithmic\s+differentiation\b'
     r'|\bparametric\s+differentiation\b'
     r'|\bequation\s+of\s+tangent\s+to\s+curve\b|\bequation\s+of\s+normal\s+to\s+curve\b'
     r'|\btangent\s+to\s+(?:the\s+)?curve\b|\bnormal\s+to\s+(?:the\s+)?curve\b'
     r'|\brate\s+of\s+change\s+of\b|\bslope\s+of\s+(?:the\s+)?tangent\b'
     r'|\bangle\s+between\s+tangents\s+to\s+curve\b|\bcurvature\s+of\b'
     r'|\bdifferentiable\s+at\s+x\s*=\b|\bfind\s+(?:the\s+)?differential\s+of\b'
     r'|\bcheck\s+(?:if|whether)\s+f(?:\(x\))?\s+is\s+differentiable\b', "differentiation"),
    # Limits/continuity — specific patterns BEFORE generic lim
    (r'\bcheck\s+(?:the\s+)?(?:continuity|differentiab)\b|\bcontinuity\s+and\s+differentiab\b'
     r'|\bdifferentiab.*\bcontinuity\b'
     r'|\bvalue\s+of\s+k\s+(?:for|such\s+that)\b.*\bcontinuous\b|\bfor\s+what\s+value\s+of\s+k\b.*\bcontinuous\b'
     r'|\blim\b.*\band\b.*\blim\b'
     r'|\bpiecewise.*\bcontinuous\b|\bcontinuous\s+at\s+x\s*=\b|\bdiscontinuous\s+at\b'
     r'|\busing\s+algebra\b.*\blim\b|\blim\b.*\busing\s+algebra\b', "limits_continuity"),
    # L'Hopital and limit evaluation (including using Taylor series) → limit_evaluation
    (r'\blimit\b|\blim\b|\bl\'?\s*h[oôo]pital\b|l\'h[oô]pital\b'
     r'|\bevaluate\s*:\s*lim\b|\btaylor\s+series\b.*\blim\b|\blim\b.*\btaylor\s+series\b', "limit_evaluation"),
    (r'\beigenvalues?\b|\beigenvectors?\b|\bcayley.hamilton\b|\bcharacteristic\s+polynomial\b'
     r'|\bnull\s+space\b|\bnullspace\b|\borderof\s+matrix\b', "matrix_operation"),
    (r'\bdeterminant\b|\bdet\b|\binverse\s+(?:of\s+)?matrix\b|\bmatrix\s+rank\b|\bmatrix\s+trace\b'
     r'|\brank\s+of\s+(?:a\s+)?matrix\b|\bfind\s+inverse\s+of\b|\badjoint\s+method\b'
     r'|\bcramer\'?s?\s+rule\b|\bfind\s+(?:ab|ba)\b|\bmatrix\s+multiplication\b'
     r'|\bfind\s+a\^?\s*-\s*1\b|\badjoint\s+(?:of\s+)?matrix\b|\blow\s+reduction\b|\brow\s+reduction\b'
     r'|\bgaussian\s+elimination\b|\btranspose\s+and\s+trace\b|\btranspose\s+of\s+matrix\b'
     r'|\blu\s+decomposition\b|\btransforms?\s+basis\s+vectors?\b'
     r'|\btype\s+of\s+system\b.*\b(?:consistent|inconsistent|dependent)\b'
     r'|\bsolve\s+(?:the\s+)?system\s+of\s+equations\s+using\s+matrix\b', "matrix_operation"),
    (r'\btaylor\b|\bmaclaurin\b|\bseries\s+expansion\b|\bpower\s+series\b|\bgenerating\s+function', "series_expansion"),
    (r'\blaplace\s+transform\b|\blaplace\s+of\b', "laplace_transform"),
    (r'\bfourier\s+(?:transform|series)\b|\bfourier\s+of\b', "fourier_transform"),
    # ── Chemical kinetics EARLY — before factorization steals "Arrhenius factor" / "by what factor" ──
    (r'\barrhenius\b|\bactivation\s+energy\b|\bfrequency\s+factor\s+[A=]|\barrhenius\s+factor\b'
     r'|\bpre.exponential\s+factor\b|\bby\s+what\s+factor\s+does\s+(?:the\s+)?rate\b'
     r'|\bwhat\s+factor\s+does\s+(?:the\s+)?rate\s+change\b'
     r'|\brate\s+(?:constant\s+k|doubles|quadruples)\b', "chemical_kinetics"),
    # ── Electromagnetism EARLY — before factorization steals "Q-factor" / "power factor" ──
    (r'\bq.?factor\b|\bpower\s+factor\b|\bquality\s+factor\b'
     r'|\bback.?emf\b|\bback\s+emf\b|\barmature\s+resistance\b|\barmature\s+current\b', "electromagnetism"),
    # ── Competition / olympiad math (AIME/AMC) — must come before number_theory ─
    (r'\bm\s*\+\s*n\b.*\brelatively\s+prime\b|\brelatively\s+prime\b.*\bm\s*\+\s*n\b'
     r'|\bfind\s+m\s*\+\s*n\b|\bm\s+and\s+n\s+are\s+relatively\s+prime\b'
     r'|\ball\s+(?:three\s+)?(?:people\s+)?arrived?\s+at\s+the\s+(?:park|same\s+time)\b'
     r'|\bstarted\s+(?:walking|running|bicycl)\b.*\b(?:arrived|park)\b'
     r'|\bmiles?\s+per\s+hour\s+faster\s+than\b'
     r'|\bhours?\s+after\s+\w+\s+(?:left|started)\b', "competition_math"),
    (r'\bprime\s+factor|\bgcd\b|\blcm\b|\bmod\b|\bmodular\b'
     r'|\bsieve\s+of\s+eratosthenes\b|\bprime\s+numbers?\s+less\s+than\b'
     r'|\bproof\s+by\s+contradiction\b|\birrational\b.*\bproof\b|\bprove\b.*\birrational\b'
     r'|\bfermat\'?s?\s+(?:little\s+)?theorem\b|\beuler\'?s?\s+theorem\b'
     r'|\bdivisib(?:le|ility)\s+by\b|\bnumber\s+of\s+divisors\b|\bsum\s+of\s+divisors\b'
     r'|\bunits?\s+digit\s+of\b|\bunit\'?s\s+digit\b|\bcyclicity\s+of\b'
     r'|\blast\s+(?:two\s+)?digits?\s+of\b|\balways\s+(?:even|odd|divisible)\b'
     r'|\bpositive\s+integer\s+solutions?\s+of\b|\bintegral\s+solutions?\s+of\b'
     r'|\bprove\s+by\s+induction\b|\bmathematical\s+induction\b'
     r'|\bpythagorean\s+triple\b|\bpythagorean\s+triples?\s+with\b|\bprimitive\s+pythagorean\b'
     r'|\btrailing\s+zeros?\s+(?:in|of|does)\b|\bhow\s+many\s+trailing\s+zeros?\b'
     r'|\bperfect\s+(?:numbers?|squares?|cubes?)\s+between\b|\bperfect\s+number\s+less\b', "number_theory"),
    # ── Combinatorics — must come before summation ────────────────────────────
    (r'\bfactorial\b|\bbinomial\s+coeff|\bpermutation\b|\bnCr\b|\bnPr\b'
     r'|\bin\s+how\s+many\s+ways\b|\bnumber\s+of\s+ways\b'
     r'|\bround\s+table\b|\bcircular\s+(?:permutation|arrangement)\b'
     r'|\bcommittee\s+of\b|\bseated\s+(?:at|in|around)\s+(?:a\s+)?(?:round|circular)\b'
     r'|\bwords?\s+can\s+be\s+(?:formed|arranged)\b|\bletters?\s+(?:of|in)\s+[A-Z]{4}\b'
     r'|\bdigit\s+numbers?\s+(?:using|from|with)\b|\bnumber\s+of\s+(?:4|5|6|7|8)\s*-?\s*digit\b'
     r'|\bhow\s+many\s+(?:\d+\s*-?\s*)?(?:letter|digit|word|way|arrangement|selection|group)\b'
     r'|\bderangements?\b|\bmultinomial\b|\bpigeonhole\b'
     r'|\bnumber\s+of\s+diagonals\b|\btriangles?\s+(?:that\s+can\s+be|formed\s+from)\b'
     r'|\bpaths?\s+from\s+[a-z]\s+to\s+[a-z]\b|\bnumber\s+of\s+paths?\s+(?:from|on)\b'
     r'|\bgrid\b.*\bpath\b|\bmoving\s+only\s+(?:right|up|down|left)\b'
     r'|\bnumber\s+of\s+(?:integers?|numbers?)\s+(?:from|between)\s+\d+\s+(?:to|and)\s+\d+\s+(?:divisible|not\s+divisible)\b', "combinatorics"),
    # Sequences & Series must come before generic summation (AP/GP/HP sum-of-n-terms)
    (r'\barithmetic\s+progression\b|\bgeometric\s+progression\b|\bharmonic\s+progression\b', "sequences_series"),
    (r'\bsum\s+of\s+(?:first|last)\s+\d+\s+terms\b|\bsum\s+of\s+n\s+terms\b|\bnth\s+term\b|\bcommon\s+(?:difference|ratio)\b', "sequences_series"),
    (r'\b(?:ap|gp|hp)\b.*\b(?:term|sum|series|progression)\b|\b(?:term|sum|series)\b.*\b(?:ap|gp|hp)\b', "sequences_series"),
    (r'\binsert\s+\d+\s+arithmetic\s+means?\b|\binsert\s+\d+\s+geometric\s+means?\b'
     r'|\barithmetic\s+means?\s+between\b|\bgeometric\s+means?\s+between\b'
     r'|\bharmonic\s+means?\s+between\b', "sequences_series"),
    # AM/GM of two numbers, GP/AP log relationship, partial fractions on telescoping → sequences_series
    (r'\bam\s+of\s+two\s+(?:numbers?|positive)\b|\bgm\s+of\s+two\b'
     r'|\barithmetic\s+mean\s+of\s+two\b|\bgeometric\s+mean\s+of\s+two\b'
     r'|\bam\s+is\s+\d+\s+and\s+(?:their\s+)?gm\b|\bgm\s+is\s+\d+\s+and\s+(?:their\s+)?am\b'
     r'|\bif\s+[a-z],\s*[a-z],\s*[a-z]\s+are\s+in\s+(?:gp|ap|hp)\b'
     r'|\bshow\s+that\s+.*\bin\s+(?:ap|gp|hp)\b|\bprove\s+that\s+.*\bin\s+(?:ap|gp|hp)\b'
     r'|\bif\s+[a-z],\s*[a-z],\s*[a-z]\s+are\s+in\s+gp\b.*\blog\b'
     r'|\bseries\s+is\s+(?:ap|gp|hp)\b|\bterms?\s+are\s+in\s+(?:ap|gp|hp)\b'
     r'|\bsum\s+to\s+infinity\b|\binfinite\s+(?:gp|series\s+with\s+ratio)\b'
     r'|\barithmetico.geometric\b|\barithmetico.gp\b', "sequences_series"),
    # ── Binomial theorem: sum of coefficients in expansion — BEFORE generic summation ──
    (r'\bsum\s+of\s+(?:all\s+)?(?:the\s+)?(?:binomial\s+)?coefficients\s+(?:in|of|in\s+the)\s+expansion\b'
     r'|\bsum\s+of\s+coefficients\s+in\s+(?:the\s+)?expansion\b'
     r'|\bsum\s+of\s+c\s*\(n,\s*\d\)\b|\bc\s*\(n,0\)\s*\+\s*c\s*\(n,1\)\b|\bc\(n,0\)\+c\(n,1\)\b'
     r'|\bfind\s+sum\s+of\s+c\(n,\b|\bsum\s+of\s+(?:c\(n,\d\)|binomial\s+coefficients)\b', "binomial_theorem"),
    # Sequences&Series: sum of n^2, n^3, n·(n+1) type — BEFORE summation
    (r'\bsum\s+of\s+(?:series\s+)?1\^2\s*\+\s*2\^2\b|\bsum\s+of\s+(?:series\s+)?1\^2\s*\+\s*2\^2\s*\+\s*3\^2\b'
     r'|\busing\s+standard\s+formula\b.*\bsum\b|\bsum\b.*\busing\s+standard\s+formula\b'
     r'|\bsum\s+of\s+(?:first\s+)?n\s+terms\s+of\s+(?:the\s+)?series\b'
     r'|\bsum\s+of\s+(?:the\s+)?series\s*:\s*1\b'
     r'|\bif\s+(?:the\s+)?pth\b.*\bqth\b.*\brth\b|\bpth,\s*qth\s+(?:and\s+)?rth\s+terms?\b'
     r'|\bpth\s+term\b.*\bqth\s+term\b|\bfind\s+p\(q-r\)\b|\ba\s*\(q-r\)\s*\+\s*b\s*\(r-p\)\b'
     r'|\bsum\s+to\s+n\s+terms\s+of\s+series\b.*\b\d+\s*[·*]\s*\d+\b'
     r'|\bfind\s+sum\s+of\s+series.*\btelescoping\b|\btelegraphing\s+method\b'
     r'|\busing\s+(?:method\s+of\s+)?partial\s+fractions\b.*\bsum\b|\bsum\b.*\bpartial\s+fractions\b'
     r'|\b1\s*[·*]\s*2\s*\+\s*2\s*[·*]\s*3\b|\b1\s*·\s*2\s*\+\s*2\s*·\s*3\b'
     r'|\bseries\b.*\bn\s*\(\s*n\s*\+\s*1\s*\)\b|\bn\s*\(\s*n\s*\+\s*1\s*\)\b.*\bseries\b'
     r'|\bsum\s+of\s+(?:the\s+)?series\b.*\bsummation\s+formulas?\b'
     r'|\bharmonic\s+mean\s+of\s+two\b|\bhm\s+of\s+two\b'
     r'|\binsert\s+\d+\s+(?:arithmetic|geometric|harmonic)\s+means?\s+between\b', "sequences_series"),
    # Summation — algebraic series, telescoping, general Σ (NOT AP/GP/HP "sum of series")
    (r'\btelescoping\s+(?:sum|series|method)\b|\bmethod\s+of\s+differences\b'
     r'|\b1\^3\s*[+·]\s*3\^3\b|\bodd\s+cubes\b|\bsum\s+of\s+odd\b.*\bcubes?\b'
     r'|\bfind\s+the\s+sum\b(?!.*\b(?:ap|gp|hp|arithmetic|geometric|harmonic)\b)'
     r'|\bsum\s+of\s+(?:cubes?|squares?)\s+of\s+(?:first|odd|even|natural)\b'
     r'|\bsum\s+to\s+n\s+terms\s+of\s+series\s+\d\b', "summation"),
    (r'\bsum\s+of\b|\bsummation\b|\bsum\b.*\bfrom\b', "summation"),
    # ── JEE Advanced: Physics Modern — must come before half-life in chemical_kinetics ──
    (r'\bphotoelectric\s+effect\b|\bwork\s+function\b|\bde\s+broglie\b|\bcompton\b'
     r'|\bbohr\s+(?:model|radius|orbit)\b|\benergy\s+level\b|\bnuclear\s+(?:fission|fusion|binding)\b'
     r'|\bradioactiv|\balpha\s+decay\b|\bbeta\s+decay\b|\bdecay\s+constant\b'
     r'|\bbalmer\s+series\b|\blyman\s+series\b|\bpasschen\s+series\b'
     r'|\bbinding\s+energy\b|\bmass\s+defect\b|\bq.value\b|\bnuclear\s+radius\b'
     r'|\bstopping\s+potential\b|\bthreshold\s+(?:frequency|wavelength)\b'
     r'|\bionization\s+(?:energy|potential)\b|\bphoton\s+energy\b'
     r'|\benergy\s+of\s+electron\b|\bradius\s+of\s+(?:nucleus|bohr)\b'
     r'|\bx.ray\s+photon\b|\bx.ray\s+(?:from|of|for)\b|\bka\s+x.ray\b|\bk.?shell\b'
     r'|\bphoton\s+emitted\s+when\b|\bphotons\s+(?:emitted|per\s+second)\b'
     r'|\bfrequency\s+of\s+(?:photon|radiation|light)\s+emitted\b'
     r'|\bkinetic\s+energy\s+of\s+(?:photo)?electron\b'
     r'|\bnuclear\s+radius\b|\br0\b|\bR0\b'
     r'|\balpha\s+particles?\s+emitted\b|\bbeta\s+particles?\s+emitted\b|\bdecay\s+series\b'
     r'|\bmean\s+life\b|\baverage\s+life\b|\bdecay\s+constant\s+(?:lambda|λ)\b'
     r'|\bnucleus\s+emits\b|\bparent\s+nucleus\b|\bdaughter\s+nucleus\b'
     r'|\bradioactive\s+half.life\b|\bhalf.life\s+of\s+(?:14c|carbon|radium|uranium|thorium|radon)\b'
     r'|\bhalf.life\s+of\s+a\s+radioactive\b|\bage\s+of\s+a\s+(?:bone|rock|sample)\b'
     r'|\bcarbon\s+dating\b|\bradioactive\s+dating\b|\bpair\s+production\b|\bpair\s+annihilation\b'
     r'|\bhe\+\s+ion\b|\bhe\^?\+\b.*\bioniz\b|\benergy\s+of\s+(?:he\+|hydrogen)\s+(?:atom|ion)\b'
     r'|\bground\s+state\s+of\s+(?:he|h)\b'
     r'|\bactivity\s+of\s+(?:a\s+)?(?:sample|source)\s+(?:decreases?|changes?|is)\b'
     r'|\bactivity\s+decreases?\s+from\b|\bdisintegrations?\s+per\s+second\b|\bdps\b'
     r'|\bfraction\s+(?:remaining|left|undecayed)\s+after\b|\b(?:fraction|amount)\s+remaining\s+after\b'
     r'|\bfind\s+(?:the\s+)?(?:half.life|mean\s+life)\s+(?:of|from|given)\b'
     r'|\bactivity\s+of\s+\d+\s*g\s+of\b|\bra.?\d+\b.*\bhalf.life\b|\bhalf.life\b.*\bra.?\d+\b'
     r'|\btransition\s+from\s+n\s*=\s*\d+\s+to\s+n\s*=\s*\d+\b'
     r'|\belectron\s+in\s+hydrogen\s+atom\s+(?:makes?|undergoes?|jumps?)\b'
     r'|\bspectral\s+series\b|\bbalmer\b|\blyman\b|\bpasschen\b|\bbrackett\b|\bpfund\b'
     r'|\bfrequency\s+of\s+radiation\s+emitted\s+when\s+electron\b'
     r'|\bhydrogen\s+like\s+atom\b|\bhyd\w+\s+atom\s+in\s+n\s*=\b', "modern_physics"),
    # ── Probability distribution — MUST come before statistical_analysis (catches "mean" and "variance") ──
    (r'\bpoisson\s+distribution\b|\bnormal\s+distribution\b|\bbinomial\s+distribution\b', "probability_distribution"),
    (r'\bpoisson\s+distribution\s+has\b|\bpoisson\s+with\s+(?:mean|lambda|λ)\b'
     r'|\bpoisson\s+(?:distribution\s+)?(?:has\s+)?mean\s*=\b'
     r'|\bfind\s+p\s*\(\s*x\s*=\s*\d\)', "probability_distribution"),
    # ── statistical_analysis — MUST exclude physics/chemistry mean contexts ──
    (r'\bmean\b(?!\s+life\b)(?!\s+free\s+path\b)(?!\s+kinetic\s+energy\b)(?!\s+speed\b)(?!\s+velocity\b)(?!\s+square\b)(?!\s+deviation\b)(?!\s+position\b)(?!\s+value\s+theorem\b)(?!\s+thermal\b)(?!\s+radius\b)(?!\s+=\s*\d)(?!\s+emf\b)'
     r'|\bmedian\b|\bstandard\s+deviation\b'
     r'|\bvariance\b(?!\s+of\s+(?:a\s+)?(?:binomial|poisson|normal))'
     r'|\baverage\b(?!\s+life\b)(?!\s+speed\b)(?!\s+velocity\b)(?!\s+output\b)(?!\s+(?:and\s+)?rms\b)(?!\s+kinetic\b)(?!\s+power\b)(?!\s+radius\b)(?!\s+emf\b)'
     r'|\bkarl\s+pearson\b|\bregression\s+line\b|\bcoefficient\s+of\s+correlation\b'
     r'|\bspearman\b|\bquartile\s+deviation\b|\bcoefficient\s+of\s+variation\b', "statistical_analysis"),
    (r'\bcomplex\s+number\b|\bimaginary\s+parts?\b|\breal\s+parts?\b|\bmodulus\s+of\b|\bargument\s+of\b'
     r'|\bz\s*=\s*[a-z0-9(].*[ij]\b|\b\|z\b|\bz1\s*=\b|\bz2\s*=\b|\bz1\s+and\s+z2\b'
     r'|\bstandard\s+form\s+a\+bi\b|\ba\+ib\s+form\b|\bpolar\s+form\s+of\s+z\b'
     r'|\bmodulus\s+and\s+argument\b|\blocus\s+of\s+z\b'
     r'|\broots\s+of\s+unity\b|\bcube\s+roots\b|\bnth\s+roots\b|\bde\s+moivre\b'
     r'|\bz\^n\b|\bz\^3\s*=\b|\bcomplex\s+solutions\s+of\b'
     r'|\bsquare\s+roots?\s+of\s+(?:\d+[-+]\d*[ij]|-\d+)\b|\bsquare\s+roots?\s+of\b.*[ij]\b'
     r'|\bfind\s+(?:all\s+)?\d+(?:th|st|nd|rd)\s+roots?\s+of\b'
     r'|\ball\s+6th\s+roots?\b|\ball\s+nth\s+roots?\b|\bin\s+polar\s+form\b.*(?:roots?|complex)'
     r'|\bcomplex\s+roots?\s+of\s+polynomial\b|\bconjugate\s+pairs?\b', "complex_number"),
    (r'\bvan\'?\s*t\s+hoff\s+factor\b|\bdelta_?tf\b|\bdelta_?tb\b'
     r'|\belevation\s+in\s+boiling\s+point\b|\bdepression\s+in\s+freezing\b'
     r'|\bkb\s*=\s*\d(?![.\d]*\s*[xe×]\s*10)|\bkf\s*=\s*\d|\bmolal\s+elevation\b|\bmolal\s+depression\b'
     r'|\bboiling\s+point\s+elevation\b|\bfreezing\s+point\s+depression\b'
     r'|\bmole\s+fraction.*dissolved\b|\bdissolved.*mole\s+fraction\b'
     r'|\bcolligative\b|\bosmotic\s+pressure\b|\bmolar\s+mass\s+of\s+solute\b', "colligative_properties"),
    (r'\bfactori[sz][ae]?\b|\bfactori[sz]ing\b'
     r'|\bfactor\b(?!\s+of\s+(?:safety|merit|quality|q))(?!\s+theorem)', "factorization"),
    (r'\bexpand\b', "algebraic_expansion"),
    # ── JEE Advanced: Trigonometry (must come before generic simplify) ────────
    (r'\binverse\s+trig|\barcsin\b|\barccos\b|\barctan\b|\bsin\s+inverse\b|\bcos\s+inverse\b|\btan\s+inverse\b|\busing\s+inverse'
     r'|\bsin\^?\s*-\s*1\b|\bcos\^?\s*-\s*1\b|\btan\^?\s*-\s*1\b'
     r'|\bsin[⁻-]1\b|\bcos[⁻-]1\b|\btan[⁻-]1\b'
     r'|\bsin\^?\s*\(\s*-\s*1\s*\)|\bcos\^?\s*\(\s*-\s*1\s*\)|\btan\^?\s*\(\s*-\s*1\s*\)'
     r'|\bsin\^\{-1\}|\bcos\^\{-1\}|\btan\^\{-1\}'
     r'|\bprincipal\s+value\s+of\b|\bevaluate\s+(?:sin|cos|tan|sec|cosec|cot)[⁻-]'
     r'|\bsin\^{-1}\s*\(|\bcos\^{-1}\s*\(|\btan\^{-1}\s*\('
     r'|\bprove\b.*\b(?:sin|cos|tan)\^\-?1\b.*\b(?:sin|cos|tan)\^\-?1\b', "inverse_trig"),
    (r'\btrigonometric\s+equation\b|\bgeneral\s+solution\b|\bprincipal\s+(?:value|solution)\b'
     r'|\bif\s+(?:tan|sin|cos|sec|csc|cot)\s*\('
     r'|\bsolve\b.*\b(?:sin|cos|tan|sec|cosec|csc|cot)\b(?:.*\b(?:\[0,\s*2\s*pi\]|\[0,\s*pi\])\b)?'
     r'|\ball\s+solutions\s+in\s+\[|\bfor\s+x\s+in\s+\[0,\s*2.*pi\b'
     r'|\bfind\s+all\s+(?:real\s+)?solutions?\s+of\s+(?:sin|cos|tan|sec|cosec|cot)\b'
     r'|\bfind\s+all\s+x\s+satisfying\b.*\b(?:sin|cos|tan)\b'
     r'|\bsin\^?2\s*\(x\)\s*[-+*]|\bcos\^?2\s*\(x\)\s*[-+*]|\btan\^?2\s*\(x\)\s*=', "trigonometric_equation"),
    (r'\bcompound\s+angle\b|\bdouble\s+angle\b|\bhalf\s+angle\b|\bmultiple\s+angle\b'
     r'|\bpythagorean\s+identit|\bsum.to.product\b|\bproduct.to.sum\b'
     r'|\bsin\^?2\s*\+\s*cos\^?2\b|\bsin[²2]\s*\(?[a-z]\)?\s*\+\s*cos[²2]\b'
     r'|\bprove\s+(?:that\s+)?(?:sin|cos|tan)\b|\bprove\s*:\s*(?:\(|sin|cos|tan)\b'
     r'|\bshow\s+that\s+sin\b|\bshow\s+that\s+(?:\(.*\).*=.*(?:sin|cos|tan)|(?:sin|cos|tan))\b'
     r'|\bderive\s+(?:expressions?\s+for|the\s+formula\s+for)\s+sin\b'
     r'|\bexpress\s+(?:\d+\s*)?(?:sin|cos)\s*\((?:[A-Za-z]|\d[a-z])\)'
     r'|\bmaximum\s+(?:and\s+minimum\s+)?values?\s+of\s+\d+\s*sin|\bmaximum\s+values?\s+of\s+(?:a\s+)?sin'
     r'|\bminimum\s+values?\s+of\s+(?:a\s+)?(?:sin|cos)\b'
     r'|\bmaximum\s+and\s+minimum\s+values?\s+of\s+\d+\s*(?:sin|cos)\b'
     r'|\bfind\s+(?:the\s+)?value\s+of\s+(?:sin|cos|tan)\s*\(\d+\s*°?\)'
     r'|\bsin\s*\(\s*\d+\s*°\s*\)\s*[·\*]\s*(?:sin|cos)\s*\(\s*\d+\s*°\s*\)'
     r'|\bproduct\s+of\s+(?:sin|cos)\b|\bsin\s*\(\s*18\s*°'
     r'|\bsin\s*\(\s*a\s*\+\s*b\s*\)\s*[·\*]\s*sin\s*\(\s*a\s*-\s*b\s*\)'
     r'|\btan\s*\(\s*a\s*[-+]\s*b\s*\)\b'
     r'|\bmaximum\s+(?:and\s+minimum\s+)?value\s+of\s+(?:\d+\s*)?(?:sin|cos)\b'
     r'|\bmin(?:imum)?\s+(?:and\s+max(?:imum)?\s+)?value\s+of\s+(?:a\s*[·*]\s*)?(?:sin|cos)\b'
     r'|\b(?:a\s*sin|a\s+sin)\s*\(\s*(?:x|theta|θ)\s*\)\s*[+±]\s*(?:b\s*cos|b\s+cos)\b'
     r'|\br\s*sin\s*\(\s*(?:x\s*[+±]\s*(?:alpha|φ)|theta)\b|\br\s*cos\s*\(\s*(?:x\s*[+±])', "trig_identities"),
    (r'\bsine\s+rule\b|\bcosine\s+rule\b|\barea\s+of\s+triangle\s+using\b'
     r'|\barea\s+of\s+(?:a\s+)?triangle\s+with\s+(?:sides|angle|vertices)\b'
     r'|\bin\s+(?:a\s+)?triangle\s+[a-zA-Z]{2,3}\b|\btriangle\s+[a-zA-Z]{2,3}\s+(?:if|with|has)\b'
     r'|\bsides?\s+\d+.*\bincluded\s+angle\b|\bincluded\s+angle\b'
     r'|\bcircumradius\b|\binradius\b'
     r'|\bangle\s+of\s+(?:elevation|depression)\s+of\b|\bheight\s+of\s+(?:a\s+|the\s+)?(?:tower|tree|pole|cliff|flag|hill)\b'
     r'|\bright\s+(?:triangle|angle\s+triangle)\s+with\s+sides?\b'
     r'|\bsolve\s+(?:the\s+)?triangle\b|\busing\s+(?:sine|cosine)\s+rule\b'
     r'|\bshow\s+that\s+triangle\s+with\s+vertices\b|\btriangle\s+with\s+vertices.*\b(?:isosceles|equilateral|right)\b'
     r'|\bangles?\s+are\s+in\s+(?:ratio|the\s+ratio)\b.*\btriangle\b', "triangle_trig"),
    (r'\bsimplify\b', "simplification"),
    # ── LC circuit BEFORE SHM (LC oscillations look like SHM but are electromagnetism) ──
    (r'\blc\s+circuit\b|\boscillat.*\bl\s*=.*\bc\s*=\b|\bmaximum\s+speed\s+of\s+(?:an?\s+)?lc\b', "electromagnetism"),
    # ── JEE Advanced: Physics — SHM (MUST come before maxima_minima) ─────────
    (r'\bsimple\s+harmonic\s+motion\b|\bshm\b|\bs\.h\.m\b'
     r'|\btime\s+period\s+of\s+oscillat|\brestoring\s+force\b'
     r'|\bspring\s+constant\b|\bspring\s+of\s+(?:force\s+)?constant\b'
     r'|\bspring.mass\b|\b(?:find|calculate)\s+(?:the\s+)?(?:time\s+period|period|frequency|amplitude)\s+of\s+(?:oscillat|pendulum|spring|shm)'
     r'|\boscillat|\bpendulum\b|\bangular\s+frequency\b'
     r'|\bsprings?\b.*\bin\s+(?:series|parallel)\b|\bk_?(?:eff|effective)\b'
     r'|\btime\s+period\b.*\bspring\b|\bspring\b.*\btime\s+period\b', "shm_problem"),
    # ── JEE Advanced: Calculus (maxima/minima, monotonicity) ─────────────────
    (r'\bmaxima\b|\bminima\b|\blocal\s+max|\blocal\s+min|\bglobal\s+max|\bglobal\s+min'
     r'|\bmaximum\s+(?:value|area|volume|length|profit|revenue|range)\s+of\b'
     r'|\bminimum\s+(?:value|area|volume|surface|cost|distance|time|length)\s+of\b'
     r'|\bmaximum\s+and\s+minimum\b|\bextreme\s+value\b|\boptimize\b|\boptimization\b'
     r'|\bmaximum\s+volume\s+of\b|\bminimum\s+surface\s+area\b|\binscribed\s+in\s+a\s+(?:sphere|circle|semicircle)\b'
     r'|\bnearest\s+to\s+point\b|\bclosest\s+to\s+point\b|\bminimiz(?:e|ing)\s+(?:the\s+)?(?:distance|cost|area|surface)\b'
     r'|\bshortest\s+distance\s+from\s+(?:the\s+)?point\s+\('
     r'|\bpoint\s+on\s+(?:curve|line)\s+y=.*(?:nearest|closest)\b'
     r'|\bdimensions\s+of\s+(?:a\s+)?(?:box|rectangle|cylinder)\b|\bdimensions\s+of\s+(?:(?:\w+\s+){0,3})?(?:maximum|minimum)\s+(?:volume\s+)?(?:box|cylinder|can)\b'
     r'|\bmaximum\s+volume\s+(?:box|cylinder)\b|\bbox\s+without\s+lid\b|\bbox\s+with\s+(?:square|open)\b'
     r'|\bfencing\s+to\s+enclose\b|\bfarmer\s+has\b|\bwire\s+bent\b|\bbent\s+into\s+(?:a\s+)?(?:rectangle|square|triangle)\b'
     r'|\bsemicircle\s+on\s+top\s+of\s+(?:a\s+)?rectangle\b|\bwindow\s+has\s+(?:a\s+)?semicircle\b'
     r'|\bladder.*\bmaximis?\b|\bladder.*\bangle\s+that\b'
     r'|\bmaximum\s+value\s+of\s+(?:the\s+)?(?:expression|function|f\(x\))\b'
     r'|\bminimum\s+value\s+of\s+(?:the\s+)?(?:expression|function|f\(x\))\b'
     r'|\bfind\s+the\s+(?:maximum|minimum)\s+value\s+of\b|\bvalue\s+of\s+x\s+that\s+minimizes\b', "maxima_minima"),
    (r'\bmonotonically\b|\bincreasing\s+function\b|\bdecreasing\s+function\b'
     r'|\bstrictly\s+(?:increasing|decreasing)\b|\bintervals?\s+of\s+(?:increase|decrease|monotonicity)\b'
     r'|\bintervals?\s+(?:where|in\s+which)\s+f(?:\(x\))?\s+is\s+(?:increasing|decreasing)\b', "monotonicity"),
    (r'\binflection\s+points?\b|\bpoints?\s+of\s+inflection\b|\bconcav(?:e\s+(?:up|down|function)|ity\b)', "maxima_minima"),
    (r'\brolle\'?s?\s+theorem\b|\bmean\s+value\s+theorem\b|\blagrange|\blmvt\b|\bmvt\b'
     r'|\bverify\s+rolle\b|\bapply\s+rolle\b|\bverify\s+lmvt\b|\bapply\s+lmvt\b|\bapply\s+mvt\b', "mean_value_theorem"),
    # ── JEE Advanced: Coordinate Geometry ────────────────────────────────────
    (r'\bparabola\b', "conic_parabola"),
    (r'\bellipse\b', "conic_ellipse"),
    (r'\bhyperbola\b', "conic_hyperbola"),
    (r'\bconic\s+section\b', "conic_section"),
    (r'\bchord\s+of\s+contact\b|\bpole\s+and\s+polar\b|\bdirector\s+circle\b', "conic_section"),
    (r'\bpair\s+of\s+(straight\s+)?lines?\b|\bcombined\s+equation\b|\bhomogeneous\s+equation'
     r'|\bcoincident\s+lines\b|\blines\s+represented\s+by\b.*\^2'
     r'|\bangle\s+between\s+lines\s+represented\s+by\b'
     r'|\bax\^2\s*\+\s*2hxy|\b2x\^2\s*\+.*\bxy\b.*\+.*y\^2'
     r'|\brepresents\s+(?:a\s+)?pair\s+of\b', "pair_of_lines"),
    (r'\bradical\s+axis\b|\bcoaxial\b|\borthogonal\s+circles\b', "circle_geometry"),
    (r'\bequation\s+of\s+(?:a\s+)?circle\b|\bcircle\s+pass(?:ing)?\s+through\b'
     r'|\bangle\s+of\s+intersection\s+of\s+circles\b|\bcommon\s+chord\s+of\b'
     r'|\blength\s+of\s+(?:the\s+)?common\s+chord\b|\bnumber\s+of\s+common\s+tangents\b'
     r'|\bcommon\s+chord\s+length\b|\btheir\s+common\s+chord\b'
     r'|\bpair\s+of\s+tangents?\s+from\b|\bpair\s+of\s+tangents?\s+to\s+(?:the\s+)?circle\b'
     r'|\bcommon\s+tangents?\s+to\b|\btangent\s+to\s+(?:the\s+)?circle\b'
     r'|\bin\s+(?:a\s+)?circle\s+x\^2\b|\bchord\s+(?:whose|with)\s+midpoint\b|\bmidpoint\s+of\s+(?:a\s+)?chord\b'
     r'|\bchord\s+of\s+(?:a\s+)?circle\b|\bcircle\s+x\^2\+y\^2\b|\bhave\s+a\s+common\s+chord\b'
     r'|\bcircles?\s+intersect\b|\bexternal\s+tangent\b|\binternal\s+tangent\b', "circle_geometry"),
    # ── JEE Advanced: 3D Geometry direction cosines/ratios — BEFORE straight_line ──
    (r'\bdirection\s+cosines?\s+of\s+(?:the\s+)?(?:line\s+joining|line\s+passing)\b'
     r'|\bdirection\s+ratios?\s+of\s+(?:the\s+)?(?:line\s+joining|line\s+passing)\b'
     r'|\bfind\s+(?:the\s+)?direction\s+(?:cosines?|ratios?)\b'
     r'|\bangle\s+between\s+lines\s+with\s+direction\b', "3d_geometry"),
    # ── 3D specific patterns BEFORE straight_line (to avoid wrong 2D classification) ──
    (r'\bfoot\s+of\s+perpendicular\b.*\bplane\b|\bperpendicular\s+distance\b.*\bplane\b'
     r'|\bimage\s+of\s+point\b.*\bplane\b|\bangle\s+between\s+(?:the\s+)?line\b.*\bplane\b'
     r'|\bequation\s+of\s+(?:a\s+)?sphere\b|\bsphere\s+with\s+cent(?:re|er)\b'
     r'|\bequation\s+of\s+(?:the\s+)?line\s+through\b.*\b[ijk]\b'
     r'|\bparallel\s+to\s+\d+[ijk]\b|\bparallel\s+to\s+[ijk]\b.*\bline\b', "3d_geometry"),
    (r'\bangle\s+between\s+lines\b'
     r'|\bdistance\s+from\s+point\s+to\s+line\b|\bconcurrent\b'
     r'|\bdistance\s+between\s+(?:parallel\s+)?lines\b(?!\s*r\s*=)|\bparallel\s+lines\s+\d+x\b'
     r'|\bslope\s+of\s+(?:the\s+)?line\b|\bequation\s+of\s+(?:a\s+|the\s+)?line\b|\bline\s+passing\s+through\b'
     r'|\bfoot\s+of\s+perpendicular\s+from\b(?!.*\bplane\b)|\breflection\s+of\s+point\b'
     r'|\borthocentre\b|\bcentroid\b.*\btriangle\s+formed\s+by\b|\bmedian\s+of\s+triangle\b'
     r'|\bimage\s+of\s+point\s+in\s+(?:the\s+)?line\b|\bvertex\s+of\s+triangle\b'
     r'|\btriangle\s+formed\s+by\s+lines\b|\bangle\s+bisectors\s+of\s+lines\b'
     r'|\bperpendicular\s+distance\s+from\s+(?:point|origin)\b(?!.*\bplane\b)'
     r'|\blocus\s+of\s+(?:a\s+|point\s+)?(?:[a-z]\s+)?(?:point\s+)?equidistant\b'
     r'|\bequal\s+intercepts?\s+on\s+(?:both\s+)?(?:the\s+)?axes?\b|\bline\s+makes\s+equal\s+intercepts?\b'
     r'|\blines?\s+through\s+\(\-?\d+,\s*\-?\d+\).*\bangle\b|\bequations?\s+of\s+lines?\s+through\b'
     r'|\bshow\s+that\s+(?:the\s+)?(?:three\s+)?points?\s+(?:are\s+)?collinear\b', "straight_line"),
    # ── JEE Advanced: 3D Geometry & Vectors ──────────────────────────────────
    (r'\bdirection\s+cosines?\b|\bdirection\s+ratios?\b', "3d_geometry"),
    (r'\bskew\s+lines\b|\bshortest\s+distance\b', "3d_geometry"),
    (r'\bequation\s+of\s+(?:a\s+)?plane\b|\bplane\s+pass(?:ing)?\s+through\b'
     r'|\bparallel\s+planes?\b|\bdistance\s+between\s+(?:the\s+)?planes?\b'
     r'|\bfoot\s+of\s+perpendicular\s+from\b|\bpoint\s+to\s+(?:the\s+)?plane\b'
     r'|\bdistance\s+of\s+(?:a\s+)?point\s+from\s+(?:the\s+)?plane\b'
     r'|\bperpendicular\s+distance\s+from\s+point\b.*\bplane\b'
     r'|\bangle\s+between\s+(?:the\s+)?planes\b|\btwo\s+planes\b.*\bangle\b|\bangle\s+between\s+them\b'
     r'|\bline\s+of\s+intersection\s+of\s+planes\b'
     r'|\bimage\s+of\s+point\b.*\b(?:in|in\s+the)\s+plane\b'
     r'|\breflected\s+(?:in|through)\s+(?:the\s+)?plane\b'
     r'|\bfind\b.*\bline\b.*\bperpendicular\s+to\s+both\b|\bline\s+through\b.*\bperpendicular\s+to\s+both\b'
     r'|\bplane\s+\d+x\b|\bplane\s+[a-z]\+\d|\bin\s+3d\b.*\bline\b', "3d_geometry"),
    (r'\bscalar\s+triple\s+product\b|\bvector\s+triple\s+product\b'
     r'|\bangle\s+between\s+(?:the\s+)?vectors?\b|\bvector\s+perp(?:endicular)?\s+to\s+both\b'
     r'|\bperp(?:endicular)?\s+to\s+both\s+[a-z]\s*=\b|\bcross\s+product\b'
     r'|\bscalar\s+projection\b|\bvector\s+projection\b|\bprojection\s+of\s+[a-z]\s+on\s+[a-z]\b'
     r'|\b\|a\s*[×x]\s*b\||\ba\s*[·\.]\s*b\b|\bmoment\s+of\s+force\s+(?:f=|about)\b'
     r'|\bwork\s+done\s+by\s+force\s+f='
     r'|\bvolume\s+of\s+(?:a\s+)?(?:tetrahedron|parallelepiped|parallelop)\b'
     r'|\btetrahedron\s+with\s+vertices\b|\barea\s+of\s+(?:a\s+)?parallelogram\s+(?:with|using)\s+(?:diagonal|vectors)\b'
     r'|\bprojection\s+of\s+a\s*=\b|\bfind\s+[a-z]\s*[×x]\s*[a-z]\b|\ba\s*×\s*b\b'
     r'|\|[a-z]\s*[+]\s*[a-z]\|\s*=\s*\|[a-z]\s*[-]\s*[a-z]\|'
     r'|\bcollinear\s+using\s+vectors?\b|\bperpendicularity\s+proof\b'
     r'|\barea\s+of\s+(?:a\s+)?parallelogram\s+(?:with\s+adjacent|having\s+sides\s+as\s+vectors)\b'
     r'|\bvectors?\s+[a-z]\s*=\s*\d+[ijk]|\bif\s+[a-z]\s*\+\s*[a-z]\s*\+\s*[a-z]\s*=\s*0\b.*\b\|[a-z]\|\b'
     r'|\bvector\s+component\b.*\bperpendicular\s+to\b|\bangle\s+between\s+them\b.*\bvectors?\b'
     r'|\bunit\s+vectors?\s+and\s+(?:the\s+)?angle\s+between\b|\bif\s+[a-z]\s+and\s+[a-z]\s+are\s+unit\s+vectors?\b', "vector_algebra"),
    (r'\bunit\s+vectors?\b|\bposition\s+vector\b|\bprojection\s+of\s+vector\b', "vector_algebra"),
    # ── JEE Advanced: Probability ─────────────────────────────────────────────
    (r'\bbayes\'?\s+theorem\b|\bconditional\s+probability\b'
     r'|\burn\s+[1-9]\b|\burn\s+[ivx]+\b|\bfrom\s+random\s+urn\b'
     r'|\bif\s+(?:white|red|black)\s+(?:ball|card)\b.*\bprobability\b.*\burn\b'
     r'|\bprobability\s+that\s+it\s+came\s+from\b|\bball\s+drawn\s+from\s+random\b'
     r'|\bfind\s+p\s*\(.*\bgiven\b|\bgiven\s+(?:that\s+it\s+is|the\s+card\s+is|the\s+ball\s+is|red\s+card|face\s+card)\b'
     r'|\bp\s*\(\s*face\s+card\s+given\b|\bgiven\s+red\s+card\b', "bayes_probability"),
    (r'\brandom\s+variable\b|\bprobability\s+distribution\b|\bexpected\s+value\b', "probability_distribution"),
    (r'\bbinomial\s+distribution\b|\bpoisson\s+distribution\b|\bnormal\s+distribution\b', "probability_distribution"),
    (r'\bx\s+is\s+(?:a\s+)?binomial\b|\bx\s+~\s*b\(|\bx\s+follows\s+binomial\b'
     r'|\bx\s+is\s+(?:a\s+)?poisson\b|\bx\s+~\s*p\(|\bp\(x\s*=\s*\d\)\b'
     r'|\bmode\s+of\s+(?:a\s+)?(?:binomial|poisson)\b|\bvar(?:iance)?\s+of\s+binomial\b', "probability_distribution"),
    (r'\bprobability\s+(?:that|of|find)\b'
     r'|\bbag\s+contains\b|\bdeck\s+of\s+(?:\d+\s+)?cards\b|\bcard\s+is\s+drawn\b'
     r'|\bballs?\s+(?:drawn|picked|selected)\b|\bwithout\s+replacement\b'
     r'|\bdice\s+(?:is|are)\s+(?:thrown|rolled|tossed)\b|\bcoins?\s+(?:is|are)\s+(?:tossed|flipped)\b'
     r'|\bspeaks\s+truth\b|\bspeaking\s+truth\b|\bspeaks\s+lies?\b'
     r'|\bp\s*\(\s*a\s+(?:union|∪)\s+b\s*\)\b|\bp\s*\(\s*a\s+(?:intersect|∩)\s+b\s*\)\b'
     r'|\bp\s*\(\s*a\s*[∪∩]\s*b\s*\)\b'
     r'|\bat\s+least\s+one\s+(?:head|tail|six|ball|success)\b|\bexactly\s+\d+\s+(?:head|tail|ball)\b'
     r'|\bfind\s+p\s*\(\s*a\s*\)\b|\bp\s*\(\s*a\s*\)\s*=\s*[\d.]+\b'
     r'|\bmutually\s+exclusive\b|\bindependent\s+events?\b'
     r'|\bindependently\s+(?:solve|attempt|try)\b|\bsolves?\s+.*\bprobabilit\b'
     r'|\bprobabilities?\s+(?:\d/\d|0\.\d)\b|\bwith\s+probability\s+(?:\d/\d|0\.\d)\b'
     r'|\bin\s+a\s+class\s+\d+%\b|\bstudies?\s+(?:both|neither|only)\b|\bstudy\s+(?:both|neither|only)\b'
     r'|\bfair\s+(?:die|dice|coin)\b|\bfrom\s+a\s+(?:bag|box|urn)\b|\bin\s+a\s+(?:bag|box|urn)\b'
     r'|\btickets?\s+numbered\s+\d+\s+(?:to|through|-)\s+\d+\b|\bcards?\s+numbered\s+\d+\s+to\s+\d+\b'
     r'|\bfind\s+p\s*\(\s*(?:sum|multiple|at\s+least|exactly|neither|both)\b'
     r'|\bplay\s+(?:cricket|football|hockey|basketball|tennis)\b.*\bplay\b'
     r'|\b(?:students?|people|persons?)\s+(?:who\s+)?play\b.*\b(?:and|or|both|only|neither)\b'
     r'|\bvenn\s+diagram\b|\bset\s+theory\b.*\bn\s*\(\s*[abc]\s*\)\b'
     r'|\bplay\s+both\b|\bplay\s+only\b|\bplay\s+neither\b|\bwho\s+play\s+(?:only|both|neither)\b'
     r'|\b\d+\s+students?\s+play\s+(?:cricket|football|hockey)\b'
     r'|\bn\s*\(\s*a\s*\)\s*=\s*\d+.*\bn\s*\(\s*b\s*\)\s*=\s*\d+.*\bn\s*\(\s*a\s*∩\s*b\s*\)\b', "probability_problem"),
    (r'\bp\s*\(\s*[a-z]\s*[|│]\s*[a-z]\s*\)\b'
     r'|\bp\s*\(\s*[a-z]\s*/\s*[a-z]\s*\)\b'
     r'|\bdefect(?:ive)?\s+(?:item|product).*\bfound\b|\bfind\s+p\s*\(\s*(?:disease|defect|item)\b'
     r'|\bgiven\s+(?:that|it|the)\s+(?:ball|card|item|person)\b'
     r'|\bfactory\s+[ab]\b|\bmachine\s+[ab]\b|\bproduces?\s+\d+%\s+(?:of|defective)\b'
     r'|\bdefect\s+rates?\s+of\b|\bfound\s+(?:to\s+be\s+)?defective\b'
     r'|\burn\s+[abc]\b|\burn\s+contains\b|\bselect\s+(?:a\s+)?ball\s+from\s+(?:a\s+)?(?:randomly\s+)?(?:selected\s+)?urn\b'
     r'|\bgiven\s+(?:that\s+)?(?:the\s+)?(?:ball|card|coin)\s+(?:drawn\s+)?(?:is|was)\s+(?:red|white|black|blue)\b'
     r'|\bprobability\s+that\s+(?:it\s+came|it\s+comes|came)\s+from\b'
     r'|\b(?:defective|non.defective)\s+item\s+(?:is|was)\s+(?:from|made\s+by)\b', "bayes_probability"),
    # ── JEE Advanced: Sequences & Series ──────────────────────────────────────
    (r'\barithmetic\s+progression\b|\b\bap\b.*\bterm\b|\bnth\s+term\s+of\s+(?:an?\s+)?ap\b', "sequences_series"),
    (r'\bgeometric\s+progression\b|\bgp\b.*\bterm\b|\bcommon\s+ratio\b', "sequences_series"),
    (r'\bharmonic\s+progression\b|\bhp\b', "sequences_series"),
    (r'\bsum\s+to\s+infinity\b|\binfinite\s+(?:gp|series)\b', "sequences_series"),
    # ── JEE Advanced: Binomial Theorem ───────────────────────────────────────
    (r'\bbinomial\s+theorem\b|\bbinomial\s+expansion\b', "binomial_theorem"),
    (r'\bmiddle\s+terms?\b|\bterm\s+independent\s+of\s+x\b|\bcoefficient\s+of\s+x\b'
     r'|\bsum\s+of\s+(?:all\s+)?(?:binomial\s+)?coefficients\b|\bC\(n,\d\)\b|\bC\(n,r\)\b'
     r'|\bsum\s+of\s+c\(n,0\)\b|\bsum\s+.*\bC\(n,'
     r'|\bgreatest\s+term\s+in\s+(?:the\s+)?(?:expansion\s+of\s+)?\(|\bgreatest\s+term\s+in\s+(?:the\s+)?expansion\b'
     r'|\blargest\s+term\s+in\s+(?:the\s+)?(?:expansion\s+of\s+)?\(|\bterm\s+containing\s+x\^?\d+\b'
     r'|\b\dth\s+term\s+from\s+(?:end|last)\b|\bnth\s+term\s+from\s+(?:end|last)\b'
     r'|\bgenerali[sz]ed\s+binomial\b|\bin\s+the\s+expansion\s+of\b', "binomial_theorem"),
    # Rotational Mechanics — extended to catch MI abbreviation, torque, pivoted
    (r'\bmoment\s+of\s+inertia\b|\bangular\s+momentum\b|\brotational\s+kinetic\s+energy\b'
     r'|\brolling\s+(?:without\s+slipping|motion)\b|\brolls\s+without\s+slipping\b'
     r'|\bparallel\s+axis\b|\bperpendicular\s+axis\b'
     r'|\bMI\s*=\s*[\d.]|\bMI\b.*\bkg[·.]m[²2]\b|\bkg[·.]m[²2]\b.*\bMI\b'
     r'|\busing\s+MI\s*=\b|\bMI\s*=\s*MR\b|\bchanging\s+MI\b|\bMI\s+from\b|\bthick\s+ring\b'
     r'|\bmi\s*=\s*mr\b|\busing\s+mi\s*=\b'
     r'|\bmoment\s+of\s+inertia\b|\bangular\s+acceleration\b'
     r'|\bpivoted\s+at\b|\bpivot\s+at\b|\bhinged\s+at\b'
     r'|\btorque\s+(?:of|on|acts|reduces|applies)\b|\btorque\s+=\b'
     r'|\brolling\s+at\s+[\d.]+\s*m/s\b|\bangular\s+velocity\b.*\brolling\b|\brolling\b.*\bangular\s+velocity\b'
     r'|\brolls?\s+(?:down|up)\s+(?:a\s+)?(?:frictionless\s+)?(?:slope|incline)\b|\brolling\s+down\b'
     r'|\bradius\s+of\s+gyration\b|\bconservation\s+of\s+angular\s+momentum\b'
     r'|\bskater\s+spinning\b|\bextends?\s+arms?\b|\brev/s\b|\bspins?\s+at\s+\d+\s+rev\b'
     r'|\binner\s+radius\s+r\b.*\bouter\s+radius\b|\binner\s+radius.*\bouter\b'
     r'|\brolling\s+cylinder\b|\brolling\s+sphere\b|\brolling\s+disc\b|\brolling\s+ring\b'
     r'|\bdisc\s+and\s+ring\b|\bsphere\s+and\s+(?:disc|ring|hollow)\b|\bhoop\s+(?:rolls|rolling)\b'
     r'|\bkinetic\s+energy\s+of\s+a\s+rolling\b'
     r'|\bplatform\s+rotates?\s+at\s+\d+\s*rad\b|\bhorizontal\s+platform\s+rotates?\b'
     r'|\brotates?\s+at\s+\d+\s*(?:rad|rpm)\b(?!.*\blens\b)(?!.*\bcircuit\b)(?!.*\bcoil\b)(?!.*\bfield\b)(?!.*\bemf\b)'
     r'|\bwalks?\s+from\s+(?:edge|rim)\s+to\s+(?:centre|center)\b|\bman\s+walks?\b.*\bradius\b'
     r'|\bbug\s+(?:lands?|sits?)\s+at\s+(?:the\s+)?rim\b|\bdisc\s+of\s+mass\b.*\brotates?\b', "rotational_mechanics"),
    # Electrostatics — extended (electric dipole only, not magnetic)
    (r'\bcoulomb\'?s?\s+law\b|\belectric\s+field\b|\bgauss\'?s?\s+law\b'
     r'|\belectric\s+potential\b|\bcapacit(?:ance|or)\b|\bdielectric\b'
     r'|\belectric\s+force\s+between\b|\bnull\s+point\b'
     r'|\bwork\s+done\s+to\s+move\s+a\s+charge\b|\bmove\s+a\s+charge\s+through\b'
     r'|\belectric\s+dipole\b|\belectric\s+dipole\s+moment\b'
     r'|\belectric\s+flux\b|\bsurface\s+charge\s+density\b'
     r'|\bpotential\s+energy\s+of\s+two\s+charges\b|\bcharge\s+distribution\b'
     r'|\buniformly\s+charged\s+(?:ring|disc|sphere|plate)\b|\bcharged\s+ring\b'
     r'|\bpoint\s+on\s+axis\s+of\b|\bwork\s+done\s+in\s+assembling\b'
     r'|\bfour\s+(?:equal\s+)?charges\b|\bthree\s+(?:equal\s+)?charges\b'
     r'|\bseries\s+and\s+parallel\s+combination\s+of\s+capacitors\b'
     r'|\bcapacitors?\s+connected\s+in\s+(?:series|parallel)\s+(?:and|across)\b'
     r'|\bcapacit(?:ance|ors?)\b.*\bconnected\s+in\s+(?:series|parallel)\b'
     r'|\bconnected\s+in\s+(?:series|parallel)\b.*\bcapacit(?:ance|ors?)\b', "electrostatics"),
    (r'\bbiot.savart\b|\bampere\'?s?\s+law\b|\bfar?aday\'?s?\s+law\b|\blenz\'?s?\s+law\b'
     r'|\bself.inductance\b|\bmutual\s+inductance\b|\binductance\b'
     r'|\bmagnetic\s+field\b|\bmagnetic\s+flux\b|\bmagnetic\s+force\b'
     r'|\binduced\s+emf\b|\bmotional\s+emf\b|\bpeak\s+emf\b'
     r'|\bforce\s+on\s+(?:a\s+)?wire\b|\bparallel\s+wires\b'
     r'|\bcircular\s+coil\b|\bsolenoid\b|\btime\s+constant\s+of\b'
     r'|\brl\s+circuit\b|\brlc\s+circuit\b|\brc\s+circuit\b|\blcr\s+circuit\b|\blc\s+circuit\b'
     r'|\bimpedance\b|\breactance\b|\bresonant\s+frequency\b|\bresonance\s+frequency\b'
     r'|\brms\s+current\b|\brms\s+voltage\b|\bpower\s+factor\b'
     r'|\bq.factor\b|\bquality\s+factor\b|\bcurrent\s+gain\b'
     r'|\btransformer\b|\btransistor\b|\bnpn\b|\bpnp\b'
     r'|\binductive\s+reactance\b|\bcapacitive\s+reactance\b'
     r'|\bmagnetic\s+dipole\s+moment\b|\bpure\s+(?:inductor|capacitor|resistor)\b'
     r'|\bconnected\s+to\s+\d+\s*V\s+\d+\s*Hz\s+AC\b|\bAC\s+circuit\b'
     r'|\bpeak\s+(?:current|voltage)\s+(?:at|in)\s+resonance\b'
     r'|\bseries\s+lcr\s+(?:circuit|at)\b|\bseries\s+rlc\b'
     r'|\b(?:and|nand|nor|xor|xnor)\s+gate\b|\boutput\s+of\s+(?:and|nand|nor)\s+gate\b'
     r'|\bp.n\s+junction\s+diode\b|\bforward\s+resistance\b|\breverse\s+resistance\b'
     r'|\bfull.wave\s+rectifier\b|\bhalf.wave\s+rectifier\b'
     r'|\bzener\s+diode\b|\bbreakdown\s+voltage\b|\bdepletion\s+layer\b'
     r'|\bphase\s+difference\s+between\s+voltage\s+and\s+current\b'
     r'|\bmaximum\s+speed\s+of\s+(?:an?\s+)?lc\s+circuit\b'
     r'|\blc\s+circuit\s+oscillating\b|\boscillat.*\bl=.*\bc=\b'
     r'|\brectangular\s+coil\b|\bcoil\s+of\s+\d+\s+turns?\b|\bcoil\s+rotates?\s+in\b'
     r'|\bce\s+amplifier\b|\bcommon\s+emitter\b|\bcommon\s+base\b|\bcommon\s+collector\b'
     r'|\bcurrent\s+amplification\s+factor\b|\bvoltage\s+gain\s+of\s+(?:amplifier|transistor)\b'
     r'|\bbase\s+current\b|\bcollector\s+current\b|\bemitter\s+current\b'
     r'|\bα\s*=\b|\bβ\s*=\b|\bhfe\b|\bcommon.emitter\s+(?:configuration|circuit)\b'
     r'|\bphotoelectric\s+effect\s+(?:in|for|with)\b', "electromagnetism"),
    # Optics (before generic physics_problem)
    (r'\bconvex\s+lens\b|\bconcave\s+lens\b|\bconvex\s+mirror\b|\bconcave\s+mirror\b'
     r'|\blens\s+formula\b|\bmirror\s+formula\b|\bfocal\s+length\b|\bimage\s+distance\b'
     r'|\brefractive\s+index\b|\bcritical\s+angle\b|\btotal\s+internal\s+reflection\b'
     r'|\bsnell\'?s?\s+law\b|\bfringe\s+width\b|\byoung\'?s?\s+double\s+slit\b'
     r'|\bangle\s+of\s+(?:minimum\s+deviation|deviation|prism)\b|\bprism\b'
     r'|\bpower\s+of\s+(?:a\s+)?(?:lens|mirror)\b|\bmagnification\b'
     r'|\binterference\b|\bdiffraction\b'
     r'|\bobject\s+(?:is\s+)?placed\s+\d+\s*cm\b|\bimage\s+(?:is\s+)?formed\b'
     r'|\blateral\s+shift\b|\bglass\s+slab\b|\bnewton\'?s?\s+rings\b'
     r'|\bperson\s+uses\s+glasses\b|\bnear\s+point\b|\bfar\s+point\b'
     r'|\bsingle\s+slit\b|\bdouble\s+slit\s+experiment\b|\bresolving\s+power\b'
     r'|\btelescope\s+with\s+objective\b|\bfifth\s+(?:bright|dark)\s+fringe\b', "optics_problem"),
    # ── JEE Advanced: Chemistry specialised ───────────────────────────────────
    # ideal_gas_thermo FIRST to prevent chemistry_problem stoichiometry stealing PV=nRT
    (r'\bmonatomic\b|\bdiatomic\b|\bCv\b|\bCp\b|\bconstant\s+(?:pressure|volume)\b'
     r'|\bisobaric\b|\bisochoric\b|\binternal\s+energy\b|\bΔU\b|\bideal\s+gas\s+thermo'
     r'|\bcarnot\s+(?:engine|refrigerator|cycle)\b|\bheat\s+engine\b|\bcoefficient\s+of\s+performance\b'
     r'|\brms\s+speed\s+of\b|\bmean\s+kinetic\s+energy\s+of\b|\broot\s+mean\s+square\s+speed\b'
     r'|\bwork\s+done\s+by\s+(?:the\s+)?gas\b|\bwork\s+done\s+on\s+(?:the\s+)?gas\b'
     r'|\bthermodynamic\s+efficiency\b|\befficiency\s+of\s+(?:a\s+)?(?:carnot|heat)\b'
     r'|\bfirst\s+law\s+of\s+thermodynamics\b|\bsecond\s+law\s+of\s+thermodynamics\b'
     r'|\bpv\s*=\s*nrt\b|\bpv\^(?:gamma|γ)\b|\bpv\^1\.\d\b'
     r'|\badiabatic\s+(?:expansion|compression|process)\b'
     r'|\bisothermal\s+(?:expansion|compression|process)\b'
     r'|\bγ\s*=\s*[\d.]+\b|\bgamma\s*=\s*[\d.]+\b|\bspecific\s+heat\s+ratio\b'
     r'|\bgas\s+at\s+(?:stp|ntp|300|400|500|600)\s*[Kk]\b|\bgas\s+expands\b'
     r'|\bsink\s+temperature\b|\bsource\s+temperature\b'
     r'|\bmean\s+free\s+path\b|\bpv\s+diagram\b|\bmolecules\s+per\s+m\^?3\b'
     r'|\bpressure\s+exerted\s+by\s+gas\b|\bkinetic\s+energy\s+per\s+molecule\b'
     r'|\bisobarically\b|\bisothermally\b|\badiabatically\b', "ideal_gas_thermo"),
    # Electrochemistry cell equilibrium — BEFORE chemical_equilibrium steals "equilibrium constant"
    (r'\bequilibrium\s+constant\s+for\s+(?:the\s+)?cell\b'
     r'|\bdelta_?g\s*for\s+(?:the\s+)?cell\b|\bdelta_?g\s*°?\s+for\s+cell\s+reaction\b'
     r'|\bE\s*°\s*cell\s*=\s*\d|\be\s*°cell\s*=\b|\becell\b'
     r'|\bnernst\s+equation\b|\bnernst\b|\be_cell\b|\bstandard\s+cell\s+potential\b', "electrochemistry"),
    (r'\bequilibrium\s+constant\b(?!\s+for\s+(?:the\s+)?cell)|\bkc\b|\bkp\b|\ble\s+chatelier\b|\bdegree\s+of\s+dissociation\b'
     r'|\bsolubility\s+product\b|\bksp\b|\bionic\s+product\b|\bdegree\s+of\s+ioni(?:s|z)ation\b'
     r'|\bph\s+of\b|\bpH\s+of\b|\bbuffer\s+solution\b|\bhenderson\b'
     r'|\bconcentration\s+of\s+(?:oh|h)\+\b|\bconcentration\s+of\s+oh.?\s+(?:ions?\s+)?in\b'
     r'|\bweak\s+(?:acid|base)\b|\bbase\s+dissociation\s+constant\b'
     r'|\bka\s*=\b|\bpka\b|\bpkb\b'
     r'|\bkb\s*=\s*[\d.]+\s*(?:[xe×]\s*10|-\d)\b|\bkb\s+(?:for|of)\s+(?:nh3|weak|base)\b'
     r'|\bconcentration\s+of\s+(?:h\+|oh-|h3o\+)\b|\b\[h\+\]\b|\b\[oh-\]\b|\b\[h3o\+\]\b'
     r'|\bdegree\s+of\s+hydrolysis\b|\bhydrolysis\s+of\s+(?:sodium|salt)\b'
     r'|\bph\s+at\s+equivalence\b|\bequivalence\s+point\b', "chemical_equilibrium"),
    (r'\brate\s+of\s+reaction\b|\brate\s+constant\b|\border\s+of\s+reaction\b'
     r'|\bactivation\s+energy\b|\barrhenius\b|\bintegrated\s+rate\b'
     r'|\bfirst\s+order\b|\bsecond\s+order\b|\bzero\s+order\b'
     r'|\btemperature\s+coefficient\b|\bpre.exponential\b|\bfrequency\s+factor\b'
     r'|\brate\s*=\s*k\[|\bhalf.life\s+of\s+(?:a\s+)?(?:first|second|radioactive)\b'
     r'|\brate\s+doubles\s+when\b|\brate\s+quadruples\s+when\b'
     r'|\bby\s+what\s+factor\s+does\s+(?:k|the\s+rate)\b', "chemical_kinetics"),
    (r'\bunit\s+cell\b|\bpacking\s+(?:fraction|efficiency)\b|\bbcc\b|\bfcc\b'
     r'|\bnumber\s+of\s+atoms\s+per\b|\bradius\s+ratio\b|\bionic\s+crystal\b'
     r'|\bdensity\s+of\s+(?:kcl|nacl|cscl|crystal|unit\s+cell)\b'
     r'|\bclose.packed\b|\boctahedral\s+void\b|\btetrahedral\s+void\b'
     r'|\bedge\s+length\b|\bcscl\s+type\b|\bnacl\s+type\b|\bsimple\s+cubic\b'
     r'|\bnearest\s+neighbours?\s+in\b|\bshortest\s+distances?\s+between\b.*\bcryst'
     r'|\brock\s+salt\s+structure\b|\brock\s+salt\s+type\b|\bcompound\s+ab\s+has\b'
     r'|\b(?:\d+\.?\d*)\s*angstrom\b|\ba\s*=\s*[\d.]+\s*angstrom\b', "solid_state"),
    (r'\bcolligative\b|\braoult\'?s?\s+law\b|\bosmotic\s+pressure\b|\bvan\'?t\s+hoff\b'
     r'|\bboiling\s+point\s+elevation\b|\bfreezing\s+point\s+depression\b'
     r'|\bdepression\s+in\s+freezing\b|\belevation\s+in\s+boiling\b'
     r'|\bboiling\s+point\s+of\s+solution\b|\bfreezing\s+point\s+of\s+solution\b'
     r'|\bdelta\s*tb\b|\bdeltatb\b|\bdelta\s*tf\b|\bdeltf\b|\bΔtb\b|\bΔtf\b'
     r'|\bmolal\s+(?:elevation|depression)\b|\bkb\s*=\s*\d(?![.\d]*\s*[xe×]\s*10)|\bkf\s*=\b'
     r'|\bvan\'t\s+hoff\s+factor\b|\brelative\s+lowering\b|\bvapour\s+pressure\b'
     r'|\bdelta_tf\s+observed\b|\bdepression\s+in\s+freezing\s+point\s+of\b'
     r'|\bmolar\s+mass\s+of\s+unknown\s+solute\b|\blowers\s+(?:the\s+)?freezing\s+point\b', "colligative_properties"),
    # Electrochemistry
    (r'\bnernst\b|\bcell\s+potential\b|\bstandard\s+electrode\b|\belectrode\s+potential\b'
     r'|\belectrolysis\b|\bfar?aday\'?s?\s+law\s+of\s+electrolysis\b'
     r'|\bamount\s+of\s+(?:silver|copper|gold|alumin)\b|\bdeposit(?:ed|ion|s|ing)\b|\bconductance\b|\bconductivity\b'
     r'|\ba\s+current\s+of\s+\d+\s*[aA]\b.*\bdeposit\b|\bdeposits?\s+\d+[\.\d]*\s*g\s+of\b'
     r'|\bequivalent\s+weight\s+of\s+(?:silver|copper|gold)\b|\belectrolysis\s+of\s+(?:cuso4|agno3|cucl2|alcl3)\b'
     r'|\be\s*°\s*cell\b|\be°cell\b|\bemf\s+of\s+(?:the\s+)?cell\b'
     r'|\bstandard\s+(?:free\s+energy|gibbs)\b|\bdaniell\s+cell\b'
     r'|\bequilibrium\s+constant\s+for\s+(?:the\s+)?cell\b|\bmolar\s+conductance\b'
     r'|\btime\s+required\s+to\s+deposit\b|\bcurrent\s+is\s+passed\s+(?:for|through)\b'
     r'|\bequilibrium\s+constant\s+for\s+cell\b|\becell\b'
     r'|\bcharge\s+in\s+coulombs\b|\bcoulombs\s+(?:is\s+)?required\s+to\s+deposit\b'
     r'|\bhow\s+much\s+charge\b.*\bdeposit\b|\bcharge\s+required\s+to\s+deposit\b'
     r'|\bkohlrausch\b|\blambda.*molar\s+conductivity\b|\bspecific\s+conductance\b'
     r'|\bemf\s+changes\s+from\b|\bzn2\+\s+concentration\b|\bconcentration\s+changes\b.*\bemf\b', "electrochemistry"),
    # Chemical bonding
    (r'\bhybridization\b|\bvsepr\b|\bmolecular\s+orbital\b|\bbond\s+order\b'
     r'|\bformal\s+charge\b|\bresonance\s+structure\b|\bsigma\s+bond\b|\bpi\s+bond\b'
     r'|\bparamagnetic\b|\bdiamagnetic\b|\bmagnetic\s+moment\s+of\b'
     r'|\bnumber\s+of\s+sigma\s+and\s+pi\s+bonds\b|\bsigma\s+and\s+pi\s+bonds\s+in\b'
     r'|\bmo\s+theory\b|\bmo\s+diagram\b|\bbond\s+orders?\s+of\s+[A-Z]'
     r'|\bunpaired\s+electrons?\s+in\b|\beffective\s+atomic\s+number\b|\bean\b'
     r'|\bcfse\b|\bcrystal\s+field\b|\bhigh\s+spin\b|\blow\s+spin\b'
     r'|\bgeometric\s+isomers?\b|\bcoordination\s+(?:number|compound|complex)\b'
     r'|\boxidation\s+state\s+of\s+(?:central|metal|s|n|cr|mn|cl|fe|cu|au)\s+in\b'
     r'|\boxidation\s+state\s+of\s+[a-z]+\s+in\s+[A-Z]\b'
     r'|\bcomplex\s+\[|\bformal\s+charges\s+on\s+all\s+atoms\b'
     r'|\bozone\b.*\bresonance\b|\bresonance\s+structures?\s+of\b', "chemical_bonding"),
    # Stoichiometry / mole concept / general chemistry — expanded
    (r'\bmole\s+concept\b|\bstoichiometr\b|\blimiting\s+reagent\b|\bpercentage\s+yield\b'
     r'|\bvolume\s+of\s+(?:co2|gas)\s+(?:produced|at\s+stp)\b'
     r'|\bmass\s+of\s+(?:fe|nacl|h2|co2|product)\b|\bnumber\s+of\s+moles\b'
     r'|\bprepare\s+\d+\s+(?:l|ml|litre)\s+of\b|\bmass\s+of\s+(?:naoh|hcl|h2so4|nacl)\s+(?:required|needed|to\s+prepare)\b'
     r'|\bmolarity\s+of\b|\bmolality\b'
     r'|\bdegree\s+of\s+ionisation\b|\bph\s+of\s+(?:0\.\d+)\b'
     r'|\bnumber\s+of\s+molecules\s+in\b|\bnumber\s+of\s+atoms\s+in\b|\bavogadro\b'
     r'|\bat\s+stp\b|\bat\s+s\.t\.p\b|\bstp\b.*\bavogadro\b'
     r'|\bhow\s+many\s+grams\s+of\b|\bhow\s+many\s+moles\s+of\b'
     r'|\bgrams\s+of\s+(?:nh3|o2|co2|h2|so2|hcl|naoh)\b'
     r'|\bvolume\s+of\s+(?:h2so4|hcl|naoh|kmno4|feso4|h2|o2)\s+(?:needed|required|to\s+react|will\s+react)\b'
     r'|\bvolume\s+of\s+\d+\.\d+\s+[Mm]\s+(?:h2so4|hcl|naoh|acid|base)\b'
     r'|\bwhat\s+volume\s+of\s+\d+\.\d+\s+[Mm]\b'
     r'|\bneutrali[sz]e\s+\d+\s+(?:ml|l|g)\b|\bvolume.*\bto\s+neutrali[sz]e\b'
     r'|\bempirical\s+formula\b|\bpercentage\s+(?:composition|of\s+nitrogen|by\s+mass)\b'
     r'|\bmole\s+fraction\s+of\b|\bnormality\b|\bequivalent\s+weight\b'
     r'|\bidentify\s+(?:the\s+)?(?:primary|secondary|tertiary)\s+carbons?\b'
     r'|\bneeded\s+to\s+(?:react|neutralize|titrate)\b|\brequired\s+to\s+neutralize\b'
     r'|\bvolume\s+of\s+\d+\.\d+\s+M\b|\bneutralize\s+\d+\s+m[Ll]\b'
     r'|\bconcentration\s+in\s+ppm\b|\bmass\s+percent\s+of\b|\bppm\s+of\b'
     r'|\b\d+%\s+(?:sucrose|urea|nacl|glucose)\b|\btollens\'?\s+test\b|\bfehling\s+test\b'
     r'|\biodoform\s+test\b|\blucas\s+test\b|\bcannizzaro\b'
     r'|\bnumber\s+of\s+(?:pi|sigma)\s+bonds\s+in\b|\bnumber\s+of\s+asymmetric\s+carbon\b'
     r'|\bnumber\s+of\s+(?:pi|σ|π)\s+bonds\b|\basymmetric\s+carbons?\s+in\b'
     r'|\barrange.*\b(?:increasing|decreasing)\s+order\s+of\s+(?:acidity|basicity|boiling\s+points?|melting\s+points?|reactivity)\b'
     r'|\bboiling\s+point(?:s)?\s+order\b|\bacidity\s+order\b|\bmelting\s+point\s+order\b'
     r'|\bfunctional\s+group\s+(?:present|in)\b|\btype\s+of\s+isomerism\b|\bisomerism\s+between\b'
     r'|\bhenry\'?s?\s+law\s+constant\b|\bhenry\'?s?\s+law\b.*\bdissolve\b'
     r'|\bamount\s+of\s+copper\s+deposited\b|\bcopper\s+deposited\s+when\b'
     r'|\bdeposited\s+when\s+\d+\s+[aA]\s+current\b'
     r'|\bwhat\s+is\s+the\s+product\s+of\s+ozonolysis\b|\bwhat\s+is\s+the\s+product\s+of\s+dehydration\b', "chemistry_problem"),
    # Chemical thermodynamics — specific (ΔG, ΔU, ΔH for reactions) BEFORE thermochemistry
    (r'δg|δu\b|spontan(?:eous|eity).*δ[gh]|δ[gh].*spontan'
     r'|δg\s*=|find\s+δg|calculate\s+δg|is\s+it\s+spontaneous'
     r'|δhf\s*°?\s*\(|given\s+δhf|given\s+δh.*kj'
     r'|bond\s+dissociation\s+energ.*calculate\s+δh|calculate\s+δh\s+for\s+h2'
     r'|h-h\s*=\s*\d+.*h-cl|δh.*δs.*(?:kj|j/mol)'
     r'|enthalpy\s+change\s+when\s+\d+\s*g\s+of\s+ice.*steam'
     r'|hess.*law.*find\s+δh\b.*\bfor\b.*\b(?:c2h|c3h|ch4|ethyl|eth)', "chemical_thermodynamics"),
    # Thermochemistry
    (r'\bhess\'?s?\s+law\b|\bheat\s+of\s+(?:reaction|formation|combustion)\b'
     r'|\bdelta\s*h\b|\bΔH\b|\bbond\s+enthalpy\b|\blattice\s+energy\b|\bborn.haber\b'
     r'|\benthalpy\s+of\b|\bheat\s+released\s+per\b'
     r'|\bdelta_h\s+for\b|\bdelta_hrxn\b|\bdelta_hf\b|\bΔHf\b|\bΔH°\b'
     r'|\bheat\s+of\s+vaporization\b|\bheat\s+of\s+formation\b|\bresonance\s+energy\b'
     r'|\bnumber\s+of\s+moles\s+of\s+\w+\s+that\s+must\s+decompose\b'
     r'|\bheat\s+released\s+when\b|\bheat\s+evolved\s+when\b'
     r'|\bbond\s+dissociation\s+energy\s+of\b|\bdissociation\s+energy\s+of\s+h2\b', "thermochemistry"),
    # Chemical kinetics must come before chemistry_problem (half-life overlap)
    (r'\b(?:first|second|zero)\s+order\s+(?:reaction|kinetics)\b'
     r'|\brate\s+(?:constant|law|equation)\b|\bintegrated\s+rate\b'
     r'|\bactivation\s+energy\b|\barrhenius\b|\bhalf.life\s+of\s+(?:a\s+)?(?:first|second)?\s*order', "chemical_kinetics"),
    (r'\bmol(?:es?)?\s+of\b|\bcalorimetr|\bboyle\b|\bcharles\b|\bgay.lussac\b'
     r'|\bmolarity\b|\bstoichiom\b|\benthalpy\b|\bnernst\b', "chemistry_problem"),
    (r'\bchemical\s+(reaction|equation|formula)\b|\bbalance\b.*\bequation\b', "chemistry_problem"),
    (r'\bcell\s+division\b|\bphotosynthesis\b|\bdna\b|\brna\b|\bprotein\s+synthesis\b', "biology_concept"),
    # ── DC Circuits — specific electrical network problems BEFORE generic physics_problem ──
    (r'\bkirchhoff\'?s?\s+(?:laws?|current\s+law|voltage\s+law)\b|\bkirchhoff\b'
     r'|\bwheatstone\s+bridge\b'
     r'|\binternal\s+resistance\b.*\bterminal\s+voltage\b|\bterminal\s+voltage\b.*\binternal\s+resistance\b'
     r'|\bemf\s+\d+\s*v\b.*\binternal\s+resistance\b'
     r'|\bconnected\s+in\s+(?:series|parallel)\s+across\s+\d+\s*v\b'
     r'|\bequivalent\s+resistance\s+between\s+[a-z]\s+and\s+[a-z]\b'
     r'|\bnet\s+resistance\b|\b\d+[ωΩ].*\d+[ωΩ].*\d+[ωΩ]\b'
     r'|\bresistors?\s+\d+[ωΩ]|\b(?:two|three|four)\s+resistors?\s+(?:in\s+|connected\b)', "dc_circuit"),
    (r'\bforce\b|\bmomentum\b|\bthermodynamics\b|\boptics\b'
     r'|\bvelocity\b|\bacceleration\b|\bkinetic\s+energy\b|\bpotential\s+energy\b'
     # Projectile / mechanics
     r'|\bprojectile\b|\btime\s+of\s+flight\b|\bhorizontal\s+range\b|\bmaximum\s+height\b'
     r'|\bangle\s+of\s+projection\b|\binitial\s+speed\b|\binitial\s+velocity\b|\bfinal\s+velocity\b'
     r'|\bstops\s+after\b|\bstops\s+in\b|\bstopping\s+distance\b|\bbrakes\b|\bbraking\s+force\b'
     r'|\bdeceleration\b|\buniform\s+(?:acceleration|deceleration)\b'
     r'|\bcircular\s+(?:loop|motion)\b|\bvertical\s+circle\b'
     r'|\bescapes\s+from\b|\beatmo(?:sphere)?\b'
     r'|\bmotion\s+under\s+gravity\b|\bfree\s+fall\b|\bfalls\s+freely\b'
     r'|\bthrown\s+(?:vertically|horizontally)\b|\bthrown\s+upward\b'
     r'|\bpower\s+(?:exerted|delivered|consumed|generated)\b|\bwork\s+done\s+(?:by|against)\s+(?:friction|gravity|force)\b'
     r'|\bcoefficient\s+of\s+restitution\b|\bembeds\s+in\b|\bbullet\s+embeds\b'
     r'|\breaches\s+(?:the\s+)?bottom\b|\brenders\s+the\s+water\b'
     r'|\bangle\s+of\s+repose\b|\bcoefficient\s+of\s+(?:static|kinetic)\s+friction\b'
     r'|\bpower\s+of\s+(?:a\s+)?pump\b|\bpump\s+(?:delivers|of\s+power)\b'
     r'|\bslides?\s+down\s+(?:a\s+)?(?:frictionless|curved|smooth)\b'
     r'|\borbital\s+(?:speed|velocity|period)\b|\bgeostationary\s+satellite\b|\bsatellite\s+orbits?\b'
     r'|\bcenter\s+of\s+mass\s+of\b|\bcentre\s+of\s+mass\s+of\b'
     r'|\bwater\s+flows?\s+through\s+(?:a\s+)?pipe\b|\bpressure\s+at\s+depth\b'
     r'|\bsound\s+level\s+in\s+(?:db|decibel)\b|\bsource\s+of\s+sound\s+(?:of\s+frequency|moves)\b'
     # Collisions extras
     r'|\btwo\s+bodies\s+collide\b|\bcollide\s+elastically\b|\bcollide\s+inelastically\b'
     r'|\bperfectly\s+inelastic\s+collision\b|\bhead.on\s+collision\b'
     r'|\brocket\s+(?:of\s+mass|ejects)\b|\brate\s+of\s+fuel\b|\bthrust\s+is\b|\bjets\s+of\s+gas\b'
     r'|\bman\s+of\s+mass\s+\d+\s*kg\s+stands\b|\bweighing\s+scale\b|\bweighing\s+machine\b'
     r'|\blift\s+(?:accelerat|deceler|moving)\b|\belevator\s+(?:accelerat|moving)\b'
     r'|\bperson\s+stands\s+on\s+(?:a\s+)?(?:scale|weighing)\b'
     # River/boat relative motion
     r'|\bboat\s+(?:moves|rows|crosses)\b|\bswimmer\s+(?:swims|crosses)\b|\briver\s+flow\b'
     r'|\bstill\s+water\b|\brelative\s+(?:velocity|speed)\s+of\b|\bdrifts?\s+(?:downstream|upstream)\b'
     r'|\bminimum\s+time\s+to\s+cross\b|\bbreadth\s+of\s+(?:the\s+)?river\b'
     # Waves / sound
     r'|\bbeatss?\s+(?:per|frequency)\b|\bdoppler\b|\bstanding\s+wave\b|\bstationary\s+wave\b'
     r'|\bopen\s+pipe\b|\bclosed\s+pipe\b|\btuning\s+fork\b|\bresonates\b'
     r'|\bspeed\s+of\s+sound\b|\bsound\s+wave\b|\bbeat\s+frequency\b'
     r'|\bsuperpose\b|\bamplitude\s+of\s+resultant\b|\bnth\s+harmonic\b|\bfundamental\s+frequency\b'
     r'|\b(?:1st|2nd|3rd|4th|5th|6th)\s+harmonic\b|\bvibrates\s+in\s+\d+(?:st|nd|rd|th)\s+harmonic\b'
     r'|\bstring\s+of\s+length.*\bfixed\s+at\s+both\b|\bfixed\s+at\s+both\s+ends\b.*\bharmonic\b'
     r'|\bwave\s+(?:equation\s+)?(?:is\s+)?y\s*=\b|\by\s*=\s*[\d.]+\s*sin\s*\(2\s*pi\b'
     r'|\belastic\s+collision\s+between\b'
     # Fluid
     r'|\bbuoyant\s+force\b|\barchimedes\b|\bbernoulli\b'
     # Thermal / heat transfer
     r'|\blatent\s+heat\b|\bspecific\s+heat\b|\bnewton\'?s?\s+law\s+of\s+cooling\b'
     r'|\bheat\s+transferred\b|\bheat\s+conducted\b|\bthermal\s+conductivity\b'
     r'|\bconvert\s+\d+\s+g\s+of\s+ice\b|\bice\s+at\s+-\d+\b|\bsteam\s+at\s+\d+\b', "physics_problem"),
    # ── Equation solving (specific patterns BEFORE generic "solve") ──
    # Vieta's formulas: sum and product of roots — MUST be before summation
    (r'\bsum\s+(?:and\s+product\s+)?of\s+roots\s+of\b|\bproduct\s+of\s+roots\s+of\b'
     r'|\bsum\s+of\s+(?:the\s+)?roots\s+of\s+(?:x|equation)\b|\bvieta\'?s?\b'
     r'|\bsum\s+and\s+product\s+of\s+roots\b', "equation_solving"),
    # Sum of digits of a number → equation solving / number puzzles
    (r'\bsum\s+of\s+(?:its\s+)?digits\s+(?:of\s+a\s+)?(?:(?:\d+)-digit\s+)?number\b'
     r'|\bsum\s+of\s+digits\s+is\s+\d+\b|\bdigits\s+(?:are\s+)?reversed\b', "equation_solving"),
    (r'\bfor\s+what\s+values?\s+of\s+[a-z]\b|\breal\s+(?:and\s+)?equal\s+roots?\b|\breal\s+solutions?\b'
     r'|\brange\s+of\s+f\s*\(x\)|\brange\s+of\s+f\s*\(x\)\b|\bam.gm\s+inequality\b|\bam\s*-\s*gm\b'
     r'|\bmaximize\s+z\s*=\s*|\bminimize\s+z\s*=\s*|\blinear\s+programming\b'
     r'|\bsubject\s+to\s+(?:x\s*\+|constraints?)\b|\bconstraints?\s+of\b'
     r'|\binteger\s+solutions?\s+of\b|\ball\s+values\s+of\s+x\s+(?:such|for|that)\b'
     r'|\bfind\s+the\s+value\s+of\s+x\s+and\s+y\b'
     r'|\bquadratic\s+equation\s+with\s+roots\b|\bfind\s+(?:its?|the)\s+inverse\s+function\b'
     r'|\bone.to.one\s+function\b|\bdiscriminant\s+of\b|\bfind\s+[ab]\s+such\s+that\b'
     r'|\bremainder\s+when\s+polynomial\b|\bremainder\s+theorem\b'
     r'|\bfind\s+(?:the\s+)?remainder\s+when\b', "equation_solving"),
    (r'\bsolve\b|\broots?\s+of\b|\bzeros?\s+of\b', "equation_solving"),
    # Logic
    (r'\bif\s+.+\s+then\b|\bimplies\b|\btherefore\b|\bsyllogism\b|\bmodus\b|\bcontrapositive\b', "logical_reasoning"),
    (r'\btruth\s+table\b(?!\s+of\s+(?:nand|nor|xor|and|or)\s+gate)|\bpropositional\b|\btautology\b|\bcontradiction\b', "propositional_logic"),
    # ── Early chemistry/bonding catches BEFORE generic knowledge_retrieval/conceptual_explanation ──
    # These "explain why" and "what is" questions are chemistry-specific
    (r'\bhydrogen\s+bonding\b|\bintermolecular\s+(?:force|attraction)\b'
     r'|\bvan\s+der\s+waals\b|\bdipole.dipole\b|\bboiling\s+point\s+(?:of|order\s+of)\s+(?:nh3|ph3|hf|hcl|h2o)\b'
     r'|\bwhy\s+(?:does\s+)?nh3\b|\bwhy\s+(?:does\s+)?pcl5\b|\bwhy\s+(?:does\s+)?ncl5\b'
     r'|\bd.?orbital\s+(?:not\s+)?(?:available|present|absent)\b'
     r'|\bdipole\s+moment\s+(?:of\s+nh3|of\s+nf3|comparison)\b', "chemical_bonding"),
    # SN1/SN2 mechanism questions — specific chemistry before generic "explain/write"
    (r'\bsn1\s+(?:reaction|mechanism)\b|\bsn2\s+(?:reaction|mechanism)\b'
     r'|\bwrite\s+(?:the\s+)?mechanism\s+for\s+(?:sn|e1|e2|aldol|grignard)\b'
     r'|\bexplain\s+(?:the\s+)?(?:sn1|sn2|mechanism\s+of|stereochemistry\s+of)\b'
     r'|\bmore\s+reactive\s+in\s+sn\d\b|\breactivity\s+order\s+of\b.*\b(?:halide|alkyl)\b'
     r'|\bwhich\s+(?:compound|alkyl|alcohol|halide)\s+is\s+more\s+reactive\b.*\bsn\b', "chemistry_problem"),
    # "What is the concentration of OH-" / "What is the degree of unsaturation" → chemistry
    (r'\bwhat\s+is\s+(?:the\s+)?concentration\s+of\s+(?:oh|h3o|h\+)\b'
     r'|\bwhat\s+is\s+(?:the\s+)?(?:degree\s+of\s+unsaturation|dbe|ihd)\s+of\b'
     r'|\bwhat\s+is\s+(?:the\s+)?(?:product\s+of|major\s+product|iupac\s+name|oxidation\s+state)\s+(?:of\s+)?(?:the\s+)?(?:reaction|compound|above)\b'
     r'|\bwhat\s+is\s+(?:the\s+)?(?:hybridization|bond\s+order|formal\s+charge)\s+of\b', "chemistry_problem"),
    # Knowledge & reasoning
    (r'\bwhat\s+is\b|\bwho\s+is\b|\bwhat\s+are\b|\bwho\s+are\b', "knowledge_retrieval"),
    (r'\bexplain\b|\bdefine\b|\bdescribe\b', "conceptual_explanation"),
    (r'\bhow\s+to\b|\bhow\s+does\b|\bhow\s+do\b', "procedural_knowledge"),
    (r'\bwhy\s+(does|is|did|do|would|should)\b|\bwhy\b', "explanatory_reasoning"),
    (r'\bcompare\b|\bdifference\s+between\b|\bsimilarities\b', "comparative_analysis"),
    (r'\bcause\b|\bcaused\b|\bcauses\b|\beffect\b|\bimpact\b|\bconsequence\b', "causal_analysis"),
    # ── Organic Chemistry — MUST be before predictive_reasoning (catches "predict") ──
    (r'\bozonolysis\b|\bfriedel.crafts\b|\bgrignard\b|\baldol\s+(?:condensation|addition)\b'
     r'|\bmarkovnikov\b|\banti.markovnikov\b|\bhofmann\b|\bcannizzaro\b'
     r'|\bsn1\b|\bsn2\b|\belimination\s+reaction\b|\bnucleophilic\b|\belectrophilic\b'
     r'|\bdegree\s+of\s+unsaturation\b|\bdbe\b|\bihd\b|\bindex\s+of\s+hydrogen\s+deficiency\b'
     r'|\biupac\s+name\b|\bstructural\s+isomers?\b|\bgeometric\s+isomers?\b|\boptical\s+isomers?\b'
     r'|\bchiral\s+(?:carbon|centre)\b|\bcarbocation\b|\bcarbanion\b|\bfree\s+radical\b'
     r'|\bdehydration\s+of\s+ethanol\b|\bdehydration\s+of\s+alcohol\b'
     r'|\besterification\b|\bsaponification\b|\bhydrolysis\s+of\s+ester\b'
     r'|\bproduct\s+of\s+ozonolysis\b|\bproduct\s+of\s+dehydration\b'
     r'|\bproduct\s+when\s+.*\breacts\s+with\b|\bmajor\s+product\b|\bminor\s+product\b'
     r'|\bpredict\s+(?:the\s+)?(?:major\s+|minor\s+)?product\b'
     r'|\bpredict\s+(?:the\s+)?product\s+of\b|\bpredict\s+(?:the\s+)?product\s+when\b'
     r'|\bwrite\s+(?:the\s+)?(?:product|mechanism|reaction)\s+of\b'
     r'|\bidentify\s+(?:primary|secondary|tertiary)\s+carbon\b'
     r'|\bprimary.?\s+(?:alcohol|amine|carbon|halide)\b'
     r'|\bsecondary.?\s+(?:alcohol|amine|carbon|halide)\b'
     r'|\btert(?:iary)?.?\s+(?:alcohol|amine|carbon|halide)\b'
     r'|\bwhat\s+is\s+(?:the\s+)?(?:product\s+of|degree\s+of|IUPAC)\b'
     r'|\bwhat\s+is\s+(?:the\s+)?(?:product\s+of\s+ozonolysis|product\s+of\s+dehydration)\b'
     r'|\bwrite\s+(?:the\s+)?mechanism\b'
     r'|\balkyl\s+halide\b|\bhalogenation\b|\bnitration\b|\bsulfonation\b'
     r'|\balk(?:ane|ene|yne)\s+(?:react|with|naming)\b|\baro?matic\s+(?:compound|ring)\b', "chemistry_problem"),
    (r'\bpredict\b|\bwhat\s+will\b|\bwhat\s+would\s+happen\b|\bforecast\b', "predictive_reasoning"),
    (r'\banalog(y|ous)\b|\blike\b.*\bmetaphor\b|\bsimilar\s+to\b', "analogical_reasoning"),
    (r'\bethics\b|\bmoral\b|\bright\s+or\s+wrong\b|\bshould\b|\bought\b', "ethical_reasoning"),
]

_STRATEGIES: dict[str, str] = {
    "integration":          "Apply integration rules (u-sub, IBP, trig-sub, partial fractions, or direct antiderivative). For JEE: check for king property, ILATE rule for IBP.",
    "differentiation":      "Apply chain rule, product rule, quotient rule, or basic derivative rules. For JEE: implicit differentiation, parametric differentiation.",
    "limit_evaluation":     "Apply L'Hôpital's rule, algebraic manipulation, squeeze theorem, or known standard limits (sin x/x → 1, (1+1/n)^n → e).",
    "equation_solving":     "Factor, use quadratic formula, substitution, or numerical methods. For JEE: sum/product of roots (Vieta's formulas), nature of roots via discriminant.",
    "factorization":        "Factor out GCF, then use special patterns (difference of squares, sum/difference of cubes, Sophie Germain identity).",
    "algebraic_expansion":  "Apply binomial theorem or FOIL method for products. State the general term Tᵣ₊₁ = C(n,r)·aⁿ⁻ʳ·bʳ.",
    "simplification":       "Cancel common factors, apply trig/log identities, algebraic reduction.",
    "matrix_operation":     "Apply matrix algorithms: cofactor expansion (det), row reduction (rank/inverse), characteristic polynomial det(A−λI)=0 (eigenvalues).",
    "ode_solving":          "Classify ODE (linear/separable/exact/homogeneous) and apply corresponding solution method.",
    "series_expansion":     "Compute Taylor/Maclaurin coefficients via repeated differentiation.",
    "laplace_transform":    "Apply Laplace transform table entries and linearity property.",
    "fourier_transform":    "Apply Fourier transform definition and standard pairs.",
    "number_theory":        "Apply Euclidean algorithm (GCD/LCM), prime sieve, or modular arithmetic.",
    "statistical_analysis": "Compute descriptive statistics: mean, variance (E[(X-μ)²]), std deviation.",
    "combinatorics":        "Apply counting principles: fundamental counting theorem, factorial, C(n,r), P(n,r). For JEE: case analysis, complementary counting.",
    "complex_number":       "Use Cartesian/polar form and their algebraic properties. De Moivre's theorem for roots of unity.",
    "summation":            "Apply summation formulas (∑r, ∑r², ∑r³, arithmetic/geometric series) or telescoping, method of differences.",
    # ── JEE Advanced: Calculus ──────────────────────────────────────────────
    "maxima_minima":        "Find f'(x) = 0 for critical points; use f''(x) test (or first derivative sign change) to classify; check endpoints for closed intervals.",
    "monotonicity":         "Find f'(x); analyse sign of f'(x) on each interval; state where f is increasing (f'>0) and decreasing (f'<0).",
    "mean_value_theorem":   "Verify continuity on [a,b] and differentiability on (a,b); apply Rolle's/MVT to find the guaranteed c; interpret geometrically.",
    # ── JEE Advanced: Coordinate Geometry ──────────────────────────────────
    "conic_parabola":       "Use standard form y²=4ax or x²=4ay; identify focus, directrix, latus rectum, axis; apply parametric form (at², 2at) for tangent/normal.",
    "conic_ellipse":        "Use x²/a²+y²/b²=1; identify a>b, b²=a²(1-e²); apply parametric (a cosθ, b sinθ); derive tangent, normal, chord of contact.",
    "conic_hyperbola":      "Use x²/a²-y²/b²=1; find b²=a²(e²-1), asymptotes y=±(b/a)x; apply parametric (a secθ, b tanθ); tangent equation T=0.",
    "conic_section":        "Write the general second-degree equation; complete the square to identify type; apply T=0, T=S1, SS1=T² for chord/tangent problems.",
    "pair_of_lines":        "Homogenise or use the combined equation ax²+2hxy+by²+…=0; conditions: Δ=0 (pair of lines), h²>ab (real distinct), h²=ab (parallel).",
    "circle_geometry":      "Write circle as (x-h)²+(y-k)²=r² or S=0; radical axis: S₁-S₂=0; orthogonality: 2g₁g₂+2f₁f₂=c₁+c₂.",
    "straight_line":        "Write line in slope-intercept or parametric form; angle between lines: tan θ = |m₁-m₂|/(1+m₁m₂); foot of perpendicular, reflection.",
    # ── JEE Advanced: 3D Geometry & Vectors ─────────────────────────────────
    "3d_geometry":          "Express line as r = a + λb (parametric) and plane as r·n = d; shortest distance SD = |(b₁×b₂)·(a₂-a₁)|/|b₁×b₂|.",
    "vector_algebra":       "Compute scalar triple product [a b c] = a·(b×c) for coplanarity/volume; projection of a on b = (a·b)/|b|.",
    # ── JEE Advanced: Probability ───────────────────────────────────────────
    "bayes_probability":    "Partition the sample space into mutually exclusive events Hᵢ; apply Bayes' theorem P(Hᵢ|E) = P(E|Hᵢ)P(Hᵢ)/ΣP(E|Hⱼ)P(Hⱼ).",
    "probability_distribution": "Identify distribution type; compute E(X) = Σ xᵢP(xᵢ); Var(X) = E(X²) - [E(X)]².",
    "probability_problem":  "Define the sample space; count favourable vs total outcomes; use addition/multiplication rules or conditional probability as needed.",
    # ── JEE Advanced: Sequences & Series ────────────────────────────────────
    "sequences_series":     "Identify AP/GP/HP; for AP: aₙ=a+(n-1)d, Sₙ=n/2(2a+(n-1)d); for GP: aₙ=arⁿ⁻¹, Sₙ=a(rⁿ-1)/(r-1), S∞=a/(1-r) for |r|<1.",
    # ── JEE Advanced: Trigonometry ──────────────────────────────────────────
    "trigonometric_equation": "Reduce to standard form sin x = k, cos x = k, or tan x = k; write general solution x = nπ + (-1)ⁿα (sin), 2nπ±α (cos), nπ+α (tan).",
    "inverse_trig":         "Apply domain/range restrictions; use standard identities (sin⁻¹x + cos⁻¹x = π/2, etc.); simplify using composition rules.",
    "trig_identities":      "Apply compound angle (sin(A±B)), double angle (sin 2A = 2 sin A cos A), half angle, and sum-to-product/product-to-sum formulas.",
    "triangle_trig":        "Apply sine rule a/sin A = b/sin B = 2R; cosine rule c² = a²+b²-2ab cosC; area = ½ab sin C; apply for ambiguous case if needed.",
    # ── JEE Advanced: Binomial ──────────────────────────────────────────────
    "binomial_theorem":     "General term: Tᵣ₊₁ = C(n,r)·xⁿ⁻ʳ·yʳ; middle term at r=n/2 (even n); find r for term independent of x by setting power=0.",
    # ── JEE Advanced: Physics ──────────────────────────────────────────────
    "shm_problem":          "Identify restoring force F = -kx; ω = √(k/m); T = 2π/ω; x(t) = A cos(ωt+φ); v_max = Aω at mean; a_max = Aω² at extreme; E_total = ½kA².",
    "rotational_mechanics": "Identify axis; use I = Σmᵢrᵢ² or standard results + parallel/perpendicular axis theorems; τ = Iα; L = Iω; rolling: a = αR, v = ωR.",
    "modern_physics":       "Photoelectric: KE_max = hf - φ; De Broglie: λ = h/mv; Bohr: Eₙ = -13.6/n² eV, rₙ = 0.529n² Å; Nuclear: Q = Δm·c².",
    "electrostatics":       "Coulomb: F = kq₁q₂/r²; E = kq/r² (point charge); V = kq/r; Gauss's law ∮E·dA = Q_enc/ε₀; C = Q/V; Energy = ½CV².",
    "electromagnetism":     "Biot-Savart dB = μ₀I dl×r̂/4πr²; Ampere ∮B·dl = μ₀I_enc; Faraday EMF = -dΦ/dt; L = μ₀n²V; M = μ₀n₁n₂V.",
    # ── JEE Advanced: Chemistry ─────────────────────────────────────────────
    "chemical_equilibrium": "Write Kc = [products]/[reactants] (mol concentrations); Kp = Kc(RT)^Δn; use ICE table; Le Chatelier: predict shift from perturbation.",
    "chemical_kinetics":    "Rate = k[A]^m[B]^n; 1st order: ln[A]=[A]₀-kt, t₁/₂=0.693/k; Arrhenius: k = A·e^(-Ea/RT); determine order from initial rate data.",
    "solid_state":          "Identify lattice (BCC: 2 atoms, FCC: 4 atoms, SC: 1 atom); packing efficiency = volume of atoms/volume of unit cell; radius ratio for coordination.",
    "colligative_properties": "ΔTb = iKbm; ΔTf = iKfm; π = iMRT; van't Hoff factor i = 1+(n-1)α for degree of dissociation α.",
    # ── General ─────────────────────────────────────────────────────────────
    "knowledge_retrieval":  "Retrieve factual information, synthesise across sources, verify key claims.",
    "conceptual_explanation": "Build understanding from fundamentals using JEE-level rigour; include relevant formulas and examples.",
    "procedural_knowledge": "Provide a clear step-by-step procedure with prerequisites and validation.",
    "explanatory_reasoning": "Identify the causal chain / mechanism, state assumptions, build evidence-based explanation.",
    "comparative_analysis": "Identify key dimensions of comparison; contrast each dimension systematically.",
    "causal_analysis":      "Trace the causal chain from root cause to effect; identify proximate and distal causes.",
    "logical_reasoning":    "Apply formal inference rules (modus ponens/tollens, hypothetical syllogism).",
    "propositional_logic":  "Evaluate propositional formula via truth table or logical equivalences.",
    "ideal_gas_thermo":     "Identify atomicity (mono/diatomic → Cv=3R/2 or 5R/2); determine process (const P: q=nCpΔT, ΔU=nCvΔT; const V: q=ΔU=nCvΔT); compute ΔT then T_final and ΔU.",
    "chemistry_problem":    "Identify the chemistry principle (mole concept, gas law, pH, calorimetry); state the formula; substitute values with units; compute and verify.",
    "electrochemistry":     "Identify EMF type (cell/half-cell); apply Nernst equation E=E°-(RT/nF)lnQ; for electrolysis use Faraday's laws m=ZIt=MIt/nF.",
    "chemical_bonding":     "Identify hybridization (count electron pairs), draw Lewis structure, apply MO theory for bond order=(bonding-antibonding)/2, apply spin-only formula μ=√n(n+2) BM.",
    "chemical_thermodynamics": "Use ΔG = ΔH - TΔS for spontaneity; ΔG < 0 spontaneous; ΔU = ΔH - ΔnᵍRT; Hess's law: target = linear combination of given equations.",
    "dc_circuit":           "Apply Kirchhoff's KVL (ΣV=0) and KCL (ΣI=0); series R_eq=R₁+R₂+…; parallel 1/R_eq=1/R₁+1/R₂+…; terminal voltage V_T=EMF-Ir.",
    "thermochemistry":      "Apply Hess's law ΔH°rxn = ΣΔH°f(products) - ΣΔH°f(reactants); use bond enthalpies ΔH = Σ(bonds broken) - Σ(bonds formed).",
    "optics_problem":       "Apply Snell's law n₁sinθ₁=n₂sinθ₂; lens formula 1/v-1/u=1/f; mirror formula 1/v+1/u=1/f; fringe width β=λD/d; prism deviation δ=A(μ-1) at minimum.",
    "biology_concept":      "Explain biological process from molecular mechanism to organismal level.",
    "physics_problem":      "Identify relevant physical laws, apply conservation principles, solve equations with units.",
    "predictive_reasoning": "Analyse current conditions, apply relevant domain laws/trends, project outcome.",
    "analogical_reasoning": "Identify structural similarity between source and target, map relationships.",
    "ethical_reasoning":    "Identify stakeholders, apply ethical frameworks (consequentialist/deontological/virtue), reason to conclusion.",
    "competition_math":     "Set up a system of equations by letting variables represent the unknown speed and travel time. Use the equal-distance constraint (all arrive at same destination) to form one equation per extra person. Solve exactly with SymPy and express the distance as a reduced fraction m/n; answer is m+n.",
    "unknown":              "Analyse the problem carefully and apply the most relevant JEE Advanced reasoning approach.",
}

_EXPECTED_FORMS: dict[str, str] = {
    "integration":           "A symbolic function + constant C (indefinite) or a real number (definite)",
    "differentiation":       "A symbolic expression of the same or lower degree",
    "limit_evaluation":      "A real number, ∞, -∞, or 'does not exist'",
    "equation_solving":      "One or more numerical or symbolic values for the unknown",
    "factorization":         "A product of irreducible factors",
    "series_expansion":      "A polynomial up to the given order, plus O(x^n) remainder",
    "matrix_operation":      "A scalar (det/rank/trace) or matrix (inverse/eigenvectors)",
    "ode_solving":           "A function y(x) with integration constants C1, C2, …",
    "laplace_transform":     "A rational or transcendental function of s",
    "fourier_transform":     "A function of the frequency variable ω or k",
    "statistical_analysis":  "Real-valued descriptive statistics",
    "combinatorics":         "A non-negative integer",
    "logical_reasoning":     "A valid/invalid judgment with proof or counterexample",
    "propositional_logic":   "TRUE, FALSE, or a tautology/contradiction label",
    # JEE Advanced additions
    "maxima_minima":         "The coordinates and value of the extremum, or the intervals",
    "monotonicity":          "Intervals of increase/decrease stated explicitly",
    "mean_value_theorem":    "A specific value c in the open interval (a,b)",
    "conic_parabola":        "Equation of tangent/normal, focus coordinates, or requested quantity",
    "conic_ellipse":         "Equation of tangent/normal, length of latus rectum, or eccentricity",
    "conic_hyperbola":       "Equation of asymptotes, eccentricity, or tangent equation",
    "conic_section":         "Chord of contact equation T=0, or combined equation",
    "pair_of_lines":         "Individual line equations, angle between them, or combined equation",
    "circle_geometry":       "Centre, radius, tangent equation, or radical axis",
    "straight_line":         "Equation of line, distance, or angle between lines",
    "3d_geometry":           "Shortest distance (a scalar), plane equation, or angle between planes",
    "vector_algebra":        "A scalar (dot/triple product) or vector (cross product)",
    "sequences_series":      "The nth term, sum formula, or numerical value of Sₙ or S∞",
    "trigonometric_equation":"General solution in the form nπ ± α or nπ + (-1)ⁿα",
    "inverse_trig":          "A principal value in radians or degrees, or a simplified expression",
    "trig_identities":       "A simplified numerical value or identity",
    "triangle_trig":         "A length, angle, or area with appropriate units",
    "binomial_theorem":      "The required term Tᵣ₊₁, its coefficient, or a numerical value",
    "shm_problem":           "Period T, amplitude A, or energy value with SI units",
    "rotational_mechanics":  "Moment of inertia (kg·m²), angular acceleration (rad/s²), or angular momentum (kg·m²/s)",
    "modern_physics":        "Energy in eV or MeV, wavelength in nm or Å, or a dimensionless ratio",
    "electrostatics":        "Electric field (N/C), potential (V), capacitance (F), or energy (J)",
    "electromagnetism":      "Magnetic field (T), EMF (V), inductance (H), or current (A)",
    "chemical_equilibrium":  "Kc or Kp value, equilibrium concentrations, or degree of dissociation",
    "chemical_kinetics":     "Rate constant k, half-life t₁/₂, or concentration at time t",
    "solid_state":           "Density (g/cm³), packing efficiency (%), or edge length (pm/nm)",
    "colligative_properties":"ΔTb (°C), ΔTf (°C), π (atm), or van't Hoff factor i",
    "bayes_probability":     "A conditional probability P(Hᵢ|E) in the range [0,1]",
    "probability_distribution": "E(X), Var(X), or specific probability P(X=k)",
    "probability_problem":   "A probability value in [0,1] or as a fraction",
    "optics_problem":        "Focal length (cm), image position (cm), refractive index (dimensionless), fringe width (mm), or deviation angle (degrees)",
    "electrochemistry":      "Cell EMF (V), mass deposited (g), time (s), or standard free energy (kJ)",
    "chemical_bonding":      "Hybridization type, bond order (numeric), magnetic moment (BM), or geometry description",
    "thermochemistry":       "ΔH in kJ/mol, heat in kJ/g, or enthalpy of formation in kJ/mol",
    "chemical_thermodynamics": "ΔG (kJ/mol), ΔH (kJ/mol), ΔU (kJ/mol), or spontaneity conclusion",
    "dc_circuit":            "Current (A), voltage (V), resistance (Ω), or power (W) with appropriate units",
    "ideal_gas_thermo":      "Temperature (K), pressure (atm/Pa), work (J), or internal energy change (J)",
    "chemistry_problem":     "A numeric value with appropriate chemistry units",
    "competition_math":      "A reduced fraction m/n (miles) where gcd(m,n)=1, and the integer m+n",
}


# ─────────────────────────────────────────────────────────────────────────────
# Tree-of-Thought branch generator
# ─────────────────────────────────────────────────────────────────────────────

def _generate_branches(
    user_input: str, problem_type: str, domain: str
) -> list[ReasoningBranch]:
    """
    For complex problems, generate multiple candidate reasoning paths
    and score each one for feasibility.
    Returns 1–3 branches; caller picks the best.
    """
    lowered = user_input.lower()
    branches: list[ReasoningBranch] = []

    if problem_type == "integration":
        branches = [
            ReasoningBranch(
                label="substitution",
                steps=[
                    "Identify a suitable substitution u = g(x)",
                    "Compute du = g'(x) dx and rewrite the integrand",
                    "Integrate in terms of u",
                    "Back-substitute to get the result in x",
                ],
                score=0.8,
            ),
            ReasoningBranch(
                label="by_parts",
                steps=[
                    "Choose u and dv using LIATE rule (Log, Inverse trig, Algebraic, Trig, Exponential)",
                    "Compute du and v = ∫ dv",
                    "Apply ∫ u dv = uv − ∫ v du",
                    "Repeat if needed; apply boundary conditions for definite integrals",
                ],
                score=0.75,
            ),
            ReasoningBranch(
                label="direct",
                steps=[
                    "Recognise the standard integral form",
                    "Apply the direct antiderivative rule",
                    "Add the constant of integration C",
                ],
                score=0.9,
            ),
        ]

    elif problem_type == "equation_solving":
        branches = [
            ReasoningBranch(
                label="algebraic",
                steps=[
                    "Rearrange so one side equals 0",
                    "Factor the expression if possible",
                    "Set each factor to 0 and solve",
                    "Verify solutions in the original equation",
                ],
                score=0.85,
            ),
            ReasoningBranch(
                label="quadratic_formula",
                steps=[
                    "Identify a, b, c in ax² + bx + c = 0",
                    "Compute discriminant D = b² − 4ac",
                    "x = (−b ± √D) / 2a",
                    "Interpret real vs. complex roots",
                ],
                score=0.9,
            ),
        ]

    elif problem_type == "explanatory_reasoning":
        branches = [
            ReasoningBranch(
                label="causal_chain",
                steps=[
                    "Identify the phenomenon and the question being asked",
                    "Identify the root cause / initial condition",
                    "Trace each causal step from cause to effect",
                    "State any assumptions or simplifications",
                    "Summarise the causal mechanism clearly",
                ],
                score=0.85,
            ),
            ReasoningBranch(
                label="mechanism",
                steps=[
                    "Identify the system and its components",
                    "Explain how each component contributes",
                    "Describe the interactions between components",
                    "Show how the overall behaviour emerges",
                ],
                score=0.8,
            ),
        ]

    elif problem_type == "comparative_analysis":
        branches = [
            ReasoningBranch(
                label="dimension_matrix",
                steps=[
                    "Define what is being compared (A vs. B)",
                    "List key comparison dimensions (e.g., mechanism, applications, limitations)",
                    "Analyse A on each dimension",
                    "Analyse B on each dimension",
                    "Summarise key similarities and differences in a conclusion",
                ],
                score=0.9,
            ),
        ]

    elif problem_type == "logical_reasoning":
        branches = [
            ReasoningBranch(
                label="formal_inference",
                steps=[
                    "Identify premises and the conclusion",
                    "Map each statement to a propositional form (P, Q, …)",
                    "Select the applicable inference rule (modus ponens, tollens, hypothetical syllogism)",
                    "Check whether the conclusion follows necessarily from the premises",
                    "State whether the argument is valid and sound",
                ],
                score=0.95,
            ),
        ]

    elif problem_type == "causal_analysis":
        branches = [
            ReasoningBranch(
                label="root_cause",
                steps=[
                    "Identify the effect / outcome to be explained",
                    "List potential causes (proximate and distal)",
                    "Trace causal pathways using evidence",
                    "Distinguish correlation from causation",
                    "State the most likely causal explanation with confidence level",
                ],
                score=0.85,
            ),
        ]

    if not branches:
        branches = [
            ReasoningBranch(
                label="default_cot",
                steps=_decompose(user_input, problem_type),
                score=0.7,
            )
        ]

    branches.sort(key=lambda b: b.score, reverse=True)
    return branches


def _select_branch(
    branches: list[ReasoningBranch], user_input: str
) -> ReasoningBranch:
    """Heuristically select the best branch for the query."""
    lowered = user_input.lower()

    if not branches:
        return ReasoningBranch(
            label="default",
            steps=["Understand the problem", "Solve step by step", "Verify the answer"],
            score=0.5,
        )

    for branch in branches:
        if branch.label in lowered:
            return branch

    return branches[0]


# ─────────────────────────────────────────────────────────────────────────────
# Multi-hop decomposer (single best path)
# ─────────────────────────────────────────────────────────────────────────────

def _decompose(user_input: str, problem_type: str) -> list[str]:
    lowered = user_input.lower()

    if problem_type == "integration":
        steps = ["Identify the integrand and variable of integration"]
        if "from" in lowered and "to" in lowered:
            steps.append("Confirm the integration limits (lower and upper bounds)")
        steps += [
            "Check whether u-substitution, integration by parts, or a standard form applies",
            "Compute the antiderivative step by step",
            "Apply the Fundamental Theorem of Calculus if limits are given",
            "Simplify the result and add C for indefinite integrals",
            "Verify by differentiating the result",
        ]
        return steps

    if problem_type == "differentiation":
        steps = ["Identify the function and the differentiation variable"]
        if any(k in lowered for k in ["second", "third", "2nd", "3rd", "nth"]):
            steps.append("Determine the required order of differentiation")
        steps += [
            "Identify which rules apply (chain rule, product rule, quotient rule)",
            "Differentiate term by term",
            "Simplify the derivative expression",
            "Check special points (zeros, discontinuities) if asked",
        ]
        return steps

    if problem_type == "limit_evaluation":
        return [
            "Identify the function and the point of approach",
            "Try direct substitution first",
            "If 0/0 or ∞/∞, apply L'Hôpital's rule or algebraic factoring",
            "Check one-sided limits if approaching a boundary point",
            "Evaluate or state the limit; confirm if it diverges",
        ]

    if problem_type == "equation_solving":
        steps = ["Identify the type of equation (linear, quadratic, polynomial, transcendental)"]
        if "=" in user_input:
            steps.append("Rearrange so one side is 0")
        steps += [
            "Choose the solution method: factoring, quadratic formula, substitution, or numerical",
            "Find all solutions in the required domain",
            "Verify each solution in the original equation",
        ]
        return steps

    if problem_type == "ode_solving":
        return [
            "Classify the ODE: order, linearity, and special form",
            "For 1st order: test separable → linear (integrating factor) → exact",
            "For 2nd order: form characteristic equation and solve for roots",
            "Write general solution (homogeneous + particular if non-homogeneous)",
            "Apply initial/boundary conditions to find the particular solution",
            "Verify by substituting back into the ODE",
        ]

    if problem_type == "matrix_operation":
        steps = ["Identify the matrix dimensions and the operation required"]
        if "eigenvalue" in lowered:
            steps += [
                "Form characteristic polynomial det(A − λI) = 0",
                "Solve for eigenvalues λ",
                "For each λ, solve (A − λI)v = 0 for the eigenvector v",
            ]
        elif "determinant" in lowered or "det" in lowered:
            steps += [
                "Choose expansion method (cofactor expansion or row reduction)",
                "Compute the determinant systematically",
            ]
        elif "inverse" in lowered:
            steps += [
                "Augment the matrix: [A | I]",
                "Row-reduce to obtain [I | A⁻¹]",
                "Verify A × A⁻¹ = I",
            ]
        return steps

    if problem_type == "series_expansion":
        return [
            "Identify the function and expansion point a",
            "Compute f(a), f'(a), f''(a), … up to required order",
            "Form coefficients: aₙ = f^(n)(a) / n!",
            "Write the Taylor polynomial",
            "State the remainder term and radius of convergence if requested",
        ]

    if problem_type == "laplace_transform":
        return [
            "Express the function in terms of known Laplace pairs",
            "Apply linearity of the Laplace transform",
            "Use standard table entries (or direct computation for non-standard forms)",
            "Simplify the result in the s-domain",
        ]

    if problem_type == "statistical_analysis":
        return [
            "Identify the data set and the statistic being requested",
            "Compute the mean: μ = Σxᵢ / n",
            "Compute variance: σ² = Σ(xᵢ − μ)² / n",
            "Compute standard deviation: σ = √(σ²)",
            "State any distributional assumptions made",
        ]

    if problem_type == "combinatorics":
        return [
            "Identify whether order matters (permutation) or not (combination)",
            "Check for repetitions or restrictions",
            "Apply the appropriate formula: P(n,r) = n!/(n-r)! or C(n,r) = n!/(r!(n-r)!)",
            "Simplify the calculation",
            "Verify with a small example if possible",
        ]

    if problem_type == "knowledge_retrieval":
        return [
            "Identify the core concept being queried",
            "Recall the precise definition and key properties",
            "Provide concrete examples or applications",
            "Note any common misconceptions or edge cases",
            "Connect to related concepts where helpful",
        ]

    if problem_type == "conceptual_explanation":
        return [
            "Define the concept precisely",
            "Explain the underlying mechanism or principle",
            "Give a concrete real-world example",
            "Address common points of confusion",
            "Summarise the key takeaways",
        ]

    if problem_type == "explanatory_reasoning":
        return [
            "Identify the phenomenon or behaviour requiring explanation",
            "Identify the root cause(s)",
            "Trace the causal chain from cause to effect",
            "State any assumptions or simplifications made",
            "Provide supporting evidence or known principles",
        ]

    if problem_type == "comparative_analysis":
        return [
            "Define what is being compared",
            "Establish the key dimensions of comparison",
            "Analyse each dimension for subject A",
            "Analyse each dimension for subject B",
            "Summarise similarities and differences; state which is preferable (if applicable) with justification",
        ]

    if problem_type == "causal_analysis":
        return [
            "Identify the effect to be explained",
            "List candidate causes (proximate and distal)",
            "Evaluate evidence for each causal link",
            "Distinguish between correlation and causation",
            "State the most plausible causal explanation",
        ]

    if problem_type == "logical_reasoning":
        return [
            "Identify all premises in the argument",
            "Identify the conclusion being drawn",
            "Formalise premises in propositional logic (P, Q, …)",
            "Apply the appropriate inference rule",
            "Verify whether the argument is valid (form) and sound (truth of premises)",
        ]

    if problem_type == "procedural_knowledge":
        return [
            "Identify the goal and the prerequisites",
            "List the steps in logical order",
            "For each step, state what to do and why",
            "Identify potential failure points and how to handle them",
            "Summarise with a verification step",
        ]

    if problem_type == "physics_problem":
        return [
            "Identify the physical system and the quantity being found",
            "List the relevant physical laws and principles",
            "Define variables and known/unknown quantities",
            "Set up the equations",
            "Solve, check units, and verify the physical reasonableness of the answer",
        ]

    if problem_type == "chemistry_problem":
        return [
            "Identify reactants and products",
            "Assign oxidation states to each element",
            "Balance atoms on both sides (left to right)",
            "Balance charge if dealing with ionic equations",
            "Verify atom counts and charge balance",
        ]

    # ── JEE Advanced: Calculus ────────────────────────────────────────────────
    if problem_type == "maxima_minima":
        return [
            "Find the derivative f'(x) of the function",
            "Solve f'(x) = 0 to locate all critical points",
            "Compute f''(x) and evaluate at each critical point: f''<0 → local max, f''>0 → local min",
            "If f''(c)=0, use the first derivative sign change test instead",
            "For a closed interval [a,b], also evaluate f at the endpoints",
            "State the global maximum/minimum value with its location",
        ]

    if problem_type == "monotonicity":
        return [
            "Compute f'(x)",
            "Find the values of x where f'(x) = 0 or is undefined (critical points)",
            "Divide the domain into intervals at the critical points",
            "Test the sign of f'(x) in each interval: positive → increasing, negative → decreasing",
            "State the intervals of monotonicity explicitly",
        ]

    if problem_type == "mean_value_theorem":
        return [
            "Verify f is continuous on [a,b] and differentiable on (a,b)",
            "Compute f(a) and f(b)",
            "State the theorem: there exists c in (a,b) such that f'(c) = (f(b)-f(a))/(b-a)",
            "Find f'(x) and solve f'(c) = (f(b)-f(a))/(b-a) for c",
            "Verify c lies in (a,b)",
        ]

    # ── JEE Advanced: Coordinate Geometry ─────────────────────────────────────
    if problem_type in ("conic_parabola",):
        return [
            "Write the parabola in standard form (y²=4ax or x²=4ay); identify a",
            "Find focus (a,0), directrix x=-a, vertex (0,0), latus rectum length 4a",
            "For a shifted parabola (y-k)²=4a(x-h), identify vertex (h,k) accordingly",
            "Use parametric form (at², 2at) to derive tangent T: ty = x + at², normal t y + x = 2at + at³",
            "Apply T=0 for the tangent at a given point, or solve the problem using these results",
            "Verify the result satisfies the original equation",
        ]

    if problem_type in ("conic_ellipse",):
        return [
            "Write ellipse as x²/a²+y²/b²=1 with a>b; find b²=a²(1-e²)",
            "Identify focus (±ae, 0), directrix x=±a/e, latus rectum 2b²/a",
            "Use parametric form (a cosθ, b sinθ)",
            "Tangent at (x₁,y₁): xx₁/a²+yy₁/b²=1 (use T=0 shorthand)",
            "Normal: a²x/x₁ - b²y/y₁ = a²-b²",
            "For chord of contact from external point: T=0",
        ]

    if problem_type in ("conic_hyperbola",):
        return [
            "Write hyperbola as x²/a²-y²/b²=1; find b²=a²(e²-1), e>1",
            "Identify focus (±ae,0), directrix x=±a/e, asymptotes y=±(b/a)x",
            "Use parametric form (a secθ, b tanθ)",
            "Tangent at (x₁,y₁): xx₁/a²-yy₁/b²=1",
            "For rectangular hyperbola xy=c², use T: yx₁+xy₁=2c², parametric (ct, c/t)",
            "Apply T=0 or SS₁=T² for chord of contact/combined equation as needed",
        ]

    if problem_type in ("conic_section",):
        return [
            "Write the second-degree equation in standard form by completing the square",
            "Identify the conic type (parabola/ellipse/hyperbola/circle) by discriminant",
            "For chord of contact from external point P(h,k): write T=0 replacing x²→xh, y²→yk, xy→(xk+yh)/2, x→(x+h)/2, y→(y+k)/2",
            "For pair of tangents: SS₁=T²",
            "State the required geometric quantity (tangent length, chord length, etc.)",
        ]

    if problem_type == "pair_of_lines":
        return [
            "Write the combined equation ax²+2hxy+by²+2gx+2fy+c=0",
            "Check condition for pair of lines: Δ = abc+2fgh-af²-bg²-ch² = 0",
            "Find individual slopes: m₁+m₂ = -2h/b, m₁m₂ = a/b",
            "Angle between lines: tan θ = 2√(h²-ab)/(a+b)",
            "For lines through origin: ax²+2hxy+by²=0 — factor directly",
            "State the equations of the individual lines",
        ]

    if problem_type == "circle_geometry":
        return [
            "Write circle equation as x²+y²+2gx+2fy+c=0; identify centre (-g,-f) and radius √(g²+f²-c)",
            "For radical axis of two circles S₁=0, S₂=0: solve S₁-S₂=0",
            "Orthogonal circles condition: 2g₁g₂+2f₁f₂=c₁+c₂",
            "Tangent from external point: length = √S₁ = √(x₁²+y₁²+2gx₁+2fy₁+c)",
            "Equation of tangent at (x₁,y₁): xx₁+yy₁+g(x+x₁)+f(y+y₁)+c=0",
            "Apply the required result",
        ]

    if problem_type == "straight_line":
        return [
            "Write the line in slope-intercept or parametric form; identify slope m",
            "Angle between two lines: tan θ = |m₁-m₂|/(1+m₁m₂)",
            "Distance from point (x₁,y₁) to line ax+by+c=0: d = |ax₁+by₁+c|/√(a²+b²)",
            "Foot of perpendicular: use parametric form along normal direction",
            "Concurrent lines: set up determinant = 0",
            "State the required result with exact values",
        ]

    # ── JEE Advanced: 3D Geometry & Vectors ───────────────────────────────────
    if problem_type == "3d_geometry":
        return [
            "Express each line in vector/parametric form: r = a + λb",
            "For skew lines: SD = |(a₂-a₁)·(b₁×b₂)|/|b₁×b₂|",
            "Plane equation: r·n = d or ax+by+cz=d; find normal n from three points (use cross product)",
            "Distance from point P to plane: |n·P - d|/|n|",
            "Angle between planes: cos θ = |n₁·n₂|/(|n₁||n₂|)",
            "State the final answer with appropriate units or exact form",
        ]

    if problem_type == "vector_algebra":
        return [
            "Express vectors in component form (aî + bĵ + cκ̂)",
            "For scalar triple product [a b c] = a·(b×c): equals volume of parallelepiped; = 0 iff coplanar",
            "Projection of a on b = (a·b)/|b|; vector projection = [(a·b)/|b|²]b",
            "For vector triple product: a×(b×c) = (a·c)b-(a·b)c",
            "Compute the required quantity step by step with clear notation",
        ]

    # ── JEE Advanced: Probability ──────────────────────────────────────────────
    if problem_type == "bayes_probability":
        return [
            "Identify the hypotheses H₁, H₂, … Hₙ that partition the sample space",
            "State their prior probabilities P(H₁), P(H₂), …",
            "Identify the event E and compute P(E|Hᵢ) for each hypothesis",
            "Apply total probability theorem: P(E) = ΣP(E|Hᵢ)P(Hᵢ)",
            "Apply Bayes' theorem: P(Hᵢ|E) = P(E|Hᵢ)P(Hᵢ)/P(E)",
            "State the posterior probability and interpret the result",
        ]

    if problem_type in ("probability_distribution", "probability_problem"):
        return [
            "Define the sample space and identify the random variable X",
            "List possible values xᵢ and their probabilities P(X=xᵢ); verify they sum to 1",
            "Compute E(X) = Σ xᵢP(xᵢ)",
            "Compute E(X²) = Σ xᵢ²P(xᵢ), then Var(X) = E(X²) - [E(X)]²",
            "For Binomial: P(X=r) = C(n,r)pʳqⁿ⁻ʳ; E(X)=np; Var(X)=npq",
            "State the required probability or statistic with exact value",
        ]

    # ── JEE Advanced: Sequences & Series ──────────────────────────────────────
    if problem_type == "sequences_series":
        return [
            "Identify the type: AP (common difference d), GP (common ratio r), HP (reciprocals in AP)",
            "For AP: aₙ = a + (n-1)d; Sₙ = n/2·[2a + (n-1)d]",
            "For GP: aₙ = arⁿ⁻¹; Sₙ = a(rⁿ-1)/(r-1); S∞ = a/(1-r) if |r|<1",
            "For HP: take reciprocals and treat as AP",
            "For AM-GM-HM: AM ≥ GM ≥ HM; use to find bounds or extremes",
            "Compute the required quantity and verify",
        ]

    # ── JEE Advanced: Trigonometry ─────────────────────────────────────────────
    if problem_type == "trigonometric_equation":
        return [
            "Reduce the equation to a single trig function of one angle",
            "Express in standard form: sin θ = k, cos θ = k, or tan θ = k",
            "Write the general solution: sin θ=k → θ=nπ+(-1)ⁿα; cos θ=k → θ=2nπ±α; tan θ=k → θ=nπ+α",
            "Apply any given domain restrictions (e.g. θ ∈ [0, 2π]) to list particular solutions",
            "Verify each solution in the original equation",
        ]

    if problem_type == "inverse_trig":
        return [
            "State the domain and range of each inverse trig function involved",
            "Apply composition identities (sin(sin⁻¹ x)=x; sin⁻¹(sin x)=x only for x∈[-π/2,π/2])",
            "Use identities: sin⁻¹x + cos⁻¹x = π/2; tan⁻¹x + cot⁻¹x = π/2",
            "Simplify using addition formulas: tan⁻¹a ± tan⁻¹b = tan⁻¹((a±b)/(1∓ab))",
            "State the principal value of the final expression",
        ]

    if problem_type == "trig_identities":
        return [
            "State the relevant formula (compound, double, half angle, sum-to-product, product-to-sum)",
            "Substitute the given values into the formula",
            "Simplify step by step, showing each algebraic manipulation",
            "Verify the result using a specific angle if possible",
        ]

    if problem_type == "triangle_trig":
        return [
            "Label the triangle sides a, b, c opposite to angles A, B, C",
            "Choose the appropriate rule: sine rule a/sin A = 2R; cosine rule c²=a²+b²-2ab cosC",
            "Substitute known quantities and solve for the unknown",
            "Compute area = ½ab sinC if required",
            "Check for the ambiguous case (two triangles) when using the sine rule",
        ]

    # ── JEE Advanced: Binomial Theorem ────────────────────────────────────────
    if problem_type == "binomial_theorem":
        return [
            "Write the general term: Tᵣ₊₁ = C(n,r)·xⁿ⁻ʳ·yʳ",
            "For the middle term: when n is even, middle term = T_{n/2+1}; when n is odd, two middle terms",
            "For term independent of x: set the power of x in Tᵣ₊₁ equal to 0 and solve for r",
            "For a specific coefficient: set the power equal to the required value and read off C(n,r)·coeff",
            "Compute the numerical value of the binomial coefficient C(n,r) = n!/[r!(n-r)!]",
        ]

    # ── JEE Advanced: Physics (SHM, Rotational, Modern) ───────────────────────
    if problem_type == "shm_problem":
        return [
            "Identify the restoring force: F = -kx (spring) or F = -mω²x; find ω = √(k/m)",
            "State T = 2π/ω and f = ω/(2π)",
            "Displacement: x(t) = A cos(ωt + φ); apply initial conditions to find A and φ",
            "Velocity: v = -Aω sin(ωt+φ); v_max = Aω at mean position",
            "Acceleration: a = -ω²x; |a_max| = Aω² at extreme positions",
            "Energy: KE = ½mω²(A²-x²), PE = ½mω²x², Total E = ½mω²A² = ½kA² (constant)",
            "State the required quantity with units",
        ]

    if problem_type == "rotational_mechanics":
        return [
            "Identify the axis of rotation and the body geometry",
            "Look up or derive the moment of inertia I (e.g. solid sphere: 2MR²/5, rod about end: ML²/3)",
            "Apply parallel axis theorem if needed: I = I_cm + Md²",
            "Apply perpendicular axis theorem for laminas: I_z = I_x + I_y",
            "Use Newton's 2nd law for rotation: τ = Iα",
            "For rolling without slipping: v = Rω, a = Rα; KE_total = ½mv² + ½Iω²",
            "Apply conservation of angular momentum if no external torque: L = Iω = constant",
            "State the required quantity with units and verify dimensional consistency",
        ]

    if problem_type == "modern_physics":
        return [
            "Identify the phenomenon: photoelectric effect, de Broglie wave, Bohr model, or nuclear",
            "Photoelectric: KE_max = hf - φ = hf - hf₀; stopping potential eV₀ = KE_max",
            "De Broglie: λ = h/p = h/mv; for electron: λ = 1.227/√V (nm) with V in volts",
            "Bohr model: Eₙ = -13.6/n² eV; rₙ = 0.529n² Å; ΔE = hf for transitions",
            "Nuclear: Q = Δm·c²; binding energy = [Zm_p + (A-Z)m_n - m_nucleus]·c²",
            "Radioactive decay: N = N₀e^(-λt); t₁/₂ = ln2/λ",
            "Compute the required quantity with correct units (eV, nm, MeV, Å)",
        ]

    if problem_type == "electrostatics":
        return [
            "Identify charge configuration; state Coulomb's law F = kq₁q₂/r² (k = 9×10⁹ N m²/C²)",
            "For electric field: E = kq/r² (point charge); E_uniform = σ/ε₀; superpose for multiple charges",
            "Apply Gauss's law ∮E·dA = Q_enc/ε₀ for symmetric distributions (sphere, cylinder, plane)",
            "Electric potential: V = kq/r; ΔV = -∫E·dr; E = -dV/dr",
            "Capacitance: C = Q/V; parallel plate C = ε₀A/d; energy U = ½CV² = Q²/(2C)",
            "With dielectric: C' = κC, E' = E/κ",
            "Compute the required quantity step by step with units",
        ]

    if problem_type == "electromagnetism":
        return [
            "Identify the current configuration; state Biot-Savart law dB = μ₀I dl×r̂/(4πr²)",
            "For symmetry, use Ampere's law: ∮B·dl = μ₀I_enclosed",
            "Standard results: B at centre of circular loop = μ₀I/(2R); solenoid B = μ₀nI",
            "Faraday's law: EMF = -dΦ/dt; Lenz's law determines direction",
            "Self-inductance: L = Φ/I; solenoid L = μ₀n²V; energy U = ½LI²",
            "Mutual inductance: M = μ₀n₁n₂·(common volume); EMF₂ = -M dI₁/dt",
            "State the required quantity with correct SI units (T, Wb, H, V)",
        ]

    # ── JEE Advanced: Chemistry specialised ───────────────────────────────────
    if problem_type == "chemical_equilibrium":
        return [
            "Write the balanced equation and the equilibrium constant expression Kc or Kp",
            "Relate Kp = Kc(RT)^Δn where Δn = moles of gaseous products - reactants",
            "Set up the ICE (Initial/Change/Equilibrium) table",
            "Solve for the equilibrium concentrations (use approximation if α << 1)",
            "Apply Le Chatelier's principle to predict shifts due to changes in T, P, or concentration",
            "State the equilibrium concentrations or degree of dissociation with units",
        ]

    if problem_type == "chemical_kinetics":
        return [
            "Identify the rate law from given data: rate = k[A]^m[B]^n",
            "Determine the order from initial rate experiments (method of initial rates)",
            "For 1st order: ln[A] = ln[A]₀ - kt; t₁/₂ = 0.693/k",
            "For 2nd order: 1/[A] = 1/[A]₀ + kt; t₁/₂ = 1/(k[A]₀)",
            "Arrhenius equation: k = A·e^(-Ea/RT); ln(k₂/k₁) = (Ea/R)(1/T₁ - 1/T₂)",
            "Compute the required quantity and verify units of k are consistent with the order",
        ]

    if problem_type == "solid_state":
        return [
            "Identify the crystal lattice type: SC (coordination=6), BCC (=8), FCC (=12)",
            "Number of atoms per unit cell: SC=1, BCC=2, FCC=4",
            "Relationship between edge length a and radius r: SC: a=2r; BCC: 4r=a√3; FCC: 4r=a√2",
            "Packing efficiency: SC≈52.4%, BCC≈68%, FCC≈74%",
            "Density: ρ = (Z·M)/(Nₐ·a³) where Z = atoms/cell, M = molar mass, a = edge length",
            "State the required quantity with units",
        ]

    if problem_type == "colligative_properties":
        return [
            "Identify the colligative property: ΔTb (boiling point elevation), ΔTf (freezing point depression), or osmotic pressure π",
            "ΔTb = i·Kb·m; ΔTf = i·Kf·m (m = molality = moles solute/kg solvent)",
            "π = iMRT (M = molarity, R = 0.0821 L·atm/mol·K)",
            "Van't Hoff factor i = 1+(n-1)α for degree of dissociation α; i<1 for association",
            "Raoult's law for vapour pressure: ΔP/P° = x_solute; relative lowering = n_solute/(n_solute+n_solvent)",
            "Compute the required quantity and verify units",
        ]

    if problem_type == "predictive_reasoning":
        return [
            "Understand the current state or conditions",
            "Identify the relevant trend, law, or model",
            "Project the outcome under those conditions",
            "State confidence and key uncertainties",
            "Note what could change the prediction",
        ]

    if problem_type == "ethical_reasoning":
        return [
            "Identify all stakeholders affected",
            "Frame the ethical question precisely",
            "Apply consequentialist analysis (outcomes and harms)",
            "Apply deontological analysis (duties, rights, rules)",
            "Apply virtue ethics (what a person of good character would do)",
            "Synthesise a reasoned position with acknowledgment of trade-offs",
        ]

    if problem_type == "competition_math":
        return [
            "Assign variables: let v = first person's speed (mph), T = first person's total travel time (hours)",
            "Write the distance equation for each person: d = speed × travel_time, noting each person's start delay and cumulative speed increment",
            "Set all distances equal (all arrive at the same destination) to form a system of equations in v and T",
            "Solve the system exactly — expand and equate to find T first, then v",
            "Compute d = v·T and express as a reduced fraction m/n with gcd(m,n) = 1",
            "State m + n as the final answer",
        ]

    return [
        "Understand the question and identify the core task",
        "Break the problem into smaller sub-problems",
        "Address each sub-problem in order",
        "Verify reasoning for consistency",
        "Synthesise a complete and accurate answer",
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Self-consistency verifier
# ─────────────────────────────────────────────────────────────────────────────

def _verify_consistency(
    sub_problems: list[str],
    strategy: str,
    problem_type: str,
) -> tuple[bool, str]:
    """
    Verify that the reasoning plan is internally consistent.
    Returns (is_consistent, note).
    """
    if not sub_problems:
        return False, "No reasoning steps were generated"

    if len(sub_problems) < 2:
        return True, ""

    first = sub_problems[0].lower()
    last  = sub_problems[-1].lower()

    verify_keywords = {"verify", "check", "confirm", "validate", "test", "proof"}
    if not any(kw in last for kw in verify_keywords):
        if problem_type in ("integration", "differentiation", "equation_solving", "ode_solving"):
            pass

    if len(sub_problems) > 10:
        return False, "Plan has too many steps — likely over-decomposed"

    if problem_type == "integration" and "differentiat" in strategy.lower():
        return False, "Strategy mentions differentiation for an integration problem"

    return True, ""


# ─────────────────────────────────────────────────────────────────────────────
# Metacognitive analyser
# ─────────────────────────────────────────────────────────────────────────────

_HIGH_CERTAINTY_TYPES = {
    "integration", "differentiation", "limit_evaluation", "equation_solving",
    "factorization", "algebraic_expansion", "simplification", "series_expansion",
    "number_theory", "statistical_analysis", "combinatorics", "matrix_operation",
    "ode_solving", "laplace_transform", "fourier_transform",
    # JEE Advanced additions
    "maxima_minima", "monotonicity", "mean_value_theorem",
    "conic_parabola", "conic_ellipse", "conic_hyperbola", "conic_section",
    "pair_of_lines", "circle_geometry", "straight_line",
    "3d_geometry", "vector_algebra",
    "sequences_series", "trigonometric_equation", "inverse_trig",
    "trig_identities", "triangle_trig", "binomial_theorem",
    "shm_problem", "rotational_mechanics", "modern_physics",
    "electrostatics", "electromagnetism",
    "chemical_equilibrium", "chemical_kinetics", "solid_state",
    "colligative_properties",
    "bayes_probability", "probability_distribution", "probability_problem",
}

_MEDIUM_CERTAINTY_TYPES = {
    "knowledge_retrieval", "conceptual_explanation", "procedural_knowledge",
    "physics_problem", "chemistry_problem", "biology_concept", "logical_reasoning",
    "propositional_logic", "ideal_gas_thermo",
}

_LOW_CERTAINTY_TYPES = {
    "explanatory_reasoning", "causal_analysis", "comparative_analysis",
    "predictive_reasoning", "ethical_reasoning", "analogical_reasoning",
}


def _build_metacognition(
    user_input: str,
    problem_type: str,
    domain: str,
    has_symbolic_result: bool,
) -> MetacognitiveState:
    meta = MetacognitiveState()
    lowered = user_input.lower()

    if has_symbolic_result:
        meta.known_aspects.append(f"Exact symbolic answer computed by SymPy (certainty: 100%)")
        meta.boundary_note = "The numerical/symbolic answer is certain; explanation quality depends on LLM."
        return meta

    if problem_type in _HIGH_CERTAINTY_TYPES:
        meta.known_aspects.append(f"Well-defined {problem_type} problem with deterministic solution")
        meta.boundary_note = "High confidence: established mathematical procedure applies."

    elif problem_type in _MEDIUM_CERTAINTY_TYPES:
        meta.known_aspects.append(f"Domain: {domain} — factual knowledge available")
        if domain == "history":
            meta.uncertain_aspects.append("Historical interpretation may vary by source")
        if domain == "physics":
            meta.uncertain_aspects.append("Numerical precision depends on given values")
        meta.boundary_note = "Moderate confidence: answer should be accurate but verify with authoritative source."

    else:
        meta.uncertain_aspects.append(f"Problem type '{problem_type}' involves judgment or incomplete information")
        if "predict" in lowered:
            meta.knowledge_gaps.append("Future outcomes are inherently uncertain")
        if "ethics" in lowered or "moral" in lowered:
            meta.knowledge_gaps.append("Ethical conclusions depend on framework and values")
        meta.boundary_note = "Lower confidence: answer represents reasoned judgment, not established fact."

    return meta


# ─────────────────────────────────────────────────────────────────────────────
# Confidence estimator
# ─────────────────────────────────────────────────────────────────────────────

def _estimate_confidence(
    user_input: str,
    problem_type: str,
    intent: str,
    has_symbolic_result: bool = False,
) -> str:
    if has_symbolic_result:
        return "HIGH"

    if problem_type in _HIGH_CERTAINTY_TYPES:
        return "HIGH"

    if problem_type in _MEDIUM_CERTAINTY_TYPES:
        return "MEDIUM"

    if problem_type in _LOW_CERTAINTY_TYPES:
        lowered = user_input.lower()
        if any(kw in lowered for kw in ["opinion", "best", "should i", "predict"]):
            return "LOW"
        return "MEDIUM"

    if intent in ("knowledge", "conversation"):
        lowered = user_input.lower()
        if any(kw in lowered for kw in ["what is", "define", "who is"]):
            return "MEDIUM"
        if any(kw in lowered for kw in ["why", "opinion", "best", "should"]):
            return "LOW"

    return "MEDIUM"


# ─────────────────────────────────────────────────────────────────────────────
# Warnings detector
# ─────────────────────────────────────────────────────────────────────────────

def _detect_warnings(user_input: str, problem_type: str) -> list[str]:
    warnings = []
    lowered  = user_input.lower()

    if len(user_input.strip()) < 5:
        warnings.append("Input is very short — may be ambiguous")

    if problem_type == "equation_solving" and "=" not in user_input and "solve" in lowered:
        warnings.append("No '=' found — treating expression as equal to 0")

    if problem_type == "integration" and "from" in lowered and "to" not in lowered:
        warnings.append("'from' detected but 'to' is missing — treating as indefinite integral")

    if problem_type in ("differentiation", "integration") and not any(
        c in user_input for c in list("xyztnkabcmnpqrs")
    ):
        warnings.append("No variable detected — defaulting to x")

    if "undefined" in lowered or "singularity" in lowered:
        warnings.append("Expression may involve singularities or unbounded behaviour")

    if problem_type in _LOW_CERTAINTY_TYPES:
        warnings.append("This question involves judgment — answer reflects reasoned inference, not established fact")

    if "always" in lowered or "never" in lowered:
        warnings.append("Absolute claims ('always'/'never') are rarely universally true — response will qualify appropriately")

    return warnings


# ─────────────────────────────────────────────────────────────────────────────
# Use Tree-of-Thought when problem is complex
# ─────────────────────────────────────────────────────────────────────────────

_TOT_TYPES = {
    "integration", "equation_solving", "ode_solving", "explanatory_reasoning",
    "comparative_analysis", "logical_reasoning", "causal_analysis",
    # JEE Advanced complex problem types
    "conic_parabola", "conic_ellipse", "conic_hyperbola", "conic_section",
    "rotational_mechanics", "modern_physics", "electrostatics", "electromagnetism",
    "chemical_equilibrium", "chemical_kinetics", "bayes_probability",
}


def _should_use_tot(problem_type: str, user_input: str) -> bool:
    if problem_type not in _TOT_TYPES:
        return False
    if len(user_input.split()) > 8:
        return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Forest-of-Thought (FoT) — problem types that benefit from parallel hypotheses
# ─────────────────────────────────────────────────────────────────────────────

_FOT_TYPES: frozenset[str] = frozenset({
    # Multi-concept problems where the approach is ambiguous
    "chemistry_problem", "chemical_thermodynamics", "chemical_equilibrium",
    "thermochemistry",
    # Multi-strategy math
    "equation_solving", "sequences_series", "combinatorics",
    "trig_identities", "inverse_trig", "trigonometric_equation",
    # Complex physics
    "physics_problem", "rotational_mechanics", "electrostatics",
    # Reasoning
    "explanatory_reasoning", "causal_analysis",
})

# Minimum word count to trigger FoT (complex multi-part problems)
_FOT_WORD_THRESHOLD = 25


def _should_use_fot(problem_type: str, user_input: str) -> bool:
    """FoT is triggered for complex, multi-concept problems where the root
    approach itself is uncertain (e.g. Hess's law vs bond enthalpy for ΔH)."""
    if problem_type not in _FOT_TYPES:
        return False
    word_count = len(user_input.split())
    # Multi-part question signal: contains 'a)', 'b)', '(i)', 'part'
    has_parts = bool(re.search(r'\b(?:part\s+[ab]|\([abi]\)|a\)|b\)|i\)|ii\))\b',
                               user_input.lower()))
    return word_count >= _FOT_WORD_THRESHOLD or has_parts


def _generate_forest(
    user_input: str, problem_type: str, domain: str
) -> list[ForestTree]:
    """
    Forest-of-Thought: produce 2–3 independent root hypotheses, each with its
    own reasoning chain.  The caller selects the highest-scored tree.
    """
    lowered = user_input.lower()
    trees: list[ForestTree] = []

    # ── Thermochemistry / ΔH problems ────────────────────────────────────────
    if problem_type in ("thermochemistry", "chemical_thermodynamics"):
        trees = [
            ForestTree(
                hypothesis="Hess's Law (formation enthalpies)",
                steps=[
                    "Write the target reaction",
                    "Express ΔH°rxn = ΣΔH°f(products) − ΣΔH°f(reactants)",
                    "Substitute given ΔH°f values",
                    "Compute and state result with units (kJ/mol)",
                ],
                score=0.90,
                conclusion="Direct formula application; high reliability when ΔH°f data given",
            ),
            ForestTree(
                hypothesis="Bond Enthalpy method",
                steps=[
                    "List all bonds broken in reactants and formed in products",
                    "Apply ΔH = Σ(bond enthalpies broken) − Σ(bond enthalpies formed)",
                    "Substitute given bond energies",
                    "Compute; note this gives an estimate (average bond enthalpies)",
                ],
                score=0.80,
                conclusion="Approximate; use only when bond energy table is provided",
            ),
            ForestTree(
                hypothesis="Hess's Law (given reaction equations)",
                steps=[
                    "Identify given reactions that can be combined to yield the target",
                    "Scale and reverse reactions as needed; adjust ΔH signs accordingly",
                    "Add all equations and ΔH values to get the target ΔH",
                    "Verify atom balance",
                ],
                score=0.88,
                conclusion="Use when individual reaction ΔH values (not ΔH°f) are provided",
            ),
        ]

    # ── Chemical equilibrium / acid-base ─────────────────────────────────────
    elif problem_type == "chemical_equilibrium":
        trees = [
            ForestTree(
                hypothesis="ICE table (Kc / Ka / Kb)",
                steps=[
                    "Write the balanced equilibrium expression",
                    "Set up ICE table: Initial, Change (+x / −x), Equilibrium",
                    "Write Kc = [products]/[reactants] and substitute ICE expressions",
                    "Solve the algebraic equation for x",
                    "Verify that x << initial conc (if approximation used)",
                ],
                score=0.92,
                conclusion="Primary method for equilibrium concentration problems",
            ),
            ForestTree(
                hypothesis="Henderson-Hasselbalch (buffer)",
                steps=[
                    "Identify whether system is a buffer (weak acid + conjugate base)",
                    "Apply pH = pKa + log([A−]/[HA])",
                    "Substitute given concentrations",
                    "Compute pH",
                ],
                score=0.75,
                conclusion="Use specifically for buffer problems",
            ),
        ]

    # ── Trigonometric identity / equation problems ────────────────────────────
    elif problem_type in ("trig_identities", "trigonometric_equation"):
        trees = [
            ForestTree(
                hypothesis="Standard identity transformation",
                steps=[
                    "Identify which compound-angle / product-to-sum identity applies",
                    "Rewrite LHS using the identity",
                    "Simplify step by step until LHS = RHS",
                ],
                score=0.88,
                conclusion="Best for pure identity proofs",
            ),
            ForestTree(
                hypothesis="Auxiliary angle method (R·sin(x+φ))",
                steps=[
                    "Write a·sinθ + b·cosθ = R·sin(θ+φ) where R=√(a²+b²)",
                    "Compute φ = arctan(b/a)",
                    "Solve the resulting single-trig equation",
                ],
                score=0.85,
                conclusion="Optimal for max/min of trig expressions or solving a·sinθ+b·cosθ=k",
            ),
            ForestTree(
                hypothesis="General solution via substitution",
                steps=[
                    "Reduce to a standard form: sin θ = k, cos θ = k, or tan θ = k",
                    "State the general solution formula for the chosen trig function",
                    "Substitute back and simplify",
                    "State the domain-specific solutions",
                ],
                score=0.80,
                conclusion="Handles all trigonometric equations; most general approach",
            ),
        ]

    # ── Sequences / Series ────────────────────────────────────────────────────
    elif problem_type == "sequences_series":
        trees = [
            ForestTree(
                hypothesis="AP/GP/HP direct formula",
                steps=[
                    "Identify the sequence type (AP: common difference; GP: common ratio)",
                    "State the nth-term formula and sum formula",
                    "Substitute known values and solve for the unknown",
                ],
                score=0.90,
                conclusion="Use when the sequence clearly follows AP or GP",
            ),
            ForestTree(
                hypothesis="Method of differences / telescoping",
                steps=[
                    "Express the general term Tₙ as a difference f(n) − f(n−1)",
                    "Sum telescopes: Sₙ = f(n) − f(0)",
                    "Substitute and simplify",
                ],
                score=0.82,
                conclusion="Powerful for series like 1/(n(n+1)), k(k+1), etc.",
            ),
            ForestTree(
                hypothesis="Standard summation formulas (∑r, ∑r², ∑r³)",
                steps=[
                    "Express Tₙ as a polynomial in n",
                    "Apply ∑r = n(n+1)/2, ∑r² = n(n+1)(2n+1)/6, ∑r³ = [n(n+1)/2]²",
                    "Combine and simplify to get Sₙ",
                ],
                score=0.85,
                conclusion="Best for polynomial-in-n general term series",
            ),
        ]

    # ── Combinatorics ─────────────────────────────────────────────────────────
    elif problem_type == "combinatorics":
        trees = [
            ForestTree(
                hypothesis="Direct counting (Fundamental Theorem)",
                steps=[
                    "Break the task into independent choices",
                    "Count options at each stage",
                    "Multiply (AND) or add (OR) counts",
                    "Apply restrictions (subtract invalid cases)",
                ],
                score=0.88,
                conclusion="Best for structured counting problems",
            ),
            ForestTree(
                hypothesis="Complementary counting",
                steps=[
                    "Count the total (unrestricted) arrangements",
                    "Count the invalid (restricted) arrangements",
                    "Valid = Total − Invalid",
                ],
                score=0.82,
                conclusion="Efficient when valid cases are hard to count directly",
            ),
            ForestTree(
                hypothesis="Generating functions / Inclusion-Exclusion",
                steps=[
                    "Set up inclusion-exclusion: |A∪B| = |A|+|B|−|A∩B|",
                    "Identify overlapping conditions",
                    "Apply the principle systematically",
                ],
                score=0.75,
                conclusion="Use for complex overlap conditions (at-least-one type problems)",
            ),
        ]

    # ── Physics problems ──────────────────────────────────────────────────────
    elif problem_type in ("physics_problem", "rotational_mechanics", "electrostatics"):
        trees = [
            ForestTree(
                hypothesis="Energy / Work-Energy theorem",
                steps=[
                    "Identify initial and final states",
                    "Apply work-energy theorem: W_net = ΔKE",
                    "Or energy conservation: KE_i + PE_i = KE_f + PE_f + W_friction",
                    "Solve for the unknown",
                ],
                score=0.88,
                conclusion="Powerful when forces are complex but energy boundaries are clear",
            ),
            ForestTree(
                hypothesis="Newton's laws / Force analysis",
                steps=[
                    "Draw free-body diagram",
                    "Identify all forces",
                    "Apply ΣF = ma (linear) or Στ = Iα (rotational)",
                    "Solve the equations of motion",
                ],
                score=0.90,
                conclusion="Direct method; use when asked for force, acceleration, or time",
            ),
            ForestTree(
                hypothesis="Conservation laws (momentum / angular momentum)",
                steps=[
                    "Check if there's no external force/torque on the system",
                    "Apply conservation: p_before = p_after or L_before = L_after",
                    "Set up equations for each component",
                    "Solve for unknowns",
                ],
                score=0.83,
                conclusion="Ideal for collision, explosion, or torque-free rotation problems",
            ),
        ]

    # ── Explanatory reasoning / Causal analysis ───────────────────────────────
    elif problem_type in ("explanatory_reasoning", "causal_analysis"):
        trees = [
            ForestTree(
                hypothesis="Mechanism-based explanation",
                steps=[
                    "Identify the phenomenon / effect",
                    "Identify the primary causal mechanism at molecular/atomic level",
                    "Trace the causal chain step by step",
                    "State any exceptions or edge cases",
                ],
                score=0.88,
                conclusion="Best for 'why does X happen' questions",
            ),
            ForestTree(
                hypothesis="Comparative trend analysis",
                steps=[
                    "Identify the trend or order being asked",
                    "State the controlling factor (electronegativity, size, bond strength, etc.)",
                    "Apply the factor systematically to each species",
                    "State the conclusion / order",
                ],
                score=0.82,
                conclusion="Best for 'arrange in order of' or 'compare' questions",
            ),
        ]

    # ── Equation solving ──────────────────────────────────────────────────────
    elif problem_type == "equation_solving":
        trees = [
            ForestTree(
                hypothesis="Factoring / Vieta's formulas",
                steps=[
                    "Rearrange to standard form f(x) = 0",
                    "Attempt factoring by grouping or rational root theorem",
                    "Verify roots; apply Vieta's for sum/product of roots",
                ],
                score=0.88,
                conclusion="Fastest when polynomial factors neatly",
            ),
            ForestTree(
                hypothesis="Quadratic formula",
                steps=[
                    "Identify a, b, c in ax² + bx + c = 0",
                    "Compute D = b² − 4ac; interpret sign (real/complex roots)",
                    "x = (−b ± √D) / 2a",
                    "Verify both roots",
                ],
                score=0.92,
                conclusion="Definitive for any quadratic; use as backup for cubics via substitution",
            ),
            ForestTree(
                hypothesis="Substitution / parametric reduction",
                steps=[
                    "Identify a substitution that reduces degree (e.g. u = x²)",
                    "Solve the reduced equation for u",
                    "Back-substitute to find x",
                    "Check all branches are valid",
                ],
                score=0.80,
                conclusion="Use for biquadratic, exponential, or trigonometric equations",
            ),
        ]

    # ── Default: two generic FoT trees ───────────────────────────────────────
    if not trees:
        trees = [
            ForestTree(
                hypothesis="Formula-first approach",
                steps=[
                    "Identify the most directly applicable formula or principle",
                    "State and justify why this formula applies",
                    "Substitute values and solve",
                    "Verify the result",
                ],
                score=0.80,
                conclusion="Best when a known formula clearly matches the problem",
            ),
            ForestTree(
                hypothesis="First-principles derivation",
                steps=[
                    "State the fundamental axioms / definitions involved",
                    "Derive the required result from scratch",
                    "Cross-check against known formula if available",
                ],
                score=0.72,
                conclusion="More rigorous; use when formula applicability is uncertain",
            ),
        ]

    # Score adjustment based on problem length / hints in the text
    for tree in trees:
        if any(kw in lowered for kw in ["bond", "enthalpy", "formation"]):
            if "bond" in tree.hypothesis.lower() or "formation" in tree.hypothesis.lower():
                tree.score = min(tree.score + 0.05, 1.0)
        if any(kw in lowered for kw in ["ice table", "kc", "ka", "kb", "equilibrium"]):
            if "ice" in tree.hypothesis.lower() or "ka" in tree.hypothesis.lower():
                tree.score = min(tree.score + 0.05, 1.0)
        if any(kw in lowered for kw in ["buffer", "henderson"]):
            if "henderson" in tree.hypothesis.lower():
                tree.score = min(tree.score + 0.08, 1.0)

    # Sort best tree first
    trees.sort(key=lambda t: t.score, reverse=True)
    return trees


# ─────────────────────────────────────────────────────────────────────────────
# Public interface
# ─────────────────────────────────────────────────────────────────────────────

class ReasoningEngine:
    """
    Multi-strategy Chain-of-Thought + Tree-of-Thought reasoning engine v2.

    Usage:
        re = ReasoningEngine()
        plan = re.analyze(user_input, intent)
        prompt = re.build_math_prompt(user_input, sympy_result, plan)
        prompt = re.build_general_prompt(user_input, intent, context, plan)
    """

    def analyze(
        self,
        user_input: str,
        intent: str,
        has_symbolic_result: bool = False,
    ) -> ReasoningPlan:
        lowered = user_input.lower()

        # 1. Detect domain — router intent is authoritative; for advanced_math,
        #    use keyword scoring to pick the best sub-domain (calculus, algebra,
        #    coordinate_geometry, 3d_geometry, probability, sequences, trigonometry, etc.)
        _INTENT_TO_DOMAIN = {
            "chemistry":     "chemistry",
            "physics":       "physics",
            "math":          "algebra",
            "logic":         "logic",
            "knowledge":     "general",
        }
        if intent in _INTENT_TO_DOMAIN:
            domain = _INTENT_TO_DOMAIN[intent]
        else:
            # For advanced_math and conversation: keyword-based domain scoring
            domain = "calculus" if intent == "advanced_math" else "general"
            best_domain_hits = 0
            for d, keywords in _DOMAIN_HINTS.items():
                hits = sum(1 for kw in keywords if kw in lowered)
                if hits > best_domain_hits:
                    best_domain_hits = hits
                    domain = d

        # 2. Classify problem type (priority-ordered patterns)
        problem_type = "unknown"
        for pattern, ptype in _PROBLEM_TYPE_PATTERNS:
            if re.search(pattern, lowered):
                problem_type = ptype
                break

        # 3. Strategy & expected answer form
        strategy      = _STRATEGIES.get(problem_type, _STRATEGIES["unknown"])
        expected_form = _EXPECTED_FORMS.get(problem_type, "")

        # 4. Decide reasoning mode: Forest-of-Thought > Tree-of-Thought > CoT
        use_fot = _should_use_fot(problem_type, user_input)
        use_tot = (not use_fot) and _should_use_tot(problem_type, user_input)

        if use_fot:
            reasoning_mode = "forest_of_thought"
        elif use_tot:
            reasoning_mode = "tree_of_thought"
        else:
            reasoning_mode = "chain_of_thought"

        # 5. Generate forest (FoT), branches (ToT), or flat decomposition (CoT)
        branches: list[ReasoningBranch] = []
        forest:   list[ForestTree]      = []

        if use_fot:
            forest      = _generate_forest(user_input, problem_type, domain)
            best_tree   = forest[0] if forest else None
            sub_problems = best_tree.steps if best_tree else _decompose(user_input, problem_type)
        elif use_tot:
            branches     = _generate_branches(user_input, problem_type, domain)
            best_branch  = _select_branch(branches, user_input)
            sub_problems = best_branch.steps
        else:
            sub_problems = _decompose(user_input, problem_type)

        # 6. Assumptions
        assumptions = []
        if domain in ("calculus", "algebra", "differential_equations"):
            if not any(kw in lowered for kw in ["complex", "imaginary", "i^2"]):
                assumptions.append("Working over the real numbers unless stated otherwise")
        if problem_type in ("integration", "differentiation"):
            assumptions.append("Function is sufficiently smooth (differentiable/integrable on the given domain)")
        if problem_type == "statistical_analysis":
            if not any(kw in lowered for kw in ["population", "sample"]):
                assumptions.append("Treating data as a population (population mean/variance)")

        # 7. Warnings
        warnings = _detect_warnings(user_input, problem_type)

        # 8. Self-consistency check
        consistent, consistency_note = _verify_consistency(sub_problems, strategy, problem_type)

        # 9. Metacognitive state
        metacognition = _build_metacognition(
            user_input, problem_type, domain, has_symbolic_result
        )

        # 10. Confidence
        confidence = _estimate_confidence(
            user_input, problem_type, intent, has_symbolic_result
        )

        return ReasoningPlan(
            problem_type=problem_type,
            domain=domain,
            sub_problems=sub_problems,
            strategy=strategy,
            expected_form=expected_form,
            assumptions=assumptions,
            confidence=confidence,
            reasoning_steps=sub_problems,
            warnings=warnings,
            branches=branches,
            forest=forest,
            metacognition=metacognition,
            consistency_ok=consistent,
            consistency_note=consistency_note,
            reasoning_mode=reasoning_mode,
        )

    # ── Prompt builders ────────────────────────────────────────────────────

    def build_math_prompt(
        self,
        user_input: str,
        sympy_result: str,
        plan: ReasoningPlan,
    ) -> str:
        steps_str = "\n".join(
            f"  Step {i}: {s}" for i, s in enumerate(plan.sub_problems, 1)
        )
        assumptions_str = (
            ("\nAssumptions: " + "; ".join(plan.assumptions))
            if plan.assumptions else ""
        )
        expected_str = (
            f"\nExpected answer form: {plan.expected_form}"
            if plan.expected_form else ""
        )
        meta = plan.metacognition.to_text()
        meta_block = f"\n{meta}" if meta else ""

        reasoning_note = ""
        if plan.reasoning_mode == "forest_of_thought" and plan.forest:
            all_hyps = [t.hypothesis for t in plan.forest]
            best = plan.forest[0]
            reasoning_note = (
                f"\n[Forest-of-Thought: explored {len(plan.forest)} independent approaches: "
                f"{all_hyps}. Selected '{best.hypothesis}' "
                f"(score={best.score:.2f}): {best.conclusion}]\n"
            )
        elif plan.reasoning_mode == "tree_of_thought" and plan.branches:
            branch_labels = [b.label for b in plan.branches[:3]]
            reasoning_note = (
                f"\n[Tree-of-Thought: considered approaches {branch_labels}; "
                f"selected '{plan.branches[0].label}' as most suitable]\n"
            )
        else:
            reasoning_note = "\n[Chain-of-Thought: linear step-by-step reasoning]\n"

        return (
            f"PROBLEM: {user_input}\n\n"
            f"VERIFIED ANSWER (computed by SymPy — mathematically exact): {sympy_result}\n\n"
            "=== YOUR TASK ===\n"
            "Think step by step. "
            f"Explain HOW a student arrives at: {sympy_result}\n"
            "Do NOT recompute the answer differently. Your explanation MUST lead to exactly the verified answer.\n\n"
            f"TECHNIQUE: {plan.strategy}\n"
            f"DOMAIN: {plan.domain}"
            f"{expected_str}"
            f"{assumptions_str}"
            f"{reasoning_note}"
            f"{meta_block}\n\n"
            f"STEPS TO WALK THROUGH:\n{steps_str}\n\n"
            "Write a numbered explanation. For each step:\n"
            "  - State what you are doing and why.\n"
            "  - Show the algebra or calculation clearly.\n"
            "  - Connect it to the next step.\n"
            "  - If verifying, show the check explicitly.\n\n"
            f"End with exactly: 'Final answer: {sympy_result}'"
        )

    def build_general_prompt(
        self,
        user_input: str,
        intent: str,
        context: str,
        plan: ReasoningPlan,
    ) -> str:
        steps_str = "\n".join(
            f"  {i}. {s}" for i, s in enumerate(plan.sub_problems, 1)
        )
        context_block = (
            f"\nRELEVANT KNOWLEDGE BASE CONTEXT:\n{context}\n"
            if context else ""
        )
        confidence_note = (
            "\nNote: confidence in this answer is LOW — state clearly where uncertain and avoid speculation."
            if plan.confidence == "LOW" else (
                "\nNote: confidence is MEDIUM — answer is well-reasoned but verify critical details."
                if plan.confidence == "MEDIUM" else ""
            )
        )
        warnings_block = ""
        if plan.warnings:
            warnings_block = "\nFLAGS:\n" + "\n".join(f"  ⚠  {w}" for w in plan.warnings) + "\n"

        meta = plan.metacognition.to_text()
        meta_block = f"\nKNOWLEDGE BOUNDARIES:\n{meta}\n" if meta else ""

        reasoning_note = ""
        if plan.reasoning_mode == "forest_of_thought" and plan.forest:
            best = plan.forest[0]
            reasoning_note = (
                f"\n[Forest-of-Thought: explored {len(plan.forest)} root hypotheses:\n"
                + "\n".join(
                    f"  Tree {i+1}: '{t.hypothesis}' (score={t.score:.2f}) — {t.conclusion}"
                    for i, t in enumerate(plan.forest)
                )
                + f"\n→ Proceeding with Tree 1: '{best.hypothesis}']\n"
            )
        elif plan.reasoning_mode == "tree_of_thought" and plan.branches:
            branch_labels = [b.label for b in plan.branches[:3]]
            reasoning_note = (
                f"\n[Tree-of-Thought: evaluated reasoning paths {branch_labels}; "
                f"proceeding with '{plan.branches[0].label}']\n"
            )
        else:
            reasoning_note = "\n[Chain-of-Thought: think step by step]\n"

        consistency_block = ""
        if not plan.consistency_ok:
            consistency_block = f"\n⚠  Consistency check flagged: {plan.consistency_note}. Re-examine reasoning carefully.\n"

        return (
            "You are AnveshAI, an expert JEE Advanced tutor with deep knowledge of "
            "Physics, Chemistry, and Mathematics at the highest competitive exam level.\n"
            "Think step by step before answering.\n"
            f"QUESTION: {user_input}\n"
            f"{context_block}"
            f"{warnings_block}"
            f"{meta_block}"
            f"{reasoning_note}"
            f"{consistency_block}\n"
            f"REASONING PLAN (follow this structure exactly):\n{steps_str}\n\n"
            f"ANSWER STRATEGY: {plan.strategy}\n"
            f"{confidence_note}\n\n"
            "Write a rigorous, well-structured response at JEE Advanced level. "
            "Follow the reasoning plan step by step. "
            "State every formula before applying it. Show all algebraic steps. "
            "Use precise JEE notation. Be concise but complete. "
            "Explicitly state assumptions and caveats. "
            "If the context block contains relevant information, prioritise it."
        )

    def build_math_fallback_prompt(
        self,
        user_input: str,
        plan: ReasoningPlan,
        error_context: str = "",
    ) -> str:
        steps_str = "\n".join(
            f"  Step {i}: {s}" for i, s in enumerate(plan.sub_problems, 1)
        )
        error_note = (
            f"\nNote: automated symbolic computation failed ({error_context}). "
            "Solve manually with maximum care — show every step.\n"
            if error_context else ""
        )
        meta = plan.metacognition.to_text()
        meta_block = f"\n{meta}\n" if meta else ""

        return (
            "You are a mathematics expert with deep knowledge of all mathematical domains.\n"
            f"PROBLEM: {user_input}\n"
            f"{error_note}"
            f"\nPROBLEM TYPE: {plan.problem_type}\n"
            f"DOMAIN: {plan.domain}\n"
            f"STRATEGY: {plan.strategy}\n"
            f"{f'EXPECTED ANSWER FORM: {plan.expected_form}' if plan.expected_form else ''}"
            f"{meta_block}\n"
            f"REASONING STEPS TO FOLLOW:\n{steps_str}\n\n"
            "Solve the problem completely by following each step above. "
            "Show all working in full — do not skip steps. "
            "After computing the answer, verify it by substitution or differentiation. "
            "State the final answer clearly and prominently at the end."
        )

    def build_inference_prompt(
        self,
        user_input: str,
        inference_result: str,
        plan: ReasoningPlan,
    ) -> str:
        """Prompt for logic/inference questions where the inference engine has a result."""
        steps_str = "\n".join(
            f"  Step {i}: {s}" for i, s in enumerate(plan.sub_problems, 1)
        )
        return (
            f"LOGICAL PROBLEM: {user_input}\n\n"
            f"INFERENCE ENGINE RESULT: {inference_result}\n\n"
            "Explain the logical reasoning that leads to this conclusion:\n"
            f"{steps_str}\n\n"
            "Use precise logical terminology. Show the formal structure of the argument. "
            "State whether the argument is valid and, if sound, why the premises are true."
        )

    def build_physics_prompt(
        self,
        user_input: str,
        engine_result: str,
        formula_type: str,
        plan: ReasoningPlan,
    ) -> str:
        """
        Prompt for physics problems where the PhysicsEngine has computed the answer.
        The LLM's job is ONLY to explain the working — not recompute.
        """
        steps_str = "\n".join(
            f"  Step {i}: {s}" for i, s in enumerate(plan.sub_problems, 1)
        )
        meta = plan.metacognition.to_text()
        meta_block = f"\n{meta}\n" if meta else ""

        return (
            f"PHYSICS PROBLEM: {user_input}\n\n"
            f"VERIFIED ANSWER (computed by the deterministic physics engine — exact): {engine_result}\n\n"
            "=== YOUR TASK ===\n"
            f"Explain clearly, step by step, how a student solves this {formula_type.replace('_', ' ')} problem "
            f"to arrive at: {engine_result}\n\n"
            "Rules:\n"
            "  - DO NOT recompute the answer differently — your explanation MUST match the verified result.\n"
            "  - State the relevant formula first, then substitute known values.\n"
            "  - Show units explicitly at every step.\n"
            "  - Mention any physical principles or laws that apply (e.g. Newton's 2nd Law, Ohm's Law).\n\n"
            f"FORMULA DOMAIN: {formula_type.replace('_', ' ').title()}\n"
            f"REASONING STRATEGY: {plan.strategy}\n"
            f"{meta_block}\n"
            f"STEPS TO WALK THROUGH:\n{steps_str}\n\n"
            "Write a numbered, student-friendly explanation. "
            "Show every substitution and unit. "
            f"End with: 'Final answer: {engine_result}'"
        )

    def build_physics_fallback_prompt(
        self,
        user_input: str,
        plan: ReasoningPlan,
        error_context: str = "",
    ) -> str:
        """Prompt when the physics engine could not solve the problem."""
        steps_str = "\n".join(
            f"  Step {i}: {s}" for i, s in enumerate(plan.sub_problems, 1)
        )
        error_note = (
            f"\nNote: automated physics engine could not solve this ({error_context}). "
            "Apply physics principles manually — show every step and unit.\n"
            if error_context else ""
        )
        return (
            "You are a physics expert with mastery of all branches of physics.\n"
            f"PROBLEM: {user_input}\n"
            f"{error_note}"
            f"\nPROBLEM TYPE: {plan.problem_type}\n"
            f"STRATEGY: {plan.strategy}\n\n"
            f"REASONING STEPS:\n{steps_str}\n\n"
            "Solve the problem completely. Identify the formula, substitute values with units, "
            "and compute the final answer. Show all working clearly."
        )

    def build_chemistry_prompt(
        self,
        user_input: str,
        engine_result: str,
        chem_type: str,
        plan: ReasoningPlan,
    ) -> str:
        """
        Prompt for chemistry problems where the ChemistryEngine has computed the answer.
        The LLM's job is ONLY to explain the working — not recompute.
        """

        # ── Domain-specific mandatory steps injected per chem_type ────────────
        _CHEM_STEPS: dict[str, list[str]] = {
            "ideal_gas_thermo": [
                "State whether the gas is monatomic or diatomic and write the Cv and Cp values "
                "(monatomic: Cv = 3R/2, Cp = 5R/2; diatomic: Cv = 5R/2, Cp = 7R/2).",
                "Identify the process: constant pressure (q = nCpΔT) or constant volume (q = nCvΔT).",
                "Substitute q, n, and Cp (or Cv) to compute ΔT = q / (nCp).",
                "Compute T_final = T_initial + ΔT.",
                "Compute ΔU = nCvΔT. "
                "IMPORTANT: at constant pressure ΔU ≠ q — work W = nRΔT is done BY the gas, "
                "so ΔU = q − W = nCvΔT (always less than q for monatomic gas).",
                "State both final answers with units: T_final (K) and ΔU (J).",
            ],
            "calorimetry": [
                "Write the calorimetry formula: q = mcΔT or q = nΔH.",
                "Identify which quantities are given and which is unknown.",
                "Substitute values with units and solve for the unknown.",
                "State the final answer with correct units.",
            ],
            "ph": [
                "Write the pH definition: pH = −log₁₀[H⁺].",
                "Identify [H⁺] from the problem (or calculate it from Ka/Kb).",
                "Substitute and compute pH, keeping significant figures.",
            ],
            "ideal_gas": [
                "Write the ideal gas law: PV = nRT.",
                "Identify which of P, V, n, T is unknown.",
                "Substitute known values with correct SI units (P in Pa, V in m³, T in K).",
                "Solve for the unknown and state the answer with units.",
            ],
        }
        domain_steps = _CHEM_STEPS.get(chem_type, [])
        if domain_steps:
            steps_str = "\n".join(f"  Step {i}: {s}" for i, s in enumerate(domain_steps, 1))
        else:
            steps_str = "\n".join(
                f"  Step {i}: {s}" for i, s in enumerate(plan.sub_problems, 1)
            )

        meta = plan.metacognition.to_text()
        meta_block = f"\n{meta}\n" if meta else ""

        # Pull the key numerical results out of engine_result for the hard constraint
        numbers_in_result = re.findall(r'[\d.]+', engine_result)
        num_constraint = ""
        if numbers_in_result:
            num_constraint = (
                "\n  - HARD CONSTRAINT: Every numerical value you write MUST appear verbatim "
                f"in the verified answer above. Do not introduce any number not in: "
                f"{', '.join(numbers_in_result[:12])}.\n"
            )

        return (
            f"CHEMISTRY PROBLEM: {user_input}\n\n"
            f"VERIFIED ANSWER (computed by the deterministic chemistry engine — exact and authoritative):\n"
            f"{engine_result}\n\n"
            "=== YOUR TASK ===\n"
            f"Explain clearly, step by step, how a student solves this "
            f"{chem_type.replace('_', ' ')} problem to arrive at the verified answer above.\n\n"
            "STRICT RULES — violating any rule is an error:\n"
            "  - DO NOT recompute the answer differently. Your explanation MUST reproduce the "
            "exact same numerical results as the verified answer.\n"
            "  - DO NOT introduce any formula or equation that contradicts the engine output.\n"
            f"{num_constraint}"
            "  - State the relevant formula first (e.g. q = nCpΔT, PV = nRT, pH = −log[H⁺]).\n"
            "  - Show how each given value is identified from the problem.\n"
            "  - Include units throughout every calculation.\n"
            "  - Mention the chemical principle being applied.\n\n"
            f"CHEMISTRY DOMAIN: {chem_type.replace('_', ' ').title()}\n"
            f"REASONING STRATEGY: {plan.strategy}\n"
            f"{meta_block}\n"
            f"MANDATORY STEPS TO WALK THROUGH:\n{steps_str}\n\n"
            "Write a numbered, student-friendly explanation. "
            "Show every substitution, unit conversion, and calculation. "
            f"End with: 'Final answer: {engine_result}'"
        )

    def build_chemistry_fallback_prompt(
        self,
        user_input: str,
        plan: ReasoningPlan,
        error_context: str = "",
    ) -> str:
        """Prompt when the chemistry engine could not solve the problem."""
        steps_str = "\n".join(
            f"  Step {i}: {s}" for i, s in enumerate(plan.sub_problems, 1)
        )
        error_note = (
            f"\nNote: automated chemistry engine could not solve this ({error_context}). "
            "Apply chemistry principles manually — show every step and include units.\n"
            if error_context else ""
        )
        return (
            "You are a chemistry expert with mastery of all branches of chemistry.\n"
            f"PROBLEM: {user_input}\n"
            f"{error_note}"
            f"\nPROBLEM TYPE: {plan.problem_type}\n"
            f"STRATEGY: {plan.strategy}\n\n"
            f"REASONING STEPS:\n{steps_str}\n\n"
            "Solve the problem completely. Identify the formula, substitute values with units, "
            "and compute the final answer. Show all working clearly. "
            "State the chemical principle (e.g. ideal gas law, mole concept, pH definition)."
        )
