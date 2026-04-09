"""
error_analysis.py — AnveshAI Edge pipeline error breakdown.

Runs a labelled test dataset through:
  1. classify_intent()         → detects routing errors
  2. domain engine .solve()    → detects solver limitation errors
  3. output format checks      → detects formatting errors
  4. exception trapping        → detects parsing / pre-processing errors

Prints a final report:

  Error Type            % of total
  --------------------------------
  Parsing errors        ?
  Routing errors        ?
  Solver limitations    ?
  Formatting errors     ?
"""

import sys
import os
import traceback
from dataclasses import dataclass, field
from typing import Optional

sys.path.insert(0, os.path.dirname(__file__))

# ─── Lazy imports so import errors are caught as parsing errors ───────────────
def _import_router():
    from router import classify_intent
    return classify_intent

def _import_math_engine():
    from advanced_math_engine import solve as math_solve
    return math_solve

def _import_physics_engine():
    from physics_engine import PhysicsEngine
    return PhysicsEngine()

def _import_chemistry_engine():
    from chemistry_engine import ChemistryEngine
    return ChemistryEngine()


# ─────────────────────────────────────────────────────────────────────────────
# Dataset: (input_text, expected_intent, expect_solver_success)
#   expect_solver_success = True  → engine should return success=True
#   expect_solver_success = False → engine known limitation; expected failure
#   expect_solver_success = None  → engine not tested for this case
# ─────────────────────────────────────────────────────────────────────────────

TEST_CASES: list[tuple[str, str, Optional[bool]]] = [

    # ── SYSTEM ──────────────────────────────────────────────────────────────
    ("/help",                                      "system",        None),
    ("/exit",                                      "system",        None),
    ("/history",                                   "system",        None),
    ("/clear",                                     "system",        None),

    # ── LOGIC ────────────────────────────────────────────────────────────────
    ("If P then Q. P is true. What follows?",      "logic",         None),
    ("All metals are conductors. Iron is a metal. What follows?",
                                                   "logic",         None),
    ("Either A or B. Not A. Therefore?",           "logic",         None),
    ("Is the argument valid: P implies Q, not Q, therefore not P?",
                                                   "logic",         None),
    ("What is the contrapositive of 'if p then q'?",
                                                   "logic",         None),
    ("Modus ponens: if it rains, the ground is wet. It rains.",
                                                   "logic",         None),
    ("Is this a tautology: (P or not P)?",         "logic",         None),
    ("Construct a truth table for P and Q.",       "logic",         None),
    ("If all birds fly and penguins are birds, does it follow that penguins fly?",
                                                   "logic",         None),
    ("Hypothetical syllogism: if A then B, if B then C.",
                                                   "logic",         None),

    # ── PHYSICS ─────────────────────────────────────────────────────────────
    ("A ball is thrown with initial velocity 20 m/s at 30 degrees. Find range.",
                                                   "physics",       True),
    ("Calculate the centripetal force on a 2 kg object moving in a circle of radius 0.5 m at 3 m/s.",
                                                   "physics",       True),
    ("Find the escape velocity of Earth given g=9.8 and R=6.4e6 m.",
                                                   "physics",       True),
    ("A spring with k=200 N/m is compressed 0.1 m. Find the elastic potential energy.",
                                                   "physics",       True),
    ("Find the de Broglie wavelength of an electron moving at 2e6 m/s.",
                                                   "physics",       True),
    ("A resistor of 10 ohm is connected to 5V. Find current.",
                                                   "physics",       True),
    ("Find the magnetic force on a charge q=2e-6 C moving at v=3e5 m/s in B=0.4 T.",
                                                   "physics",       True),
    ("An ideal gas expands isothermally. Find work done.",
                                                   "physics",       False),  # partial
    ("Find the moment of inertia of a solid sphere of mass 5 kg and radius 0.3 m.",
                                                   "physics",       True),
    ("Photoelectric effect: threshold frequency of sodium is 5.5e14 Hz. Find work function.",
                                                   "physics",       True),
    ("A particle undergoes simple harmonic motion with amplitude 0.05 m and frequency 2 Hz.",
                                                   "physics",       False),  # underspecified
    ("Two charges +3 µC and -3 µC are placed 0.2 m apart. Find the force.",
                                                   "physics",       True),
    ("Find the gravitational potential energy between two 10 kg masses 5 m apart.",
                                                   "physics",       True),
    ("Speed of sound in air at 25°C.",             "physics",       False),  # no numbers
    ("Diffraction grating has 500 lines/mm. Find 1st order angle for 600 nm light.",
                                                   "physics",       True),

    # ── CHEMISTRY ────────────────────────────────────────────────────────────
    ("Calculate molar mass of H2SO4.",             "chemistry",     True),
    ("How many moles are in 44 g of CO2?",         "chemistry",     True),
    ("Balance the equation: H2 + O2 → H2O.",      "chemistry",     True),
    ("Find the pH of 0.01 M HCl solution.",        "chemistry",     True),
    ("Calculate the molarity of 10 g NaOH in 500 mL solution.",
                                                   "chemistry",     True),
    ("Find empirical formula if C=40%, H=6.67%, O=53.33%.",
                                                   "chemistry",     True),
    ("Ka of acetic acid is 1.8e-5. Find pH of 0.1 M solution.",
                                                   "chemistry",     True),
    ("Calculate the oxidation state of Mn in KMnO4.",
                                                   "chemistry",     True),
    ("How many grams of NaCl are produced from 2 mol Na and excess Cl2?",
                                                   "chemistry",     True),
    ("Rate constant k=2e-3 s^-1. Find half-life for first order reaction.",
                                                   "chemistry",     True),
    ("Find the normality of 0.5 M H2SO4.",         "chemistry",     True),
    ("What is the enthalpy of combustion of methane?",
                                                   "knowledge",     None),   # conceptual: no data provided
    ("Calculate pOH of 0.001 M NaOH.",             "chemistry",     True),
    ("Number of moles in 11.2 L of gas at STP.",   "chemistry",     True),
    ("Find the degree of dissociation of 0.1 M weak acid with Ka=1e-4.",
                                                   "chemistry",     True),

    # ── ADVANCED MATH ────────────────────────────────────────────────────────
    ("integrate x^2 sin(x)",                       "advanced_math", True),
    ("differentiate x^3 + 2x^2 - 5",              "advanced_math", True),
    ("solve x^2 - 5x + 6 = 0",                    "advanced_math", True),
    ("limit of sin(x)/x as x approaches 0",        "advanced_math", True),
    ("definite integral of x^2 from 0 to 3",       "advanced_math", True),
    ("find the determinant of [[1,2],[3,4]]",       "advanced_math", True),
    ("solve differential equation y'' + y = 0",    "advanced_math", True),
    ("expand (x+y)^5 using binomial theorem",       "advanced_math", True),
    ("find eigenvalues of [[2,1],[1,2]]",           "advanced_math", True),
    ("taylor series of sin(x) around x=0",         "advanced_math", True),
    ("sum of n^2 from n=1 to 10",                  "advanced_math", True),
    ("find GCD of 48 and 18",                      "advanced_math", True),
    ("factor x^3 - 6x^2 + 11x - 6",               "advanced_math", True),
    ("partial fraction decomposition of 1/(x^2-1)",
                                                   "advanced_math", True),
    ("solve system of equations: 2x+3y=7, x-y=1", "advanced_math", True),
    ("find inverse of matrix [[1,2],[3,4]]",        "advanced_math", True),
    ("laplace transform of t*e^(-2t)",             "advanced_math", True),
    ("find permutations P(8,3)",                   "advanced_math", True),
    ("combination C(10,4)",                        "advanced_math", True),
    ("10C3",                                       "advanced_math", True),
    ("mean of [2, 4, 6, 8, 10]",                  "advanced_math", True),
    ("standard deviation of [2, 4, 4, 4, 5, 5, 7, 9]",
                                                   "advanced_math", True),
    ("integrate from 0 to pi of sin^2(x)",         "advanced_math", True),
    ("second derivative of sin(x) * e^x",          "advanced_math", True),
    ("solve ode dy/dx = y",                        "advanced_math", True),

    # ── MATH (simple arithmetic) ──────────────────────────────────────────────
    ("2 + 3 * 4",                                  "math",          None),
    ("(15 - 3) / 4",                               "math",          None),
    ("2^10",                                       "math",          None),
    ("100 % 7",                                    "math",          None),
    ("calculate 3.14 * 2^2",                       "math",          None),

    # ── KNOWLEDGE ────────────────────────────────────────────────────────────
    ("What is Newton's second law?",               "knowledge",     None),
    ("Explain the Central Limit Theorem.",         "knowledge",     None),
    ("What is the difference between mitosis and meiosis?",
                                                   "knowledge",     None),
    ("What causes the greenhouse effect?",         "knowledge",     None),
    ("Define entropy in thermodynamics.",          "knowledge",     None),

    # ── CONVERSATION ─────────────────────────────────────────────────────────
    ("Hello!",                                     "conversation",  None),
    ("Thanks for your help.",                      "conversation",  None),
    ("Who made you?",                              "conversation",  None),

    # ── EDGE CASES (known tricky routing) ─────────────────────────────────────
    # Chemistry keywords that should NOT go to physics
    ("Enthalpy of reaction for CH4 + 2O2 → CO2 + 2H2O is -890 kJ/mol.",
                                                   "chemistry",     False),
    # Math-looking vector problem — cross product handler not implemented (known limitation)
    ("Find a×b where a=2i+j and b=3i-2j.",        "advanced_math", False),
    # nCr should NOT be captured by logic
    ("C(n, 2) = 15, find n.",                      "advanced_math", True),
    ("10C3 equals what?",                          "advanced_math", True),
    # Thermodynamics process without numeric data
    ("An ideal gas expands isothermally at 300K. Calculate work done if volume doubles.",
                                                   "physics",       False),
    # Knowledge question that starts with 'what is' but contains physics keywords
    ("What is escape velocity?",                   "knowledge",     None),
    # Ambiguous: could be chemistry or knowledge
    ("What is pH?",                                "knowledge",     None),
    # Parsing edge: unicode characters in input
    ("Find ∫x² dx",                               "advanced_math", True),
    # Parsing edge: very long input with embedded math
    ("A student invests Rs 5000 at 8% per annum compound interest. "
     "Find the amount after 3 years.",             "advanced_math", True),
    # Parsing edge: empty-ish input
    ("   ",                                        "conversation",  None),
    # Parsing edge: only special characters
    ("???",                                        "conversation",  None),
    # Routing edge: 'if' in math question should NOT trigger logic
    ("Find if the matrix [[1,2],[3,4]] is invertible.",
                                                   "advanced_math", True),
    # Routing edge: 'implies' in a non-logic context
    ("The result implies that x=5.",               "conversation",  None),
]


# ─────────────────────────────────────────────────────────────────────────────
# Error record
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ErrorRecord:
    error_type: str        # "parsing" | "routing" | "solver" | "formatting"
    input_text: str
    expected: str
    got: str
    detail: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# Output format checker
# ─────────────────────────────────────────────────────────────────────────────

def _is_well_formed_result(result_str: str) -> bool:
    """Return True if the solver output string is non-empty and sane."""
    if not result_str or not isinstance(result_str, str):
        return False
    stripped = result_str.strip()
    if len(stripped) == 0:
        return False
    # Detect placeholder / error bleed-through strings
    _BAD_MARKERS = [
        "none", "error:", "traceback", "exception", "nan", "undefined",
        "notimplementederror", "typeerror", "valueerror",
    ]
    lower = stripped.lower()
    for marker in _BAD_MARKERS:
        if lower.startswith(marker):
            return False
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Main runner
# ─────────────────────────────────────────────────────────────────────────────

def run_analysis(verbose: bool = False) -> None:
    errors: list[ErrorRecord] = []

    # ── Import engines (any ImportError → parsing / environment error) ─────
    try:
        classify_intent = _import_router()
    except Exception as exc:
        print(f"[FATAL] Cannot import router: {exc}")
        sys.exit(1)

    try:
        math_solve = _import_math_engine()
    except Exception as exc:
        print(f"[FATAL] Cannot import advanced_math_engine: {exc}")
        sys.exit(1)

    try:
        physics_engine = _import_physics_engine()
    except Exception as exc:
        print(f"[FATAL] Cannot import physics_engine: {exc}")
        sys.exit(1)

    try:
        chemistry_engine = _import_chemistry_engine()
    except Exception as exc:
        print(f"[FATAL] Cannot import chemistry_engine: {exc}")
        sys.exit(1)

    total = len(TEST_CASES)
    passed = 0

    for idx, (text, expected_intent, expect_solver_ok) in enumerate(TEST_CASES, 1):

        # ── Step 1: Preprocessing / parsing check ──────────────────────────
        try:
            actual_intent = classify_intent(text)
        except Exception as exc:
            tb = traceback.format_exc(limit=2)
            errors.append(ErrorRecord(
                error_type="parsing",
                input_text=text,
                expected=expected_intent,
                got="<exception>",
                detail=f"classify_intent raised: {type(exc).__name__}: {exc}",
            ))
            if verbose:
                print(f"[{idx:03d}] PARSING ERROR  » {text[:60]!r}")
                print(f"         {tb.strip()}")
            continue

        # ── Step 2: Routing check ───────────────────────────────────────────
        if actual_intent != expected_intent:
            errors.append(ErrorRecord(
                error_type="routing",
                input_text=text,
                expected=expected_intent,
                got=actual_intent,
                detail=f"expected={expected_intent!r}, got={actual_intent!r}",
            ))
            if verbose:
                print(f"[{idx:03d}] ROUTING ERROR  » {text[:60]!r}")
                print(f"         expected={expected_intent!r}, got={actual_intent!r}")
            continue  # skip solver check if routing is wrong

        # ── Step 3: Solver / engine check ──────────────────────────────────
        if expect_solver_ok is True and actual_intent in ("advanced_math", "physics", "chemistry"):
            try:
                if actual_intent == "advanced_math":
                    success, result_str, _latex = math_solve(text)
                elif actual_intent == "physics":
                    success, result_str, _ftype = physics_engine.solve(text)
                else:  # chemistry
                    success, result_str, _ctype = chemistry_engine.solve(text)

                if not success:
                    errors.append(ErrorRecord(
                        error_type="solver",
                        input_text=text,
                        expected=expected_intent,
                        got=actual_intent,
                        detail=f"Engine returned success=False: {result_str}",
                    ))
                    if verbose:
                        print(f"[{idx:03d}] SOLVER LIMIT   » {text[:60]!r}")
                        print(f"         {result_str}")
                    continue

                # ── Step 4: Formatting check ────────────────────────────────
                if not _is_well_formed_result(result_str):
                    errors.append(ErrorRecord(
                        error_type="formatting",
                        input_text=text,
                        expected=expected_intent,
                        got=actual_intent,
                        detail=f"Malformed output: {result_str!r}",
                    ))
                    if verbose:
                        print(f"[{idx:03d}] FORMAT ERROR   » {text[:60]!r}")
                        print(f"         output={result_str!r}")
                    continue

            except Exception as exc:
                errors.append(ErrorRecord(
                    error_type="parsing",
                    input_text=text,
                    expected=expected_intent,
                    got=actual_intent,
                    detail=f"Engine raised {type(exc).__name__}: {exc}",
                ))
                if verbose:
                    print(f"[{idx:03d}] PARSING ERROR  » {text[:60]!r}")
                    print(f"         {traceback.format_exc(limit=2).strip()}")
                continue

        passed += 1

    # ── Build report ─────────────────────────────────────────────────────────
    n_parsing    = sum(1 for e in errors if e.error_type == "parsing")
    n_routing    = sum(1 for e in errors if e.error_type == "routing")
    n_solver     = sum(1 for e in errors if e.error_type == "solver")
    n_formatting = sum(1 for e in errors if e.error_type == "formatting")
    n_errors     = len(errors)
    n_clean      = total - n_errors

    def pct(n: int) -> str:
        return f"{100 * n / total:.1f}%" if total else "0.0%"

    sep = "─" * 48
    print()
    print("=" * 48)
    print("  AnveshAI Edge — Pipeline Error Analysis")
    print("=" * 48)
    print(f"  Total test cases : {total}")
    print(f"  Passed (no error): {n_clean}  ({pct(n_clean)})")
    print(f"  Total errors     : {n_errors}  ({pct(n_errors)})")
    print()
    print(f"  {'Error Type':<26} {'Count':>6}   {'% of total':>10}")
    print(f"  {sep}")
    print(f"  {'Parsing errors':<26} {n_parsing:>6}   {pct(n_parsing):>10}")
    print(f"  {'Routing errors':<26} {n_routing:>6}   {pct(n_routing):>10}")
    print(f"  {'Solver limitations':<26} {n_solver:>6}   {pct(n_solver):>10}")
    print(f"  {'Formatting errors':<26} {n_formatting:>6}   {pct(n_formatting):>10}")
    print(f"  {sep}")
    print(f"  {'TOTAL errors':<26} {n_errors:>6}   {pct(n_errors):>10}")
    print("=" * 48)

    if errors:
        print()
        print("  ── Detailed Error Log ──────────────────────────")
        for e in errors:
            label = {
                "parsing":    "PARSE ",
                "routing":    "ROUTE ",
                "solver":     "SOLVER",
                "formatting": "FORMAT",
            }.get(e.error_type, e.error_type.upper()[:6])
            print(f"\n  [{label}] {e.input_text[:65]!r}")
            print(f"           {e.detail}")
        print()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    verbose_mode = "--verbose" in sys.argv or "-v" in sys.argv
    run_analysis(verbose=verbose_mode)
