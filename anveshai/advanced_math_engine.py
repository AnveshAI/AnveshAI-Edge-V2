"""
Advanced Math Engine v2 — symbolic computation using SymPy.

Handles a wide range of advanced mathematics:
    ─ Indefinite & definite integration
    ─ Differentiation (any order, any variable)
    ─ Limits (including one-sided and infinity)
    ─ Equation & system solving
    ─ Ordinary differential equations (ODEs)
    ─ Matrix operations (det, inverse, eigenvalues, rank, trace)
    ─ Taylor / Maclaurin series expansion
    ─ Laplace & inverse Laplace transforms
    ─ Fourier transform
    ─ Simplification, factoring, expansion, partial fractions
    ─ Number theory (GCD, LCM, prime factorization, modular arithmetic)
    ─ Statistics (mean, variance, std deviation, median)
    ─ Combinatorics (factorial, binomial coefficients, permutations)
    ─ Complex number operations
    ─ Summations & products
    ─ Trigonometric identity simplification

The engine parses natural language ("integrate x^2 sin(x)"), runs the
computation symbolically with SymPy, and returns:
    - a clean string result
    - a LaTeX representation

The result is then handed to the LLM, which is TOLD the correct answer
and must only produce the step-by-step explanation — preventing hallucination.
"""

import re
from typing import Optional, Tuple

from preprocess import normalize_input


# ─────────────────────────────────────────────────────────────────────────────
# Operation keyword registry
# ─────────────────────────────────────────────────────────────────────────────

_ADVANCED_OPS: dict[str, list[str]] = {
    "competition_math": [
        # Multi-person same-time arrival (AIME/AMC style)
        "all arrived at the same time",
        "all three arrived",
        "arrived at the park at the same time",
        "all three people arrived",
        "started walking at a constant speed",
        "started running at a constant speed",
        "started bicycling",
        "miles per hour faster than",
        "relatively prime positive integers",
        "find m+n",
        "m and n are relatively prime",
        "m+n",
        "constant speed along",
        "same straight road",
        "hours after",
        "one hour after",
        "two hours after",
    ],
    "word_problem": [
        # Geometry word problems
        "area of a rectangle", "area of the rectangle",
        "area of a square", "area of a circle", "area of a triangle",
        "area of the triangle", "area of a trapezoid", "area of a parallelogram",
        "perimeter of a", "perimeter of the",
        "volume of a cube", "volume of a cuboid", "volume of a cylinder",
        "volume of a sphere", "volume of a cone", "volume of the",
        "circumference of", "surface area of",
        # Percentage / interest
        "percent of", "% of", "percentage of",
        "simple interest", "compound interest",
        "increased by %", "decreased by %", "discount of",
        "profit of", "loss of", "markup of",
        # Rate / proportion word problems
        "how many days", "how many hours", "work together",
        "rate of work", "fills the tank", "pipes",
        "ratio of", "proportion",
        # Speed-distance-time (simple, no physics keywords)
        "miles per hour", "km per hour", "kmph", "mph",
        "average speed", "total distance", "time taken to travel",
    ],
    "integrate": [
        "integrate", "integral of", "antiderivative of", "indefinite integral",
        "definite integral", "∫",
    ],
    "differentiate": [
        "differentiate", "derivative of", "d/dx", "d/dy", "d/dz", "d/dt",
        "diff of", "first derivative", "second derivative", "third derivative",
        "nth derivative", "partial derivative",
    ],
    "limit": [
        "limit of", "limit as", "lim ", "lim(", "find the limit",
    ],
    "solve": [
        "solve ", "find roots of", "zeros of", "find x such that",
        "find the value of x", "find the solution",
    ],
    "ode": [
        "differential equation", "ode ", "ordinary differential",
        "dsolve", "solve the ode", "solve ode", "y'' ", "y' ",
        "d2y", "d^2y", "solve the differential",
    ],
    "eigenvalue": [
        "eigenvalue", "eigenvector", "eigen value", "eigen vector",
        "characteristic polynomial",
    ],
    "determinant": [
        "determinant of", "det of", "det(",
    ],
    "inverse": [
        "inverse of matrix", "matrix inverse", "inverse matrix",
    ],
    "matrix_rank": [
        "rank of matrix", "matrix rank", "rank(",
    ],
    "matrix_trace": [
        "trace of matrix", "matrix trace", "trace(",
    ],
    "series": [
        "taylor series", "maclaurin series", "series expansion",
        "expand in series", "power series",
    ],
    "laplace": [
        "laplace transform", "laplace of", "l{", "l(",
    ],
    "inverse_laplace": [
        "inverse laplace", "laplace inverse", "l^-1",
    ],
    "fourier": [
        "fourier transform", "fourier of",
    ],
    "simplify": [
        "simplify ", "simplify(", "reduce ",
    ],
    "trig_simplify": [
        "simplify trig", "trig simplif", "trigonometric simplif",
        "simplify the trigonometric",
    ],
    "factor": [
        "factor ", "factorise ", "factorize ", "factorise(", "factor(",
    ],
    "expand": [
        "expand ", "expand(",
    ],
    "partial_fraction": [
        "partial fraction", "partial fractions", "partial fraction decomposition",
    ],
    "gcd": [
        "gcd(", "gcd of", "greatest common divisor", "highest common factor",
        "hcf of",
    ],
    "lcm": [
        "lcm(", "lcm of", "least common multiple", "lowest common multiple",
    ],
    "prime_factors": [
        "prime factor", "prime factorization", "factorise into primes",
        "factorize into primes", "prime decomposition",
    ],
    "modular": [
        " mod ", "modulo ", "modular arithmetic", "modular inverse",
        "congruence",
    ],
    "statistics": [
        "mean of", "average of", "median of", "mode of",
        "variance of", "standard deviation of", "std dev of", "std(",
        "statistics of",
    ],
    "factorial": [
        "factorial of", "factorial(", "! ", "n factorial",
    ],
    "binomial": [
        "binomial coefficient", "choose ", "c(", "combinations of",
        "nCr", "ncr", "10c3", "10c4", "nC",
    ],
    "permutation": [
        "permutation", "nPr", "arrangements of",
    ],
    "summation": [
        "sum of ", "summation of", "sigma notation",
    ],
    "product": [
        "product of ", "∏", "pi product",
    ],
    "complex_ops": [
        "complex number", "real part", "imaginary part", "modulus of",
        "argument of", "conjugate of",
    ],
}


def detect_advanced_operation(text: str) -> Optional[str]:
    """Return the detected advanced math operation (highest-priority match), or None."""
    ranked = detect_advanced_operation_ranked(text)
    return ranked[0] if ranked else None


# Extra regex-based detectors for patterns that can't be keywords
_NCR_PATTERN  = re.compile(r'\b(\d+)\s*[Cc]\s*(\d+)\b')           # 10C3, 10c3
_NPR_PATTERN  = re.compile(r'\b(\d+)\s*[Pp]\s*(\d+)\b')           # 10P3
_INVERT_PATTERN = re.compile(r'\bmatrix\b.*\binvertible\b'
                              r'|\binvertible\b.*\bmatrix\b', re.I)


def detect_advanced_operation_ranked(text: str) -> list[str]:
    """
    Return an ordered list of candidate operations (best match first).

    The primary candidate is determined by priority-ordered keyword matching.
    A secondary candidate is added when a plausible alternative exists, so
    that solve() can fall back to it if the primary handler fails.

    Returns [] if no operation is detected.
    """
    lowered = text.lower()

    # Priority ordering — more specific ops first
    priority_order = [
        "competition_math",
        "word_problem",
        "trig_simplify", "inverse_laplace", "laplace", "fourier",
        "ode", "eigenvalue", "determinant", "inverse", "matrix_rank",
        "matrix_trace", "partial_fraction", "prime_factors", "modular",
        "statistics", "binomial", "permutation", "factorial",
        "summation", "product", "complex_ops", "gcd", "lcm",
        "integrate", "differentiate", "limit", "series",
        "simplify", "factor", "expand", "solve",
    ]

    # ── Regex-based overrides (run before keyword table) ──────────────────────
    if _NCR_PATTERN.search(text):
        return ["binomial"]
    if _NPR_PATTERN.search(text):
        return ["permutation"]
    if _INVERT_PATTERN.search(text):
        return ["determinant"]

    # ── Keyword-table scan ────────────────────────────────────────────────────
    primary: Optional[str] = None
    for op in priority_order:
        keywords = _ADVANCED_OPS.get(op, [])
        for kw in keywords:
            if kw in lowered:
                primary = op
                break
        if primary:
            break

    if primary is None:
        return []

    # ── Secondary candidate (top-2 prediction) ────────────────────────────────
    # Heuristic: if the primary op's handler is likely to fail on this input
    # (natural-language form), add an alternative that's more forgiving.
    secondary: Optional[str] = None
    after_primary = False
    for op in priority_order:
        if op == primary:
            after_primary = True
            continue
        if not after_primary:
            continue
        keywords = _ADVANCED_OPS.get(op, [])
        for kw in keywords:
            if kw in lowered:
                secondary = op
                break
        if secondary:
            break

    result = [primary]
    if secondary:
        result.append(secondary)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Expression helpers
# ─────────────────────────────────────────────────────────────────────────────

def _preprocess(expr: str) -> str:
    """Normalise user-written math to SymPy-parseable syntax."""
    expr = expr.strip()
    # Remove trailing differential (dx, dy, dt, …) for integrals
    expr = re.sub(r'\s*d[a-zA-Z]\s*$', '', expr)
    # Remove "= 0" for equation solving — SymPy's solve() takes LHS
    expr = re.sub(r'\s*=\s*0\s*$', '', expr)
    # Replace ^ with **
    expr = expr.replace('^', '**')
    # Natural log → log
    expr = re.sub(r'\bln\b', 'log', expr)
    # arc functions
    expr = re.sub(r'\barc(sin|cos|tan)\b', r'a\1', expr)
    return expr.strip()


def _parse(expr_str: str):
    """
    Parse a string into a SymPy expression.
    Uses implicit multiplication so "x sin(x)" → x*sin(x).
    Raises ValueError on failure.
    """
    from sympy.parsing.sympy_parser import (
        parse_expr,
        standard_transformations,
        implicit_multiplication_application,
        convert_xor,
    )
    from sympy import symbols
    from sympy import (
        sin, cos, tan, asin, acos, atan, sinh, cosh, tanh,
        exp, log, sqrt, pi, E, oo, I, Abs,
        sec, csc, cot, atan2, factorial, binomial,
        ceiling, floor, sign, Heaviside,
    )

    transformations = standard_transformations + (
        implicit_multiplication_application,
        convert_xor,
    )

    local_dict = {v: symbols(v) for v in "xyztnkabcmnpqrs"}
    local_dict.update({
        "sin": sin, "cos": cos, "tan": tan,
        "asin": asin, "acos": acos, "atan": atan,
        "arcsin": asin, "arccos": acos, "arctan": atan,
        "sinh": sinh, "cosh": cosh, "tanh": tanh,
        "exp": exp, "log": log, "ln": log,
        "sqrt": sqrt, "pi": pi, "e": E, "E": E,
        "oo": oo, "inf": oo, "infinity": oo,
        "I": I, "j": I, "abs": Abs, "Abs": Abs,
        "sec": sec, "csc": csc, "cot": cot, "atan2": atan2,
        "factorial": factorial, "binomial": binomial,
        "ceil": ceiling, "floor": floor, "sign": sign,
        "Heaviside": Heaviside, "H": Heaviside,
    })

    cleaned = _preprocess(expr_str)
    try:
        return parse_expr(cleaned, local_dict=local_dict,
                          transformations=transformations,
                          evaluate=True)
    except Exception as exc:
        raise ValueError(f"Cannot parse '{expr_str}': {exc}")


def _extract_variable(text: str, default: str = "x") -> str:
    """Detect the primary variable from phrases like 'with respect to y'."""
    m = re.search(r'with\s+respect\s+to\s+([a-zA-Z])', text, re.I)
    if m:
        return m.group(1)
    m = re.search(r'\bwrt\s+([a-zA-Z])', text, re.I)
    if m:
        return m.group(1)
    m = re.search(r'\bd/d([a-zA-Z])', text, re.I)
    if m:
        return m.group(1)
    return default


_NL_NOISE = re.compile(
    r'^(?:of|the|a|an|for|function|expression|expr|value|result)\s+',
    re.IGNORECASE,
)

def _strip_nl_noise(text: str) -> str:
    """Repeatedly strip leading English noise words that SymPy would misparse as variables."""
    prev = None
    while prev != text:
        prev = text
        text = _NL_NOISE.sub('', text).strip()
    return text


def _strip_prefix(text: str, keywords: list[str]) -> str:
    """Remove any matching operation prefix from the text, then strip NL noise words."""
    lowered = text.lower()
    for kw in sorted(keywords, key=len, reverse=True):
        if lowered.startswith(kw):
            return _strip_nl_noise(text[len(kw):].strip())
    for kw in sorted(keywords, key=len, reverse=True):
        idx = lowered.find(kw)
        if idx != -1:
            return _strip_nl_noise(text[idx + len(kw):].strip())
    return _strip_nl_noise(text.strip())


def _parse_matrix(text: str):
    """Extract and parse a matrix from text like [[1,2],[3,4]]."""
    from sympy import Matrix
    m = re.search(r'\[\[.*?\]\]', text, re.DOTALL)
    if not m:
        raise ValueError(
            "Please provide the matrix in format [[a,b],[c,d]] — e.g. [[1,2],[3,4]]"
        )
    mat_raw = m.group(0)
    mat_data = eval(mat_raw)
    return Matrix(mat_data)


# ─────────────────────────────────────────────────────────────────────────────
# Operation handlers
# ─────────────────────────────────────────────────────────────────────────────

def _handle_integrate(text: str) -> Tuple[str, str]:
    from sympy import integrate, symbols, latex
    import sympy as _sp

    var_name = _extract_variable(text)
    var = symbols(var_name)
    expr_text = _strip_prefix(text, _ADVANCED_OPS["integrate"])
    # Remove "with respect to X" from expression text
    expr_text = re.sub(r'\s+with\s+respect\s+to\s+[a-zA-Z]\s*$', '', expr_text, flags=re.I).strip()
    expr_text = re.sub(r'\bwrt\s+[a-zA-Z]\s*$', '', expr_text, flags=re.I).strip()

    def _parse_bound(raw: str):
        raw = raw.strip().rstrip(".,;:!?")
        raw = raw.replace("infty", "oo").replace("infinity", "oo")
        if raw == "oo":  return _sp.oo
        if raw == "-oo": return -_sp.oo
        return _parse(raw)

    # Format A: "EXPR from A to B"  (standard)
    m = re.search(
        r'(.*?)\s+from\s+([\w\.\-\+eEpioo]+)\s+to\s+([\w\.\-\+eEpioo]+)',
        expr_text, re.I
    )

    # Format B: "from A to B of EXPR"  (reversed — users often write it this way)
    if not m:
        m2 = re.search(
            r'^from\s+([\w\.\-\+eEpioo]+)\s+to\s+([\w\.\-\+eEpioo]+)\s+(?:of\s+)?(.*)',
            expr_text, re.I
        )
        if m2:
            lower  = _parse_bound(m2.group(1))
            upper  = _parse_bound(m2.group(2))
            expr   = _parse(m2.group(3).strip())
            result = integrate(expr, (var, lower, upper))
            return (
                f"∫ ({expr}) d{var_name} from {lower} to {upper} = {result}",
                latex(result),
            )

    if m:
        expr   = _parse(m.group(1).strip())
        lower  = _parse_bound(m.group(2).strip())
        upper  = _parse_bound(m.group(3).strip())
        result = integrate(expr, (var, lower, upper))
        return (
            f"∫ ({expr}) d{var_name} from {lower} to {upper} = {result}",
            latex(result),
        )
    else:
        expr   = _parse(expr_text)
        result = integrate(expr, var)
        return (
            f"∫ ({expr}) d{var_name} = {result} + C",
            latex(result) + " + C",
        )


def _handle_differentiate(text: str) -> Tuple[str, str]:
    from sympy import diff, symbols, latex

    var_name = _extract_variable(text)
    var = symbols(var_name)

    _ORDINAL_MAP = {
        "second": 2, "2nd": 2, "third": 3, "3rd": 3,
        "fourth": 4, "4th": 4, "fifth": 5, "5th": 5,
        "sixth": 6, "6th": 6, "seventh": 7, "7th": 7,
        "eighth": 8, "8th": 8, "ninth": 9, "9th": 9,
    }
    order = 1
    m_order = re.search(
        r'\b(second|2nd|third|3rd|fourth|4th|fifth|5th|sixth|6th|'
        r'seventh|7th|eighth|8th|ninth|9th)\s+derivative\b',
        text, re.I
    )
    if m_order:
        order = _ORDINAL_MAP[m_order.group(1).lower()]

    expr_text = text
    expr_text = re.sub(
        r'(?:second|2nd|third|3rd|fourth|4th|fifth|5th|sixth|6th|'
        r'seventh|7th|eighth|8th|ninth|9th)?\s*(?:partial\s+)?derivative\s+of\s+',
        '', expr_text, flags=re.I
    ).strip()
    expr_text = _strip_prefix(expr_text, _ADVANCED_OPS["differentiate"])
    expr_text = re.sub(r'^of\s+', '', expr_text, flags=re.I).strip()
    expr_text = re.sub(r'\s+with\s+respect\s+to\s+[a-zA-Z]\s*$', '', expr_text, flags=re.I).strip()
    expr_text = re.sub(r'\bwrt\s+[a-zA-Z]\s*$', '', expr_text, flags=re.I).strip()

    expr   = _parse(expr_text)
    result = diff(expr, var, order)
    order_label = {1: "d/d", 2: "d²/d", 3: "d³/d"}.get(order, f"d^{order}/d")
    return (
        f"{order_label}{var_name}[{expr}] = {result}",
        latex(result),
    )


def _handle_limit(text: str) -> Tuple[str, str]:
    from sympy import limit, symbols, latex, oo

    var_name = _extract_variable(text, default="x")
    var = symbols(var_name)

    m = re.search(
        r'(?:limit\s+of\s+|lim\s+)?(.+?)\s+as\s+'
        rf'{var_name}\s+(?:->|→|approaches|tends\s+to)\s+([^\s,]+)',
        text, re.I
    )

    if m:
        expr_raw  = m.group(1).strip()
        point_raw = m.group(2).strip()
    else:
        m2 = re.search(
            rf'lim\s+{var_name}\s*[-→>]{{1,2}}\s*([^\s]+)\s+(.+)', text, re.I
        )
        if m2:
            point_raw = m2.group(1)
            expr_raw  = m2.group(2)
        else:
            raise ValueError(
                "Could not parse limit. Expected: 'limit of EXPR as x approaches VALUE'"
            )

    point_raw = (point_raw.replace("infinity", "oo")
                          .replace("∞", "oo")
                          .replace("infty", "oo"))
    import sympy
    if point_raw == "oo":   point = oo
    elif point_raw == "-oo": point = -oo
    else: point = _parse(point_raw)

    expr   = _parse(expr_raw)
    result = limit(expr, var, point)
    return (
        f"lim({expr}) as {var_name} → {point} = {result}",
        sympy.latex(result),
    )


def _handle_solve(text: str) -> Tuple[str, str]:
    from sympy import solve, symbols, Eq, latex

    # ── System of equations detection ─────────────────────────────────────────
    # Match patterns like "2x+3y=7, x-y=1" or "system: ..."
    _SYS_PATTERN = re.compile(
        r'(?:system\s+of\s+equations?[:\s]+|equations?[:\s]+)?'
        r'(-?[\d\w\s\+\-\*\/\^\(\)\.]+=[^,]+)'
        r'(?:\s*,\s*(-?[\d\w\s\+\-\*\/\^\(\)\.]+=[^,]+))+'
    )
    # Extract all "LHS=RHS" pairs from the text
    raw_eqs = re.findall(r'(-?[\d\w\s\+\-\*\/\^\(\)\.]+=[^,\n]+)', text)
    # Filter to those that contain a variable (not just "system of equations:")
    eq_candidates = [e.strip() for e in raw_eqs if re.search(r'[a-zA-Z]', e)
                     and not re.match(r'^\s*(?:system|equation)', e, re.I)]

    if len(eq_candidates) >= 2:
        # Detect all variables used
        all_vars_found = sorted(set(re.findall(r'\b([a-zA-Z])\b', ' '.join(eq_candidates))))
        # Exclude common noise words treated as vars by SymPy
        skip = {'e', 'i', 'j', 'k', 'n', 'o', 's', 'x', 'y', 'z', 't', 'a', 'b', 'c'}
        # Keep only plausible unknowns (those that appear in equations)
        var_syms = symbols(' '.join(all_vars_found))
        if not isinstance(var_syms, (list, tuple)):
            var_syms = [var_syms]
        eqs_sympy = []
        for raw in eq_candidates:
            parts = raw.split('=', 1)
            try:
                lhs = _parse(parts[0].strip())
                rhs = _parse(parts[1].strip()) if len(parts) > 1 else _parse('0')
                eqs_sympy.append(Eq(lhs, rhs))
            except Exception:
                continue
        if len(eqs_sympy) >= 2:
            solutions = solve(eqs_sympy, var_syms)
            if solutions:
                if isinstance(solutions, dict):
                    sol_str = ", ".join(f"{k}={v}" for k, v in solutions.items())
                elif isinstance(solutions, list) and solutions and isinstance(solutions[0], dict):
                    sol_str = "; ".join(
                        ", ".join(f"{k}={v}" for k, v in sol.items()) for sol in solutions
                    )
                else:
                    sol_str = str(solutions)
                return (f"Solution: {sol_str}", sol_str)
            return ("No solution found for the system.", r"\text{No solution}")

    # ── Single equation ────────────────────────────────────────────────────────
    var_name = _extract_variable(text)
    var = symbols(var_name)

    expr_text = _strip_prefix(text, _ADVANCED_OPS["solve"])
    expr_text = re.sub(r'\s+for\s+[a-zA-Z]$', '', expr_text.strip(), flags=re.I)

    if '=' in expr_text:
        parts = expr_text.split('=', 1)
        lhs = _parse(parts[0].strip())
        rhs = _parse(parts[1].strip())
        solutions = solve(Eq(lhs, rhs), var)
    else:
        solutions = solve(_parse(expr_text), var)

    if not solutions:
        return (f"No solutions found for: {expr_text}", r"\text{No solution}")

    sol_str   = ", ".join(str(s) for s in solutions)
    sol_latex = ", ".join(latex(s) for s in solutions)
    return (f"{var_name} = {sol_str}", sol_latex)


def _handle_ode(text: str) -> Tuple[str, str]:
    """Solve ordinary differential equations using SymPy's dsolve."""
    from sympy import symbols, Function, dsolve, latex, Eq, Derivative
    from sympy.parsing.sympy_parser import parse_expr

    x = symbols('x')
    y = Function('y')

    # Normalise ^ to **
    text_norm = text.replace('^', '**')

    # Try to extract the ODE expression:
    # Support patterns like:
    #   "y'' + y = 0", "y' - 2y = 0", "dy/dx + y = x"
    # We'll try to build the ODE equation

    # Replace y'' → Derivative(y(x), x, 2), y' → Derivative(y(x), x)
    # and y → y(x) in the expression
    cleaned = text_norm
    # Strip any leading prompt words
    cleaned = re.sub(
        r'(?:solve|ode|ordinary differential equation|differential equation|solve the ode|solve ode)[\s:]*',
        '', cleaned, flags=re.I
    ).strip()

    # Replace notation
    cleaned = re.sub(r"y''", "Derivative(y(x),x,2)", cleaned)
    cleaned = re.sub(r"y'",  "Derivative(y(x),x)",   cleaned)
    # dy/dx or d^2y/dx^2
    cleaned = re.sub(r'd\*\*2y/dx\*\*2', 'Derivative(y(x),x,2)', cleaned)
    cleaned = re.sub(r'd2y/dx2',          'Derivative(y(x),x,2)', cleaned)
    cleaned = re.sub(r'dy/dx',            'Derivative(y(x),x)',   cleaned)
    # bare y that isn't followed by ( — replace with y(x)
    cleaned = re.sub(r'\by\b(?!\()', 'y(x)', cleaned)

    local_dict = {
        'x': x, 'y': y, 'Derivative': Derivative,
    }
    from sympy import sin, cos, exp, log, sqrt, pi, E, oo, tan
    local_dict.update({
        'sin': sin, 'cos': cos, 'exp': exp, 'log': log,
        'sqrt': sqrt, 'pi': pi, 'e': E, 'tan': tan,
    })

    try:
        if '=' in cleaned:
            lhs_str, rhs_str = cleaned.split('=', 1)
            lhs = parse_expr(lhs_str.strip(), local_dict=local_dict)
            rhs = parse_expr(rhs_str.strip(), local_dict=local_dict)
            ode_eq = Eq(lhs, rhs)
        else:
            expr = parse_expr(cleaned.strip(), local_dict=local_dict)
            ode_eq = Eq(expr, 0)

        sol = dsolve(ode_eq, y(x))
        return (
            f"ODE: {ode_eq}\nGeneral solution: {sol}",
            latex(sol),
        )
    except Exception as exc:
        raise ValueError(f"Could not solve ODE: {exc}")


def _handle_eigenvalue(text: str) -> Tuple[str, str]:
    from sympy import latex

    mat = _parse_matrix(text)
    eigs  = mat.eigenvals()
    evecs = mat.eigenvects()

    eig_str = "; ".join(
        f"λ={ev} (multiplicity {mult})" for ev, mult in eigs.items()
    )
    evec_parts = []
    for ev, mult, vecs in evecs:
        for v in vecs:
            evec_parts.append(f"λ={ev}: {v.T.tolist()}")
    evec_str = "; ".join(evec_parts)

    result_str = f"Eigenvalues: {eig_str}\nEigenvectors: {evec_str}"
    return (result_str, eig_str)


def _handle_determinant(text: str) -> Tuple[str, str]:
    from sympy import latex

    mat = _parse_matrix(text)
    det = mat.det()
    return (f"det = {det}", latex(det))


def _handle_inverse(text: str) -> Tuple[str, str]:
    from sympy import latex

    mat = _parse_matrix(text)
    inv = mat.inv()
    return (f"Inverse matrix:\n{inv}", latex(inv))


def _handle_matrix_rank(text: str) -> Tuple[str, str]:
    mat = _parse_matrix(text)
    rank = mat.rank()
    return (f"Rank = {rank}", str(rank))


def _handle_matrix_trace(text: str) -> Tuple[str, str]:
    from sympy import latex
    mat = _parse_matrix(text)
    trace = mat.trace()
    return (f"Trace = {trace}", latex(trace))


def _handle_series(text: str) -> Tuple[str, str]:
    from sympy import series, symbols, latex, oo

    var_name = _extract_variable(text)
    var = symbols(var_name)
    expr_text = _strip_prefix(text, _ADVANCED_OPS["series"])
    # Strip leading "of" left after prefix removal
    expr_text = re.sub(r'^of\s+', '', expr_text, flags=re.I).strip()

    point = 0
    m_point = re.search(r'(?:around|at|about|near)\s+([\w\.\-\+]+)', expr_text, re.I)
    if m_point:
        raw = m_point.group(1).replace("infinity", "oo").replace("∞", "oo")
        point = oo if raw == "oo" else _parse(raw)
        expr_text = expr_text[:m_point.start()].strip()

    order = 6
    m_order = re.search(r'(?:order|degree|up\s+to|terms?)\s+(\d+)', expr_text, re.I)
    if m_order:
        order = int(m_order.group(1))
        expr_text = (expr_text[:m_order.start()] + expr_text[m_order.end():]).strip()

    expr   = _parse(expr_text)
    result = series(expr, var, point, n=order)
    return (
        f"Series of {expr} around {var_name}={point} (order {order}): {result}",
        latex(result),
    )


def _handle_laplace(text: str) -> Tuple[str, str]:
    from sympy import symbols, laplace_transform, latex

    t, s = symbols('t s', positive=True)
    expr_text = _strip_prefix(text, _ADVANCED_OPS["laplace"])
    expr_text = re.sub(r'\bof\b', '', expr_text, flags=re.I).strip()

    expr = _parse(expr_text)
    # With noconds=True SymPy returns the expression directly (not a tuple)
    raw = laplace_transform(expr, t, s, noconds=True)
    # Guard: some SymPy versions return a 3-tuple even with noconds=True
    if isinstance(raw, tuple):
        result = raw[0]
    else:
        result = raw
    return (
        f"L{{{expr}}} = {result}",
        latex(result),
    )


def _handle_inverse_laplace(text: str) -> Tuple[str, str]:
    from sympy import symbols, inverse_laplace_transform, latex, Symbol

    # SymPy requires s to be declared positive for inverse Laplace
    t_pos, s_pos = symbols('t s', positive=True)
    expr_text = _strip_prefix(text, _ADVANCED_OPS["inverse_laplace"])
    expr_text = re.sub(r'\bof\b', '', expr_text, flags=re.I).strip()
    expr = _parse(expr_text)
    # Substitute any plain 's' or 't' with the positive versions
    s_plain = Symbol('s')
    t_plain = Symbol('t')
    expr = expr.subs([(s_plain, s_pos), (t_plain, t_pos)])
    result = inverse_laplace_transform(expr, s_pos, t_pos)
    return (
        f"L⁻¹{{{expr}}} = {result}",
        latex(result),
    )


def _handle_fourier(text: str) -> Tuple[str, str]:
    from sympy import symbols, fourier_transform, latex

    x, k = symbols('x k')
    expr_text = _strip_prefix(text, _ADVANCED_OPS["fourier"])
    expr_text = re.sub(r'\bof\b', '', expr_text, flags=re.I).strip()
    expr = _parse(expr_text)
    result = fourier_transform(expr, x, k)
    return (
        f"F{{{expr}}} = {result}",
        latex(result),
    )


def _handle_simplify(text: str) -> Tuple[str, str]:
    from sympy import simplify, latex

    expr_text = _strip_prefix(text, _ADVANCED_OPS["simplify"])
    expr   = _parse(expr_text)
    result = simplify(expr)
    return (f"Simplified: {result}", latex(result))


def _handle_trig_simplify(text: str) -> Tuple[str, str]:
    from sympy import trigsimp, latex

    # strip any trig-specific prefix then fall through
    expr_text = re.sub(
        r'simplif[y]?\s+(?:the\s+)?trigonometric\s+|trig\s+simplif[y]?\s+|simplif[y]?\s+trig\s+',
        '', text, flags=re.I
    ).strip()
    expr   = _parse(expr_text)
    result = trigsimp(expr)
    return (f"Trig-simplified: {result}", latex(result))


def _handle_factor(text: str) -> Tuple[str, str]:
    from sympy import factor, latex

    expr_text = _strip_prefix(text, _ADVANCED_OPS["factor"])
    expr   = _parse(expr_text)
    result = factor(expr)
    return (f"Factored: {result}", latex(result))


def _handle_expand(text: str) -> Tuple[str, str]:
    from sympy import expand, latex

    expr_text = _strip_prefix(text, _ADVANCED_OPS["expand"])
    # Strip trailing natural-language qualifiers ("using binomial theorem", "by hand", etc.)
    expr_text = re.sub(
        r'\s+(?:using|by|via|with)\s+.*$', '', expr_text, flags=re.I
    ).strip()
    expr   = _parse(expr_text)
    result = expand(expr)
    return (f"Expanded: {result}", latex(result))


def _handle_partial_fraction(text: str) -> Tuple[str, str]:
    from sympy import apart, symbols, latex

    var_name = _extract_variable(text)
    var = symbols(var_name)
    expr_text = _strip_prefix(text, _ADVANCED_OPS["partial_fraction"])
    expr   = _parse(expr_text)
    result = apart(expr, var)
    return (f"Partial fractions of {expr}: {result}", latex(result))


def _handle_gcd(text: str) -> Tuple[str, str]:
    from sympy import gcd, latex

    # Extract numbers from text
    numbers = re.findall(r'\d+', text)
    if len(numbers) < 2:
        raise ValueError("Please provide at least two numbers. Example: GCD of 48 and 18")
    from sympy import Integer
    result = Integer(numbers[0])
    for n in numbers[1:]:
        result = gcd(result, Integer(n))
    nums_str = ", ".join(numbers)
    return (f"GCD({nums_str}) = {result}", latex(result))


def _handle_lcm(text: str) -> Tuple[str, str]:
    from sympy import lcm, latex

    numbers = re.findall(r'\d+', text)
    if len(numbers) < 2:
        raise ValueError("Please provide at least two numbers. Example: LCM of 12 and 18")
    from sympy import Integer
    result = Integer(numbers[0])
    for n in numbers[1:]:
        result = lcm(result, Integer(n))
    nums_str = ", ".join(numbers)
    return (f"LCM({nums_str}) = {result}", latex(result))


def _handle_prime_factors(text: str) -> Tuple[str, str]:
    from sympy import factorint, latex

    numbers = re.findall(r'\d+', text)
    if not numbers:
        raise ValueError("Please provide a number. Example: prime factorization of 360")
    n = int(numbers[0])
    factors = factorint(n)
    factor_str = " × ".join(
        f"{p}^{e}" if e > 1 else str(p) for p, e in sorted(factors.items())
    )
    return (f"{n} = {factor_str}", factor_str)


def _handle_modular(text: str) -> Tuple[str, str]:
    from sympy import mod_inverse, Integer

    # modular inverse: "modular inverse of A mod M"
    m_inv = re.search(
        r'modular\s+inverse\s+of\s+(\d+)\s+mod\s+(\d+)', text, re.I
    )
    if m_inv:
        a, m_val = int(m_inv.group(1)), int(m_inv.group(2))
        inv = mod_inverse(a, m_val)
        return (f"Modular inverse of {a} mod {m_val} = {inv}", str(inv))

    # plain modulo: "A mod B"
    m_mod = re.search(r'(\d+)\s+mod(?:ulo)?\s+(\d+)', text, re.I)
    if m_mod:
        a, m_val = int(m_mod.group(1)), int(m_mod.group(2))
        result = a % m_val
        return (f"{a} mod {m_val} = {result}", str(result))

    raise ValueError(
        "Could not parse modular arithmetic. "
        "Try: '17 mod 5' or 'modular inverse of 3 mod 7'"
    )


def _handle_statistics(text: str) -> Tuple[str, str]:
    from sympy.stats import Normal
    from sympy import Rational, latex

    # Extract list of numbers from text
    numbers = re.findall(r'-?\d+(?:\.\d+)?', text)
    if not numbers:
        raise ValueError(
            "Please provide a list of numbers. Example: mean of 2, 4, 6, 8"
        )
    vals = [float(n) for n in numbers]
    n = len(vals)
    mean   = sum(vals) / n
    sorted_vals = sorted(vals)
    if n % 2 == 0:
        median = (sorted_vals[n//2 - 1] + sorted_vals[n//2]) / 2
    else:
        median = sorted_vals[n//2]
    variance = sum((v - mean) ** 2 for v in vals) / n
    std_dev  = variance ** 0.5

    result_str = (
        f"Data: {vals}\n"
        f"Mean = {mean:.6g}\n"
        f"Median = {median:.6g}\n"
        f"Variance = {variance:.6g}\n"
        f"Std Dev = {std_dev:.6g}"
    )
    return (result_str, result_str.replace("\n", r" \\ "))


def _handle_factorial(text: str) -> Tuple[str, str]:
    from sympy import factorial, latex, Integer

    numbers = re.findall(r'\d+', text)
    if not numbers:
        raise ValueError("Please provide a number. Example: factorial of 10")
    n = int(numbers[0])
    if n > 1000:
        raise ValueError("Number too large for factorial (max 1000)")
    result = factorial(Integer(n))
    return (f"{n}! = {result}", latex(result))


def _handle_binomial(text: str) -> Tuple[str, str]:
    from sympy import binomial as sym_binomial, latex, Integer

    # Try explicit nCr notation first: "10C3", "10c3", "C(10,3)"
    m_ncr = _NCR_PATTERN.search(text)
    if m_ncr:
        n, r = int(m_ncr.group(1)), int(m_ncr.group(2))
        result = sym_binomial(Integer(n), Integer(r))
        return (f"C({n}, {r}) = {result}", latex(result))

    numbers = re.findall(r'\d+', text)
    if len(numbers) < 2:
        raise ValueError("Please provide n and r. Example: binomial coefficient 10 choose 3")
    n, r = int(numbers[0]), int(numbers[1])
    result = sym_binomial(Integer(n), Integer(r))
    return (f"C({n}, {r}) = {result}", latex(result))


def _handle_permutation(text: str) -> Tuple[str, str]:
    from sympy import factorial, latex, Integer

    numbers = re.findall(r'\d+', text)
    if len(numbers) < 2:
        raise ValueError("Please provide n and r. Example: permutation 10 P 3")
    n, r = int(numbers[0]), int(numbers[1])
    result = factorial(Integer(n)) // factorial(Integer(n - r))
    return (f"P({n}, {r}) = {result}", latex(result))


def _handle_summation(text: str) -> Tuple[str, str]:
    from sympy import summation, symbols, oo, latex

    def _parse_bound(raw: str):
        raw = raw.strip().rstrip(".,;:!?")
        raw = raw.replace("infinity", "oo").replace("infty", "oo")
        if raw == "oo":  return oo
        if raw == "-oo": return -oo
        return _parse(raw)

    # ── Detect variable from several natural-language patterns ────────────────
    # "for X=" / "for X from"  or  "from X="  or  "n=A to B"
    m_var = (
        re.search(r'\bfor\s+([a-zA-Z])\s*(?:=|from)\b',   text, re.I) or
        re.search(r'\bfrom\s+([a-zA-Z])\s*=',              text, re.I) or
        re.search(r'\b([a-zA-Z])\s*=\s*\d+\s+to\s+\d+',   text, re.I)
    )
    var_name = m_var.group(1) if m_var else _extract_variable(text, default="k")
    var = symbols(var_name)

    expr_text = _strip_prefix(text, _ADVANCED_OPS["summation"])
    expr_text = re.sub(r'^of\s+', '', expr_text, flags=re.I).strip()

    # Pattern A: "EXPR for k=A to B"  /  "EXPR for k from A to B"
    m = re.search(
        rf'(.*?)\s+for\s+{var_name}\s*(?:=|from)\s*(-?[\w\.]+)\s+to\s+(-?[\w\.]+)',
        expr_text, re.I
    )
    # Pattern B: "EXPR from k=A to B"
    if not m:
        m = re.search(
            rf'(.*?)\s+from\s+{var_name}\s*=\s*(-?[\w\.]+)\s+to\s+(-?[\w\.]+)',
            expr_text, re.I
        )
    # Pattern C: "k^2 from k=A to B"  (variable already consumed by strip)
    if not m:
        m = re.search(
            r'^(.+?)\s+from\s+([a-zA-Z])\s*=\s*(-?[\w\.]+)\s+to\s+(-?[\w\.]+)',
            expr_text, re.I
        )
        if m:
            # Reinterpret groups: expr, var, lo, hi
            expr_raw2, var_name2, lo_raw, hi_raw = (
                m.group(1), m.group(2), m.group(3), m.group(4)
            )
            var2 = symbols(var_name2)
            lo   = _parse_bound(lo_raw)
            hi   = _parse_bound(hi_raw)
            expr = _parse(expr_raw2)
            result = summation(expr, (var2, lo, hi))
            return (
                f"Σ({expr}, {var_name2}={lo}..{hi}) = {result}",
                latex(result),
            )

    if m and len(m.groups()) == 3:
        expr_raw = m.group(1).strip()
        lo       = _parse_bound(m.group(2))
        hi       = _parse_bound(m.group(3))
        expr     = _parse(expr_raw)
        result   = summation(expr, (var, lo, hi))
        return (
            f"Σ({expr}, {var_name}={lo}..{hi}) = {result}",
            latex(result),
        )
    else:
        expr   = _parse(expr_text)
        result = summation(expr, (var, 0, oo))
        return (
            f"Σ({expr}, {var_name}=0..∞) = {result}",
            latex(result),
        )


def _handle_product(text: str) -> Tuple[str, str]:
    from sympy import Product, symbols, oo, latex

    var_name = _extract_variable(text, default="k")
    var = symbols(var_name)

    expr_text = _strip_prefix(text, _ADVANCED_OPS["product"])
    m = re.search(
        rf'(.*?)\s+(?:for|from)\s+{var_name}\s*=\s*(-?\w+)\s+to\s+(-?\w+)',
        expr_text, re.I
    )
    if m:
        expr_raw = m.group(1).strip()
        lo_raw   = m.group(2).replace("infty", "oo")
        hi_raw   = m.group(3).replace("infty", "oo")
        expr = _parse(expr_raw)
        lo   = oo if lo_raw == "oo" else _parse(lo_raw)
        hi   = oo if hi_raw == "oo" else _parse(hi_raw)
        result = Product(expr, (var, lo, hi)).doit()
        return (
            f"∏({expr}, {var_name}={lo}..{hi}) = {result}",
            latex(result),
        )
    else:
        expr = _parse(expr_text)
        result = Product(expr, (var, 1, oo)).doit()
        return (
            f"∏({expr}, {var_name}=1..∞) = {result}",
            latex(result),
        )


def _handle_complex_ops(text: str) -> Tuple[str, str]:
    from sympy import re as Re, im as Im, Abs, arg, conjugate, latex, symbols, I

    # Try to extract a complex expression
    # Strip common prefixes
    clean = re.sub(
        r'(?:real\s+part\s+of|imaginary\s+part\s+of|modulus\s+of|argument\s+of|conjugate\s+of|complex\s+number)\s*',
        '', text, flags=re.I
    ).strip()

    expr = _parse(clean)

    results = {
        "Real part":      Re(expr),
        "Imaginary part": Im(expr),
        "Modulus":        Abs(expr),
        "Argument":       arg(expr),
        "Conjugate":      conjugate(expr),
    }

    lines = [f"{k} = {v}" for k, v in results.items()]
    result_str  = "\n".join(lines)
    result_latex = r" \\ ".join(f"{k} = {latex(v)}" for k, v in results.items())
    return (result_str, result_latex)


# ─────────────────────────────────────────────────────────────────────────────
# Word problem solver
# ─────────────────────────────────────────────────────────────────────────────

def _handle_word_problem(text: str) -> Tuple[str, str]:  # noqa: C901
    """
    Deterministic solver for common math word problems.
    Covers: geometry, percentage, simple/compound interest,
    speed-distance-time, work-rate, ratio/proportion.
    """
    import math as _math
    t = text.lower()

    def _num(patterns):
        for pat in patterns:
            m = re.search(pat, t, re.I)
            if m:
                try:
                    return float(m.group(1).replace(',', ''))
                except ValueError:
                    pass
        return None

    # ── Geometry ──────────────────────────────────────────────────────────────

    # Rectangle area
    if re.search(r'area\s+of\s+(?:a\s+|the\s+)?rectangle', t):
        l = _num([r'length\s*(?:of\s*|=\s*|is\s*)?([\d.]+)', r'l\s*=\s*([\d.]+)', r'([\d.]+)\s*(?:m|cm|ft|units?)\s+(?:long|length)'])
        w = _num([r'width\s*(?:of\s*|=\s*|is\s*)?([\d.]+)', r'w(?:idth)?\s*=\s*([\d.]+)', r'([\d.]+)\s*(?:m|cm|ft|units?)\s+(?:wide|width)'])
        if l and w:
            area = l * w
            perim = 2 * (l + w)
            return (f"Rectangle: length={l}, width={w}\nArea = l×w = {l}×{w} = {area}\nPerimeter = 2(l+w) = 2×({l}+{w}) = {perim}", str(area))

    # Square
    if re.search(r'area\s+of\s+(?:a\s+|the\s+)?square', t):
        s = _num([r'side\s*(?:of\s*|=\s*|is\s*)?([\d.]+)', r's(?:ide)?\s*=\s*([\d.]+)', r'([\d.]+)\s*(?:m|cm|ft|units?)'])
        if s:
            area = s ** 2
            perim = 4 * s
            return (f"Square: side={s}\nArea = s² = {s}² = {area}\nPerimeter = 4s = 4×{s} = {perim}", str(area))

    # Circle
    if re.search(r'area\s+of\s+(?:a\s+|the\s+)?circle|circumference\s+of', t):
        r_ = _num([r'radius\s*(?:of\s*|=\s*|is\s*)?([\d.]+)', r'r(?:adius)?\s*=\s*([\d.]+)'])
        d_ = _num([r'diameter\s*(?:of\s*|=\s*|is\s*)?([\d.]+)', r'd(?:iameter)?\s*=\s*([\d.]+)'])
        if d_ and not r_:
            r_ = d_ / 2
        if r_:
            area = _math.pi * r_ ** 2
            circum = 2 * _math.pi * r_
            return (
                f"Circle: radius={r_}\n"
                f"Area = πr² = π×{r_}² = {area:.6g}\n"
                f"Circumference = 2πr = 2×π×{r_} = {circum:.6g}",
                f"{area:.6g}",
            )

    # Triangle area
    if re.search(r'area\s+of\s+(?:a\s+|the\s+)?triangle', t):
        b = _num([r'base\s*(?:of\s*|=\s*|is\s*)?([\d.]+)', r'b(?:ase)?\s*=\s*([\d.]+)'])
        h = _num([r'height\s*(?:of\s*|=\s*|is\s*)?([\d.]+)', r'h(?:eight)?\s*=\s*([\d.]+)'])
        # Heron's formula from three sides
        sides = re.findall(r'\b([\d.]+)\b', t)
        if b and h:
            area = 0.5 * b * h
            return (f"Triangle: base={b}, height={h}\nArea = ½bh = ½×{b}×{h} = {area}", str(area))

    # Cylinder
    if re.search(r'(?:volume|surface area)\s+of\s+(?:a\s+|the\s+)?cylinder', t):
        r_ = _num([r'radius\s*(?:of\s*|=\s*|is\s*)?([\d.]+)', r'r\s*=\s*([\d.]+)'])
        h_ = _num([r'height\s*(?:of\s*|=\s*|is\s*)?([\d.]+)', r'h\s*=\s*([\d.]+)'])
        if r_ and h_:
            vol = _math.pi * r_**2 * h_
            sa  = 2 * _math.pi * r_ * (r_ + h_)
            return (
                f"Cylinder: radius={r_}, height={h_}\n"
                f"Volume = πr²h = π×{r_}²×{h_} = {vol:.6g}\n"
                f"Surface area = 2πr(r+h) = 2π×{r_}×({r_}+{h_}) = {sa:.6g}",
                f"{vol:.6g}",
            )

    # Sphere
    if re.search(r'(?:volume|surface area)\s+of\s+(?:a\s+|the\s+)?sphere', t):
        r_ = _num([r'radius\s*(?:of\s*|=\s*|is\s*)?([\d.]+)', r'r\s*=\s*([\d.]+)'])
        d_ = _num([r'diameter\s*(?:of\s*|=\s*|is\s*)?([\d.]+)'])
        if d_ and not r_:
            r_ = d_ / 2
        if r_:
            vol = (4/3) * _math.pi * r_**3
            sa  = 4 * _math.pi * r_**2
            return (
                f"Sphere: radius={r_}\n"
                f"Volume = (4/3)πr³ = (4/3)×π×{r_}³ = {vol:.6g}\n"
                f"Surface area = 4πr² = 4×π×{r_}² = {sa:.6g}",
                f"{vol:.6g}",
            )

    # Cone
    if re.search(r'volume\s+of\s+(?:a\s+|the\s+)?cone', t):
        r_ = _num([r'radius\s*(?:of\s*|=\s*|is\s*)?([\d.]+)', r'r\s*=\s*([\d.]+)'])
        h_ = _num([r'height\s*(?:of\s*|=\s*|is\s*)?([\d.]+)', r'h\s*=\s*([\d.]+)'])
        if r_ and h_:
            vol = (1/3) * _math.pi * r_**2 * h_
            return (f"Cone: radius={r_}, height={h_}\nVolume = (1/3)πr²h = {vol:.6g}", f"{vol:.6g}")

    # Perimeter (generic)
    if re.search(r'perimeter\s+of\s+(?:a\s+|the\s+)?rectangle', t):
        l = _num([r'length\s*(?:of\s*|=\s*|is\s*)?([\d.]+)'])
        w = _num([r'width\s*(?:of\s*|=\s*|is\s*)?([\d.]+)'])
        if l and w:
            return (f"Perimeter = 2(l+w) = 2×({l}+{w}) = {2*(l+w)}", str(2*(l+w)))

    # ── Percentage ────────────────────────────────────────────────────────────

    pct_of = re.search(r'([\d.]+)\s*%\s+of\s+([\d,]+(?:\.\d+)?)', t)
    if pct_of:
        pct  = float(pct_of.group(1))
        base = float(pct_of.group(2).replace(',', ''))
        result = pct / 100 * base
        return (f"{pct}% of {base} = {pct}/100 × {base} = {result}", str(result))

    what_pct = re.search(r'what\s+(?:is\s+the\s+)?percent(?:age)?\s+(?:of\s+|is\s+)?([\d.]+)\s+(?:of\s+|out\s+of\s+|from\s+)([\d.]+)', t)
    if what_pct:
        part = float(what_pct.group(1))
        whole = float(what_pct.group(2))
        pct = (part / whole) * 100
        return (f"({part}/{whole}) × 100 = {pct:.4g}%", f"{pct:.4g}%")

    incr_pct = re.search(r'([\d.]+)\s+increased\s+by\s+([\d.]+)\s*%', t)
    if incr_pct:
        val = float(incr_pct.group(1)); pct = float(incr_pct.group(2))
        result = val * (1 + pct/100)
        return (f"{val} increased by {pct}% = {val} × {1+pct/100} = {result:.4g}", f"{result:.4g}")

    decr_pct = re.search(r'([\d.]+)\s+decreased\s+by\s+([\d.]+)\s*%', t)
    if decr_pct:
        val = float(decr_pct.group(1)); pct = float(decr_pct.group(2))
        result = val * (1 - pct/100)
        return (f"{val} decreased by {pct}% = {val} × {1-pct/100} = {result:.4g}", f"{result:.4g}")

    # ── Simple & compound interest ─────────────────────────────────────────────

    if re.search(r'simple\s+interest', t):
        P = _num([r'principal\s*(?:of\s*|=\s*|is\s*)?([\d,]+)', r'P\s*=\s*([\d,]+)', r'\$([\d,]+)', r'([\d,]+)\s+(?:rupees|dollars|pounds)'])
        r = _num([r'rate\s*(?:of\s*|=\s*|is\s*)?([\d.]+)\s*%', r'([\d.]+)\s*%\s+(?:per\s+(?:year|annum|annual))?', r'r\s*=\s*([\d.]+)'])
        T = _num([r'(?:for\s+|time\s+=?\s*)([\d.]+)\s*years?', r'T\s*=\s*([\d.]+)', r'n\s*=\s*([\d.]+)'])
        if P and r and T:
            P = float(str(P).replace(',', ''))
            SI = P * (r/100) * T
            A  = P + SI
            return (
                f"Simple Interest: P={P}, r={r}%, T={T} years\n"
                f"SI = P × r/100 × T = {P} × {r}/100 × {T} = {SI:.4g}\n"
                f"Amount = P + SI = {P} + {SI:.4g} = {A:.4g}",
                f"SI={SI:.4g}, A={A:.4g}",
            )

    if re.search(r'compound\s+interest', t):
        P = _num([r'principal\s*(?:of\s*|=\s*|is\s*)?([\d,]+)', r'P\s*=\s*([\d,]+)',
                  r'(?:rs\.?|inr|₹)\s*([\d,]+)', r'\$([\d,]+)',
                  r'([\d,]+)\s+(?:rupees?|dollars?|pounds?)'])
        r = _num([r'rate\s*(?:of\s*|=\s*|is\s*)?([\d.]+)\s*%', r'([\d.]+)\s*%', r'r\s*=\s*([\d.]+)'])
        T = _num([r'(?:for\s+|after\s+|time\s+=?\s*)([\d.]+)\s*years?',
                  r'T\s*=\s*([\d.]+)', r'n\s*=\s*([\d.]+)'])
        n_comp = 1.0  # compounding frequency (annual by default)
        if re.search(r'semi.?annually|half.?yearly', t):   n_comp = 2
        if re.search(r'quarterly', t):                     n_comp = 4
        if re.search(r'monthly', t):                       n_comp = 12
        if re.search(r'daily', t):                         n_comp = 365
        if P and r and T:
            P = float(str(P).replace(',', ''))
            A   = P * (1 + (r/100)/n_comp) ** (n_comp * T)
            CI  = A - P
            freq_str = {1:'annually',2:'semi-annually',4:'quarterly',12:'monthly',365:'daily'}.get(int(n_comp),'')
            return (
                f"Compound Interest {freq_str}: P={P}, r={r}%, T={T} yr, n={int(n_comp)}\n"
                f"A = P(1 + r/n)^(nT) = {P}×(1 + {r/100}/{int(n_comp)})^({int(n_comp)}×{T}) = {A:.4g}\n"
                f"CI = A − P = {A:.4g} − {P} = {CI:.4g}",
                f"A={A:.4g}, CI={CI:.4g}",
            )

    # ── Speed-distance-time ────────────────────────────────────────────────────

    speed_m = re.search(r'(?:speed|velocity)\s+(?:of\s+|=\s*)?([\d.]+)\s*(?:km/?h|kmph|mph|miles?\s+per\s+hour|km\s+per\s+hour)', t, re.I)
    time_m  = re.search(r'(?:for|in|time\s+of|takes?|time=)\s+([\d.]+)\s*(?:hours?|hrs?|minutes?|mins?)', t, re.I)
    dist_m  = re.search(r'(?:distance|travel[ls]?|covers?|goes?)\s+(?:of\s+|=\s*)?([\d.]+)\s*(?:km|miles?|m)\b', t, re.I)

    if speed_m and time_m and not dist_m:
        spd = float(speed_m.group(1)); tim = float(time_m.group(1))
        d = spd * tim
        return (f"Distance = speed × time = {spd} × {tim} = {d:.4g}", f"{d:.4g}")

    if speed_m and dist_m and not time_m:
        spd = float(speed_m.group(1)); dis = float(dist_m.group(1))
        tim = dis / spd
        return (f"Time = distance / speed = {dis} / {spd} = {tim:.4g} hours", f"{tim:.4g} hours")

    if time_m and dist_m and not speed_m:
        tim = float(time_m.group(1)); dis = float(dist_m.group(1))
        spd = dis / tim
        return (f"Speed = distance / time = {dis} / {tim} = {spd:.4g}", f"{spd:.4g}")

    # Average speed (two legs)
    avg_spd = re.search(r'average\s+speed', t)
    if avg_spd:
        speeds = re.findall(r'([\d.]+)\s*(?:km/?h|kmph|mph|m/s)?', t)
        times_ = re.findall(r'([\d.]+)\s*(?:hours?|hrs?)', t)
        if len(speeds) >= 2:
            # Average speed = total distance / total time
            s1, s2 = float(speeds[0]), float(speeds[1])
            if len(times_) >= 2:
                t1, t2 = float(times_[0]), float(times_[1])
                d1, d2 = s1*t1, s2*t2
                avg = (d1+d2)/(t1+t2)
                return (f"Average speed = total distance / total time = ({d1}+{d2})/({t1}+{t2}) = {avg:.4g}", f"{avg:.4g}")
            else:
                # Harmonic mean for equal distances
                hmean = 2*s1*s2/(s1+s2)
                return (f"Average speed (equal distances) = 2s₁s₂/(s₁+s₂) = 2×{s1}×{s2}/({s1}+{s2}) = {hmean:.4g}", f"{hmean:.4g}")

    # ── Work-rate ─────────────────────────────────────────────────────────────

    work_together = re.search(r'(?:work\s+together|together)', t)
    work_rates = re.findall(r'([\d.]+)\s+days?', t)
    if work_together and len(work_rates) >= 2:
        days = [float(d) for d in work_rates[:2]]
        combined = 1 / sum(1/d for d in days)
        return (
            f"Work rates: A does 1/{days[0]} per day, B does 1/{days[1]} per day\n"
            f"Together: 1/{days[0]} + 1/{days[1]} = {1/days[0]:.6g} + {1/days[1]:.6g} = {sum(1/d for d in days):.6g} per day\n"
            f"Days to finish together = {combined:.4g} days",
            f"{combined:.4g} days",
        )

    # ── Ratio / Proportion ────────────────────────────────────────────────────

    ratio_m = re.search(r'(\d+)\s*:\s*(\d+)\s*=\s*(\d+)\s*:\s*x|x\s*:\s*(\d+)\s*=\s*(\d+)\s*:\s*(\d+)', t)
    if ratio_m:
        if ratio_m.group(1):
            a, b, c = int(ratio_m.group(1)), int(ratio_m.group(2)), int(ratio_m.group(3))
            x = b * c / a
            return (f"Proportion: {a}:{b} = {c}:x\nx = (b×c)/a = ({b}×{c})/{a} = {x:.4g}", f"{x:.4g}")
        else:
            x_denom, ratio_a, ratio_b = int(ratio_m.group(4)), int(ratio_m.group(5)), int(ratio_m.group(6))
            x = ratio_a * x_denom / ratio_b
            return (f"Proportion: x:{x_denom} = {ratio_a}:{ratio_b}\nx = {x:.4g}", f"{x:.4g}")

    # Generic ratio split
    ratio_split = re.search(r'(?:in\s+the\s+ratio|ratio\s+of)\s+(\d+)\s*:\s*(\d+)', t)
    total_val   = re.search(r'total\s+(?:of\s+|=\s*)?([\d,]+)|is\s+([\d,]+)', t)
    if ratio_split and total_val:
        a_r, b_r = int(ratio_split.group(1)), int(ratio_split.group(2))
        tot_str  = (total_val.group(1) or total_val.group(2) or '').replace(',','')
        if tot_str:
            tot = float(tot_str)
            share_a = tot * a_r / (a_r + b_r)
            share_b = tot * b_r / (a_r + b_r)
            return (
                f"Ratio {a_r}:{b_r}, total = {tot}\n"
                f"Share A = {tot}×{a_r}/{a_r+b_r} = {share_a:.4g}\n"
                f"Share B = {tot}×{b_r}/{a_r+b_r} = {share_b:.4g}",
                f"A={share_a:.4g}, B={share_b:.4g}",
            )

    raise ValueError(
        "Could not identify the word problem type.\n"
        "Supported: area/perimeter/volume of shapes, percentage, "
        "simple/compound interest, speed-distance-time, work-rate, ratio/proportion."
    )


def _handle_competition_math(text: str) -> Tuple[str, str]:  # noqa: C901
    """
    Solver for competition/olympiad-style algebraic word problems (AIME/AMC).

    Currently handles:
      - Multi-person same-destination travel: N people start at staggered
        times with cumulative speed increments and all arrive simultaneously.
        Returns exact rational distance and m+n where gcd(m,n)=1.
    """
    from sympy import symbols, Eq, solve as sym_solve, Rational, simplify, Integer
    import math as _math

    t_low = text.lower()

    _WORD_NUMS = {
        'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
        'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
    }

    # ── Multi-person same-time arrival (speed-distance-time system) ───────────
    same_time = re.search(
        r'arrived?\s+at\s+the\s+(?:park|school|destination|same\s+(?:time|place))'
        r'|all\s+(?:three\s+)?(?:people\s+)?arrived?\s+at\s+the\s+same\s+time'
        r'|all\s+(?:three\s+)?(?:people\s+)?arrived?\s+at\s+the\s+park'
        r'|arrived.*at\s+the\s+same\s+time',
        t_low, re.I,
    )

    if same_time:
        # Extract speed increments: "N miles per hour faster"
        speed_incs_raw = re.findall(r'(\d+)\s+miles?\s+per\s+hour\s+faster', t_low, re.I)
        if not speed_incs_raw:
            for word, num in _WORD_NUMS.items():
                if re.search(rf'\b{word}\s+miles?\s+per\s+hour\s+faster', t_low, re.I):
                    speed_incs_raw.append(str(num))

        # Extract time offsets: "N hour(s) after" (cumulative start delays)
        time_offs_raw = re.findall(r'(\d+|one|two|three|four|five)\s+hours?\s+after', t_low, re.I)
        time_offs: list[int] = []
        for x in time_offs_raw:
            try:
                time_offs.append(int(x))
            except ValueError:
                time_offs.append(_WORD_NUMS.get(x.lower(), 1))

        if speed_incs_raw and time_offs:
            # Build cumulative speed offsets and cumulative start times
            cum_speed: list[int] = [0]
            running = 0
            for inc in speed_incs_raw:
                running += int(inc)
                cum_speed.append(running)

            cum_start: list[int] = [0]
            running = 0
            for off in time_offs:
                running += off
                cum_start.append(running)

            n_people = min(len(cum_speed), len(cum_start))

            # SymPy: v = first person's speed, T = first person's travel time
            v, T = symbols('v T', positive=True, real=True)
            d_ref = v * T

            equations = []
            for i in range(1, n_people):
                spd = v + Integer(cum_speed[i])
                tim = T - Integer(cum_start[i])
                equations.append(Eq(d_ref, spd * tim))

            try:
                sol = sym_solve(equations, [v, T], dict=True)
                if not sol:
                    sol = sym_solve(equations, [v, T])
                    if isinstance(sol, list) and sol:
                        v_val, T_val = sol[0]
                    else:
                        raise ValueError("No positive solution found")
                else:
                    v_val = sol[0][v]
                    T_val = sol[0][T]

                d_val = simplify(v_val * T_val)

                # Convert to exact rational m/n with gcd=1
                d_rat = Rational(d_val)
                m_val = int(d_rat.p)
                n_val = int(d_rat.q)
                g = _math.gcd(abs(m_val), abs(n_val))
                m_val, n_val = m_val // g, n_val // g

                # Attempt to extract person names from original text
                name_matches = re.findall(r'\b([A-Z][a-z]{2,})\b', text)
                _skip = {'One', 'Two', 'All', 'The', 'Find', 'School', 'Park',
                         'One', 'After', 'From', 'Same', 'Road', 'Hour', 'Both'}
                unique_names: list[str] = []
                seen: set[str] = set()
                for nm in name_matches:
                    if nm not in seen and nm not in _skip:
                        unique_names.append(nm)
                        seen.add(nm)
                person_names = (unique_names[:n_people]
                                if len(unique_names) >= n_people
                                else [f"Person {i+1}" for i in range(n_people)])

                # Build step-by-step explanation
                lines: list[str] = [
                    f"Let v = {person_names[0]}'s speed (mph), "
                    f"T = {person_names[0]}'s total travel time (hours).",
                    "",
                ]
                for i in range(n_people):
                    s_off = cum_speed[i]
                    t_off = cum_start[i]
                    spd_str = f"v + {s_off}" if s_off > 0 else "v"
                    tim_str = f"(T − {t_off})" if t_off > 0 else "T"
                    nm = person_names[i]
                    lines.append(f"  {nm}: speed = {spd_str} mph, travel time = {tim_str} h")

                lines += [
                    "",
                    "All arrive at the same destination, so all distances are equal:",
                ]
                for i in range(1, n_people):
                    nm = person_names[i]
                    s_off = cum_speed[i]
                    t_off = cum_start[i]
                    lines.append(
                        f"  v·T  =  (v + {s_off})·(T − {t_off})    [{nm} = {person_names[0]}]"
                    )

                lines += [
                    "",
                    "Expanding and solving the system of equations:",
                ]
                for i, eq in enumerate(equations):
                    lhs_str = "v·T"
                    s_off = cum_speed[i + 1]
                    t_off = cum_start[i + 1]
                    lines.append(
                        f"  Equation {i+1}: vT = (v+{s_off})(T−{t_off})"
                        f"  →  {s_off}T − {t_off}v = {s_off * t_off}"
                    )

                lines += [
                    "",
                    f"  Solved: v = {v_val}, T = {T_val}",
                    "",
                    f"Distance: d = v·T = {v_val} × {T_val} = {d_val}",
                    f"        = {m_val}/{n_val} miles",
                ]
                if n_val > 1:
                    lines += [
                        f"  gcd({m_val}, {n_val}) = 1  ✓  (m and n are relatively prime)",
                        f"",
                        f"  m + n  =  {m_val} + {n_val}  =  {m_val + n_val}",
                    ]

                explanation = "\n".join(lines)
                result_str = (
                    f"{m_val}/{n_val} miles"
                    + (f"  →  m + n = {m_val + n_val}" if n_val > 1 else "")
                )
                return (explanation, result_str)

            except Exception:
                pass  # fall through to error

    raise ValueError(
        "Could not solve this competition math problem.\n"
        "Supported pattern: multiple people start at staggered times with "
        "incremental speeds and all arrive at the same destination simultaneously."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Handler dispatch table
# ─────────────────────────────────────────────────────────────────────────────

_HANDLERS = {
    "competition_math": _handle_competition_math,
    "word_problem":     _handle_word_problem,
    "integrate":        _handle_integrate,
    "differentiate":    _handle_differentiate,
    "limit":            _handle_limit,
    "solve":            _handle_solve,
    "ode":              _handle_ode,
    "series":           _handle_series,
    "laplace":          _handle_laplace,
    "inverse_laplace":  _handle_inverse_laplace,
    "fourier":          _handle_fourier,
    "simplify":         _handle_simplify,
    "trig_simplify":    _handle_trig_simplify,
    "factor":           _handle_factor,
    "expand":           _handle_expand,
    "partial_fraction": _handle_partial_fraction,
    "eigenvalue":       _handle_eigenvalue,
    "determinant":      _handle_determinant,
    "inverse":          _handle_inverse,
    "matrix_rank":      _handle_matrix_rank,
    "matrix_trace":     _handle_matrix_trace,
    "gcd":              _handle_gcd,
    "lcm":              _handle_lcm,
    "prime_factors":    _handle_prime_factors,
    "modular":          _handle_modular,
    "statistics":       _handle_statistics,
    "factorial":        _handle_factorial,
    "binomial":         _handle_binomial,
    "permutation":      _handle_permutation,
    "summation":        _handle_summation,
    "product":          _handle_product,
    "complex_ops":      _handle_complex_ops,
}


# ─────────────────────────────────────────────────────────────────────────────
# Public interface
# ─────────────────────────────────────────────────────────────────────────────

def solve(user_input: str) -> Tuple[bool, str, str]:
    """
    Main entry point for the advanced math engine.

    Args:
        user_input: Natural language math query.

    Returns:
        (success, result_str, latex_str)
        success    – True if SymPy computed an answer
        result_str – Human-readable answer
        latex_str  – LaTeX of the result

    Uses top-2 type prediction: if the primary detected operation fails,
    the secondary candidate is attempted before returning failure.
    """
    # Normalize input (Unicode, arrows, superscripts, delta signs…)
    user_input = normalize_input(user_input)

    candidates = detect_advanced_operation_ranked(user_input)
    if not candidates:
        return (False, "", "")

    last_error = ""
    for op in candidates:
        handler = _HANDLERS.get(op)
        if handler is None:
            last_error = f"Operation '{op}' recognised but not yet implemented."
            continue
        try:
            result_str, latex_str = handler(user_input)
            return (True, result_str, latex_str)
        except Exception as exc:
            last_error = f"Math engine error ({op}): {exc}"

    return (False, last_error, "")
