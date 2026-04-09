"""
preprocess.py — Input normalization pipeline for AnveshAI Edge.

Normalizes raw user input before it reaches the router or engines.
Fixes broken Unicode, arrows, delta signs, superscripts, OCR artifacts,
extra whitespace, and unit prefixes so downstream regex patterns see
consistent ASCII-like text.

Call normalize_input() once at the entry point (classify_intent / solve).
"""

import re


# ─────────────────────────────────────────────────────────────────────────────
# Superscript / subscript digit tables
# ─────────────────────────────────────────────────────────────────────────────

_SUPERSCRIPT_MAP: dict[str, str] = {
    '\u2070': '^0', '\u00b9': '^1', '\u00b2': '^2', '\u00b3': '^3',
    '\u2074': '^4', '\u2075': '^5', '\u2076': '^6', '\u2077': '^7',
    '\u2078': '^8', '\u2079': '^9',
    '\u207a': '^+', '\u207b': '^-', '\u207f': '^n',
    '\u00b0': ' degrees',   # degree sign ° — fits naturally here
}

_SUBSCRIPT_MAP: dict[str, str] = {
    '\u2080': '_0', '\u2081': '_1', '\u2082': '_2', '\u2083': '_3',
    '\u2084': '_4', '\u2085': '_5', '\u2086': '_6', '\u2087': '_7',
    '\u2088': '_8', '\u2089': '_9',
}


# ─────────────────────────────────────────────────────────────────────────────
# Ordered symbol-replacement table
# (longer / more specific entries first to avoid partial overwrites)
# ─────────────────────────────────────────────────────────────────────────────

_SYMBOL_REPLACEMENTS: list[tuple[str, str]] = [
    # ── Calculus / set / summation symbols ────────────────────────────────────
    ('\u222b', 'integrate '),    # ∫
    ('\u2211', 'sum of '),       # ∑
    ('\u220f', 'product of '),   # ∏
    ('\u2202', 'd'),             # ∂ (partial diff)
    ('\u221e', 'oo'),            # ∞
    ('\u2207', 'nabla '),        # ∇

    # ── Arrows (longest / most specific first) ────────────────────────────────
    ('\u27f9', ' => '),          # ⟹ long double right
    ('\u27f6', ' -> '),          # ⟶ long right
    ('\u21d2', ' => '),          # ⇒ double right
    ('\u2192', ' -> '),          # →
    ('\u21d0', ' <= '),          # ⇐
    ('\u2190', ' <- '),          # ←
    ('\u21d4', ' <=> '),         # ⇔
    ('\u2194', ' <-> '),         # ↔

    # ── Delta / gradient ──────────────────────────────────────────────────────
    ('\u0394', 'delta '),        # Δ (capital)
    ('\u2206', 'delta '),        # ∆ (increment sign, often confused with Δ)
    ('\u03b4', 'delta '),        # δ (lowercase)

    # ── Greek letters (common in science) ────────────────────────────────────
    ('\u03bc', 'micro'),         # µ (micro prefix, e.g. µC, µm)
    ('\u00b5', 'micro'),         # µ (alternative encoding)
    ('\u03a9', 'ohm'),           # Ω
    ('\u03c9', 'omega'),         # ω
    ('\u03b1', 'alpha'),         # α
    ('\u03b2', 'beta'),          # β
    ('\u03b3', 'gamma'),         # γ
    ('\u03bb', 'lambda'),        # λ
    ('\u03b8', 'theta'),         # θ
    ('\u03c6', 'phi'),           # φ
    ('\u03c0', 'pi'),            # π  (only in isolation — keep as 'pi')
    ('\u03b5', 'epsilon'),       # ε
    ('\u03c3', 'sigma'),         # σ
    ('\u03c4', 'tau'),           # τ
    ('\u03b7', 'eta'),           # η
    ('\u03ba', 'kappa'),         # κ
    ('\u03bd', 'nu'),            # ν
    ('\u03c1', 'rho'),           # ρ
    ('\u03c7', 'chi'),           # χ

    # ── Math operators ────────────────────────────────────────────────────────
    ('\u00d7', ' cross '),       # × (vector cross / multiplication)
    ('\u00f7', '/'),             # ÷
    ('\u2212', '-'),             # − (unicode minus → ASCII hyphen)
    ('\u2013', '-'),             # – en dash
    ('\u2014', '-'),             # — em dash
    ('\u00b1', '+/-'),           # ±
    ('\u2264', '<='),            # ≤
    ('\u2265', '>='),            # ≥
    ('\u2260', '!='),            # ≠
    ('\u2248', '~='),            # ≈
    ('\u221a', 'sqrt'),          # √
    ('\u00b7', '*'),             # · (middle dot)

    # ── Fractions ─────────────────────────────────────────────────────────────
    ('\u00bd', '1/2'),           # ½
    ('\u2153', '1/3'),           # ⅓
    ('\u00bc', '1/4'),           # ¼
    ('\u00be', '3/4'),           # ¾
    ('\u2154', '2/3'),           # ⅔

    # ── Typographic / OCR artifacts ───────────────────────────────────────────
    ('\u201c', '"'), ('\u201d', '"'),   # " "
    ('\u201e', '"'), ('\u201f', '"'),   # „ ‟
    ('\u2018', "'"), ('\u2019', "'"),   # ' '
    ('\u201a', "'"), ('\u201b', "'"),   # ‚ ‛
    ('\u00a0', ' '),                    # non-breaking space
    ('\u2009', ' '),                    # thin space
    ('\u200b', ''),                     # zero-width space
    ('\u200c', ''),                     # zero-width non-joiner
    ('\u200d', ''),                     # zero-width joiner
    ('\ufeff', ''),                     # BOM
]


# ─────────────────────────────────────────────────────────────────────────────
# Compiled patterns for post-substitution cleanup
# ─────────────────────────────────────────────────────────────────────────────

_MULTI_SPACE = re.compile(r'[ \t]+')
_MULTI_NL    = re.compile(r'\n+')

# Fix "cross " that was inserted between a variable and "product" — keep as-is,
# but collapse "cross  product" to "cross product".
_CROSS_SPACE = re.compile(r'cross\s{2,}')


def normalize_input(text: str) -> str:
    """
    Normalize a raw user input string.

    Returns a cleaned string with:
      • Unicode superscripts/subscripts replaced with ASCII caret/underscore notation
      • Mathematical symbols (∫, ∑, →, ×, ∆, µ, ²…) replaced with ASCII words
      • Typographic quotes/dashes/spaces replaced with ASCII equivalents
      • Multiple spaces collapsed to one
      • Leading/trailing whitespace stripped

    The returned string is *always* a valid str (never None).
    """
    if not isinstance(text, str):
        return str(text) if text is not None else ''

    # 1. Strip
    text = text.strip()
    if not text:
        return text

    # 2. Superscript / subscript digits
    for ch, repl in _SUPERSCRIPT_MAP.items():
        text = text.replace(ch, repl)
    for ch, repl in _SUBSCRIPT_MAP.items():
        text = text.replace(ch, repl)

    # 3. Symbol table (order matters: longest / most specific first)
    for symbol, repl in _SYMBOL_REPLACEMENTS:
        text = text.replace(symbol, repl)

    # 4. Post-cleanup
    text = _CROSS_SPACE.sub('cross ', text)
    text = _MULTI_SPACE.sub(' ', text)
    text = _MULTI_NL.sub(' ', text)
    text = text.strip()

    return text
