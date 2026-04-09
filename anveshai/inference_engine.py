"""
Inference Engine — formal logical reasoning without an LLM.

Handles:
  · Modus Ponens:          If P → Q and P, then Q
  · Modus Tollens:         If P → Q and ¬Q, then ¬P
  · Hypothetical Syllogism: If P → Q and Q → R, then P → R
  · Disjunctive Syllogism:  If P ∨ Q and ¬P, then Q
  · Categorical Syllogisms: All A are B; X is A; therefore X is B
  · Propositional evaluation: "P and Q", "P or Q", "not P", "P implies Q"
  · Contradiction detection: returns False when premises are contradictory
  · Consistency checking: verifies a set of statements is mutually consistent

All processing is pure Python — zero external dependencies.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class InferenceResult:
    """Result from the inference engine."""
    conclusion:    str                  = ""
    valid:         bool                 = False
    rule_applied:  str                  = ""
    proof_steps:   list[str]            = field(default_factory=list)
    confidence:    str                  = "HIGH"   # HIGH / MEDIUM / LOW
    is_tautology:  Optional[bool]       = None
    truth_value:   Optional[bool]       = None

    def to_response(self) -> str:
        lines = [f"Conclusion: {self.conclusion}"]
        if self.rule_applied:
            lines.append(f"Inference rule: {self.rule_applied}")
        if self.proof_steps:
            lines.append("Proof:")
            for i, step in enumerate(self.proof_steps, 1):
                lines.append(f"  {i}. {step}")
        if self.is_tautology is not None:
            lines.append(f"Tautology: {'Yes' if self.is_tautology else 'No'}")
        lines.append(f"Argument is: {'VALID' if self.valid else 'INVALID'}")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Natural language premise parser
# ─────────────────────────────────────────────────────────────────────────────

def _extract_if_then(text: str) -> Optional[tuple[str, str]]:
    """Extract (antecedent, consequent) from 'if P then Q' patterns."""
    patterns = [
        r"if\s+(.+?)\s+then\s+(.+?)[\.,;]?$",
        r"(.+?)\s+implies\s+(.+?)[\.,;]?$",
        r"(.+?)\s+→\s+(.+?)[\.,;]?$",
        r"when\s+(.+?)[,]\s+(.+?)[\.,;]?$",
        r"whenever\s+(.+?)[,]\s+(.+?)[\.,;]?$",
    ]
    for pat in patterns:
        m = re.search(pat, text.lower().strip())
        if m:
            return m.group(1).strip(), m.group(2).strip()
    return None


_ALL_VERBS = (
    "are", "have", "is", "require", "need", "must", "contain",
    "use", "produce", "involve", "consist of", "depend on",
)
_ALL_VERB_RE = re.compile(
    r"all\s+(.+?)\s+"
    r"(are|have|is|require|need|must|contain|use|produce|involve|consist\s+of|depend\s+on)"
    r"\s+(.+?)[\.,;]?$"
)

def _extract_all_are(text: str) -> Optional[tuple[str, str, str]]:
    """
    Extract (category, property, predicate_verb) from
    'All A are/have/require/need/must/use B' patterns.
    Returns (category, property, verb).
    """
    t = text.lower().strip()
    m = _ALL_VERB_RE.search(t)
    if m:
        return m.group(1).strip(), m.group(3).strip(), m.group(2).strip()
    return None


def _extract_negation(text: str) -> Optional[str]:
    """Extract the negated claim from negation patterns."""
    t = text.lower().strip(" .,;?!")

    # Starts-with patterns
    for prefix in ["not ", "¬", "it is not the case that ", "it is false that "]:
        if t.startswith(prefix):
            return t[len(prefix):].strip()

    # Mid-sentence negation: "X did not Y" → "X will Y" / "X does Y" / "X is Y"
    m = re.match(r"^(.+?)\s+did\s+not\s+(.+)$", t)
    if m:
        subject, verb_phrase = m.group(1).strip(), m.group(2).strip()
        return f"{subject} will {verb_phrase}"

    # "X does not Y" → "X does Y"
    m = re.match(r"^(.+?)\s+does\s+not\s+(.+)$", t)
    if m:
        subject, verb_phrase = m.group(1).strip(), m.group(2).strip()
        return f"{subject} {verb_phrase}"

    # "X do not Y" → "X Y"  (plural subjects)
    m = re.match(r"^(.+?)\s+do\s+not\s+(.+)$", t)
    if m:
        subject, verb_phrase = m.group(1).strip(), m.group(2).strip()
        return f"{subject} {verb_phrase}"

    # "X is not Y" → "X is Y"
    m = re.match(r"^(.+?)\s+is\s+not\s+(.+)$", t)
    if m:
        subject, obj = m.group(1).strip(), m.group(2).strip()
        return f"{subject} is {obj}"

    # "X are not Y" → "X are Y"
    m = re.match(r"^(.+?)\s+are\s+not\s+(.+)$", t)
    if m:
        subject, obj = m.group(1).strip(), m.group(2).strip()
        return f"{subject} are {obj}"

    # "X was not Y" → "X was Y"
    m = re.match(r"^(.+?)\s+was\s+not\s+(.+)$", t)
    if m:
        subject, obj = m.group(1).strip(), m.group(2).strip()
        return f"{subject} was {obj}"

    # "X has not Y-ed" → "X has Y-ed"
    m = re.match(r"^(.+?)\s+has\s+not\s+(.+)$", t)
    if m:
        subject, obj = m.group(1).strip(), m.group(2).strip()
        return f"{subject} has {obj}"

    return None


def _normalise(text: str) -> str:
    return text.lower().strip(" .,;?!")


def _negate_clause(clause: str) -> str:
    """
    Produce a natural-language negation of a simple clause.
    e.g. 'the temperature drops' → 'the temperature does not drop'
         'it rains' → 'it does not rain'
         'she is happy' → 'she is not happy'
    """
    t = clause.lower().strip()

    # "X is/are/was/were Y" → "X is/are/was/were not Y"
    for verb in ("is", "are", "was", "were"):
        pat = rf"^(.*?)\b{verb}\b(.*)$"
        m = re.match(pat, t)
        if m:
            return f"{m.group(1).strip()} {verb} not{m.group(2)}"

    # "X will Y" → "X will not Y"
    m = re.match(r"^(.*?)\bwill\b(.*)$", t)
    if m:
        return f"{m.group(1).strip()} will not{m.group(2)}"

    # "X can Y" → "X cannot Y"
    m = re.match(r"^(.*?)\bcan\b(.*)$", t)
    if m:
        return f"{m.group(1).strip()} cannot{m.group(2)}"

    # General: for simple clauses (≤5 words) treat the last word as the verb
    # For longer clauses, use "does not" before the second-to-last meaningful word
    words = t.split()
    if len(words) == 1:
        return f"not {clause}"
    if len(words) <= 5:
        # Last word is the main verb; everything before it is the subject
        subject_part = " ".join(words[:-1])
        verb_part    = words[-1]
        return f"{subject_part} does not {verb_part}"
    # Longer clause: insert 'does not' after first two words
    subject_part = " ".join(words[:2])
    verb_part    = " ".join(words[2:])
    return f"{subject_part} does not {verb_part}"


def _fuzzy_match(a: str, b: str) -> bool:
    """True if a and b are close enough to be considered the same claim."""
    a, b = _normalise(a), _normalise(b)
    if a == b:
        return True
    # Simple stem: strip trailing 's', 'ed', 'ing'
    def _stem(w: str) -> str:
        for suf in ("ing", "ed", "s"):
            if w.endswith(suf) and len(w) > len(suf) + 2:
                return w[:-len(suf)]
        return w

    a_words = {_stem(w) for w in a.split()}
    b_words = {_stem(w) for w in b.split()}
    if not a_words or not b_words:
        return False
    overlap = a_words & b_words
    union   = a_words | b_words
    # Jaccard similarity ≥ 0.5 means they are about the same thing
    return len(overlap) / len(union) >= 0.5


# ─────────────────────────────────────────────────────────────────────────────
# Propositional logic evaluator
# ─────────────────────────────────────────────────────────────────────────────

def _eval_prop(expr: str, assignments: dict[str, bool]) -> Optional[bool]:
    """
    Evaluate a simple propositional expression given variable assignments.
    Supports: AND, OR, NOT, IMPLIES (→ / =>), IFF (<->)
    Variables are single uppercase letters or short words.
    Returns None if the expression cannot be parsed.
    """
    expr = expr.strip()

    # Normalise operators
    expr = re.sub(r"\bimplies\b", "=>", expr, flags=re.IGNORECASE)
    expr = re.sub(r"→", "=>", expr)
    expr = re.sub(r"<->|↔", "IFF", expr)
    expr = re.sub(r"\band\b", "AND", expr, flags=re.IGNORECASE)
    expr = re.sub(r"\bor\b", "OR", expr, flags=re.IGNORECASE)
    expr = re.sub(r"\bnot\b|¬", "NOT ", expr, flags=re.IGNORECASE)

    # Replace variables with their truth values
    for var, val in sorted(assignments.items(), key=lambda x: -len(x[0])):
        expr = re.sub(r"\b" + re.escape(var) + r"\b", str(val), expr)

    try:
        expr_py = (
            expr
            .replace("AND", " and ")
            .replace("OR", " or ")
            .replace("NOT ", " not ")
            .replace("=>", " <= ")   # P => Q  ≡  (not P) or Q
        )
        # Handle implication: P => Q is not P or Q
        # Rebuild properly:
        def _replace_implies(e: str) -> str:
            parts = re.split(r"\s*<=\s*", e)
            if len(parts) == 2:
                return f"(not ({parts[0].strip()}) or ({parts[1].strip()}))"
            return e

        expr_py2 = _replace_implies(expr_py)
        result = eval(expr_py2, {"__builtins__": {}}, {"True": True, "False": False})
        return bool(result)
    except Exception:
        return None


def _generate_truth_table(variables: list[str], formula: str) -> list[dict]:
    """Generate truth table rows for a propositional formula."""
    n = len(variables)
    rows = []
    for i in range(2 ** n):
        assignment = {}
        for j, var in enumerate(variables):
            assignment[var] = bool((i >> (n - 1 - j)) & 1)
        result = _eval_prop(formula, assignment)
        rows.append({**assignment, "result": result})
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Inference rules
# ─────────────────────────────────────────────────────────────────────────────

def _modus_ponens(premise1: str, premise2: str) -> Optional[InferenceResult]:
    """
    Modus Ponens: If P → Q and P, then Q.
    premise1 should be the conditional; premise2 should be P.
    """
    cond = _extract_if_then(premise1)
    if not cond:
        cond = _extract_if_then(premise2)
        if cond:
            premise1, premise2 = premise2, premise1

    if not cond:
        return None

    antecedent, consequent = cond
    p2_norm = _normalise(premise2)

    if _fuzzy_match(p2_norm, antecedent) or antecedent in p2_norm:
        return InferenceResult(
            conclusion=consequent.capitalize(),
            valid=True,
            rule_applied="Modus Ponens (P → Q, P ⊢ Q)",
            proof_steps=[
                f"Premise 1: {premise1.strip()}",
                f"Premise 2: {premise2.strip()}",
                f"Premise 1 is a conditional: if '{antecedent}' then '{consequent}'",
                f"Premise 2 affirms the antecedent: '{antecedent}'",
                f"By Modus Ponens, the consequent follows: '{consequent}'",
            ],
            confidence="HIGH",
        )
    return None


def _modus_tollens(premise1: str, premise2: str) -> Optional[InferenceResult]:
    """
    Modus Tollens: If P → Q and ¬Q, then ¬P.
    """
    cond = _extract_if_then(premise1)
    if not cond:
        cond = _extract_if_then(premise2)
        if cond:
            premise1, premise2 = premise2, premise1

    if not cond:
        return None

    antecedent, consequent = cond
    neg_q = _extract_negation(premise2)
    if neg_q and (_fuzzy_match(neg_q, consequent) or consequent in neg_q):
        # Build a natural negation of the antecedent
        neg_ant = _negate_clause(antecedent)
        return InferenceResult(
            conclusion=neg_ant.capitalize(),
            valid=True,
            rule_applied="Modus Tollens (P → Q, ¬Q ⊢ ¬P)",
            proof_steps=[
                f"Premise 1: {premise1.strip()}",
                f"Premise 2: {premise2.strip()}",
                f"Premise 1 is a conditional: if '{antecedent}' then '{consequent}'",
                f"Premise 2 denies the consequent: 'not {consequent}'",
                f"By Modus Tollens, the antecedent is denied: '{neg_ant}'",
            ],
            confidence="HIGH",
        )
    return None


def _hypothetical_syllogism(p1: str, p2: str) -> Optional[InferenceResult]:
    """
    Hypothetical Syllogism: If P → Q and Q → R, then P → R.
    """
    cond1 = _extract_if_then(p1)
    cond2 = _extract_if_then(p2)
    if not cond1 or not cond2:
        return None

    ant1, cons1 = cond1
    ant2, cons2 = cond2

    if _normalise(cons1) == _normalise(ant2) or cons1 in ant2:
        return InferenceResult(
            conclusion=f"If {ant1}, then {cons2}",
            valid=True,
            rule_applied="Hypothetical Syllogism (P → Q, Q → R ⊢ P → R)",
            proof_steps=[
                f"Premise 1: if '{ant1}' then '{cons1}'",
                f"Premise 2: if '{ant2}' then '{cons2}'",
                f"The consequent of Premise 1 ('{cons1}') matches the antecedent of Premise 2 ('{ant2}')",
                f"By Hypothetical Syllogism: if '{ant1}' then '{cons2}'",
            ],
            confidence="HIGH",
        )
    return None


def _categorical_syllogism(p1: str, p2: str) -> Optional[InferenceResult]:
    """
    Categorical Syllogism: All A are/have B; X is A; therefore X is/has B.
    """
    all_match = _extract_all_are(p1)
    if not all_match:
        all_match = _extract_all_are(p2)
        if all_match:
            p1, p2 = p2, p1

    if not all_match:
        return None

    category, prop, verb = all_match
    p2_norm = _normalise(p2)

    is_patterns = [
        rf"\bis\s+a\s+{re.escape(category)}\b",
        rf"\bis\s+an\s+{re.escape(category)}\b",
        rf"\bis\s+{re.escape(category)}\b",
        rf"\bare\s+{re.escape(category)}\b",
        rf"\b{re.escape(category)}\b",
    ]
    def _clean_subject(raw: str) -> str:
        """Strip articles and anything after 'is/are/was/were'."""
        # Cut at first verb ('is', 'are', 'was', 'were', 'has')
        raw = re.split(r"\s+(?:is|are|was|were|has)\b", raw)[0].strip()
        # Remove leading article
        raw = re.sub(r"^(?:a|an|the)\s+", "", raw).strip(" .,;?!")
        return raw

    subject = None
    for pat in is_patterns:
        m = re.search(pat, p2_norm)
        if m:
            raw_subj = p2_norm[:m.start()].strip(" .,;?!")
            subject = _clean_subject(raw_subj)
            if subject:
                break

    # Fallback: fuzzy match any word in p2 against category
    if not subject:
        words = p2_norm.split()
        for i, word in enumerate(words):
            if _fuzzy_match(word, category):
                raw = " ".join(words[:i]).strip(" .,;?!")
                subject = _clean_subject(raw) or words[0]
                break

    if subject:
        # Build a natural conclusion with correct verb conjugation
        if verb == "have":
            conclusion = f"{subject.capitalize()} has {prop}"
        elif verb in ("are", "is"):
            conclusion = f"{subject.capitalize()} is {prop}"
        else:
            # Preserve original verb (require → requires, need → needs, etc.)
            verb_s = verb.rstrip("e") + "s" if not verb.endswith("s") else verb
            conclusion = f"{subject.capitalize()} {verb_s} {prop}"
        return InferenceResult(
            conclusion=conclusion,
            valid=True,
            rule_applied="Categorical Syllogism (All A are B; X is A ⊢ X is B)",
            proof_steps=[
                f"Major premise: All {category} {verb} {prop}",
                f"Minor premise: {p2.strip()}",
                f"'{subject}' is identified as a member of '{category}'",
                f"By categorical syllogism: '{subject}' {verb} '{prop}'",
            ],
            confidence="HIGH",
        )
    return None


_EMBEDDED_EITHER_RE = re.compile(
    r"\beither\s+(.+?)\s+or\s+(.+?)[\.,;]?$"
)
_GENERAL_OR_RE = re.compile(
    r"(?:either\s+)?(.+?)\s+or\s+(.+?)[\.,;]?$"
)


def _extract_disjuncts(text: str) -> Optional[tuple[str, str]]:
    """
    Extract (option_a, option_b) from a disjunction sentence.
    Prefers the embedded 'either X or Y' pattern so that surrounding
    context ('The prompt is routed to either X or Y') doesn't pollute option_a.
    Falls back to the first 'X or Y' found.
    """
    t = text.lower().strip()
    # Prefer embedded "either X or Y" — strips context before "either"
    m = _EMBEDDED_EITHER_RE.search(t)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    # General fallback
    m = _GENERAL_OR_RE.search(t)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return None


def _disjunctive_syllogism(p1: str, p2: str) -> Optional[InferenceResult]:
    """
    Disjunctive Syllogism: P ∨ Q and ¬P ⊢ Q.
    """
    disjuncts = _extract_disjuncts(p1)
    if not disjuncts:
        disjuncts = _extract_disjuncts(p2)
        if disjuncts:
            p1, p2 = p2, p1

    if not disjuncts:
        return None

    option_a, option_b = disjuncts
    neg = _extract_negation(p2)
    if neg:
        neg_n = _normalise(neg)
        if _fuzzy_match(neg, option_a) or option_a in neg_n:
            return InferenceResult(
                conclusion=option_b.capitalize(),
                valid=True,
                rule_applied="Disjunctive Syllogism (P ∨ Q, ¬P ⊢ Q)",
                proof_steps=[
                    f"Disjunction: '{option_a}' OR '{option_b}'",
                    f"Negation: NOT '{option_a}'",
                    f"By Disjunctive Syllogism: '{option_b}'",
                ],
                confidence="HIGH",
            )
        if _fuzzy_match(neg, option_b) or option_b in neg_n:
            return InferenceResult(
                conclusion=option_a.capitalize(),
                valid=True,
                rule_applied="Disjunctive Syllogism (P ∨ Q, ¬Q ⊢ P)",
                proof_steps=[
                    f"Disjunction: '{option_a}' OR '{option_b}'",
                    f"Negation: NOT '{option_b}'",
                    f"By Disjunctive Syllogism: '{option_a}'",
                ],
                confidence="HIGH",
            )
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Public interface
# ─────────────────────────────────────────────────────────────────────────────

class InferenceEngine:
    """
    Pure logical inference engine.

    Usage:
        ie = InferenceEngine()
        result = ie.infer(user_input)
        if result.valid:
            print(result.to_response())
    """

    def infer(self, user_input: str) -> InferenceResult:
        """
        Attempt to infer a conclusion from the user's input.
        Tries all known inference rules in order.
        Returns an InferenceResult with valid=False if nothing can be derived.
        """
        # Split input into premises
        raw_sentences = re.split(r"[.;]\s+|\n|,\s+and\s+", user_input.strip())
        # Filter out conclusion prompts like "Therefore?", "What follows?", "What can we conclude?"
        _CONCLUSION_PROMPTS = re.compile(
            r"^(therefore|what follows|what can we conclude|"
            r"what\s+is\s+the\s+conclusion|so\s+what|"
            r"can\s+we\s+conclude|what\s+do\s+we\s+know)[?.,\s]*$",
            re.IGNORECASE,
        )
        premises = [
            s.strip() for s in raw_sentences
            if len(s.strip()) > 3 and not _CONCLUSION_PROMPTS.match(s.strip())
        ]

        if len(premises) < 2:
            return InferenceResult(
                conclusion="",
                valid=False,
                rule_applied="",
                proof_steps=["At least two premises are required for formal inference"],
            )

        # Try pairwise inference
        for i in range(len(premises)):
            for j in range(len(premises)):
                if i == j:
                    continue
                p1, p2 = premises[i], premises[j]

                result = _modus_ponens(p1, p2)
                if result and result.valid:
                    return result

                result = _modus_tollens(p1, p2)
                if result and result.valid:
                    return result

                result = _hypothetical_syllogism(p1, p2)
                if result and result.valid:
                    return result

                result = _categorical_syllogism(p1, p2)
                if result and result.valid:
                    return result

                result = _disjunctive_syllogism(p1, p2)
                if result and result.valid:
                    return result

        return InferenceResult(
            conclusion="",
            valid=False,
            rule_applied="No matching inference rule",
            proof_steps=[
                "Premises identified: " + " | ".join(f"'{p}'" for p in premises),
                "None of the standard inference rules (Modus Ponens, Modus Tollens, "
                "Hypothetical Syllogism, Categorical Syllogism, Disjunctive Syllogism) "
                "could be applied to derive a certain conclusion.",
                "The argument may require domain knowledge or be invalid.",
            ],
        )

    def check_consistency(self, statements: list[str]) -> tuple[bool, str]:
        """
        Check whether a list of statements is mutually consistent.
        Returns (is_consistent, explanation).
        """
        if len(statements) < 2:
            return True, "Only one statement — trivially consistent."

        for i, s1 in enumerate(statements):
            for j, s2 in enumerate(statements):
                if i >= j:
                    continue
                neg_s1 = _extract_negation(s1)
                neg_s2 = _extract_negation(s2)

                if neg_s1 and _normalise(neg_s1) == _normalise(s2):
                    return False, (
                        f"Contradiction: '{s1}' directly negates '{s2}'"
                    )
                if neg_s2 and _normalise(neg_s2) == _normalise(s1):
                    return False, (
                        f"Contradiction: '{s2}' directly negates '{s1}'"
                    )

        return True, "No direct contradictions detected among the statements."

    def evaluate_proposition(self, formula: str) -> InferenceResult:
        """
        Evaluate a simple propositional formula with variable assignments.
        Example: "P AND Q where P=True, Q=False"
        """
        var_m = re.search(r"where\s+(.+)$", formula, re.IGNORECASE)
        assignments: dict[str, bool] = {}

        if var_m:
            formula_part = formula[: var_m.start()].strip()
            assign_text  = var_m.group(1)
            for pair in re.split(r",\s*", assign_text):
                pair_m = re.match(r"([A-Za-z]\w*)\s*=\s*(true|false|1|0)", pair.strip(), re.IGNORECASE)
                if pair_m:
                    var_name = pair_m.group(1)
                    val_str  = pair_m.group(2).lower()
                    assignments[var_name] = val_str in ("true", "1")
        else:
            formula_part = formula

        result = _eval_prop(formula_part, assignments)

        if result is None:
            return InferenceResult(
                conclusion="Cannot evaluate: formula is not in a recognisable form",
                valid=False,
            )

        vars_in_formula = re.findall(r"\b([A-Z])\b", formula_part)
        is_tautology = None
        if vars_in_formula and not assignments:
            rows = _generate_truth_table(list(dict.fromkeys(vars_in_formula)), formula_part)
            if all(r["result"] for r in rows):
                is_tautology = True
            elif not any(r["result"] for r in rows):
                is_tautology = False

        return InferenceResult(
            conclusion=f"The expression evaluates to: {result}",
            valid=True,
            rule_applied="Propositional evaluation",
            truth_value=result,
            is_tautology=is_tautology,
            proof_steps=[
                f"Formula: {formula_part}",
                f"Assignments: {assignments if assignments else 'none given'}",
                f"Result: {result}",
            ],
        )
