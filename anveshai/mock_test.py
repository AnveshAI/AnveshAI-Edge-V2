"""
Mock Test Mode — JEE Advanced style timed practice tests.

Scoring: +4 correct, -1 wrong, 0 skipped.
Sessions: 20 questions, mixed Physics/Chemistry/Math.
"""

import time
import random
from typing import List, Tuple, Optional

try:
    from colorama import Fore, Style
except ImportError:
    class _NC:
        def __getattr__(self, _): return ""
    Fore = Style = _NC()


# ─────────────────────────────────────────────────────────────────────────────
# Question bank — (subject, topic, question, options, correct_index, explanation)
# ─────────────────────────────────────────────────────────────────────────────

QUESTIONS: List[dict] = [
    # ── Physics ──────────────────────────────────────────────────────────────
    {
        "subject": "Physics", "topic": "Kinematics",
        "q": "A body starts from rest and moves with uniform acceleration 4 m/s². Distance covered in 5 s is:",
        "opts": ["50 m", "40 m", "25 m", "100 m"],
        "ans": 0,
        "exp": "s = ½at² = ½ × 4 × 25 = 50 m",
    },
    {
        "subject": "Physics", "topic": "Newton's Laws",
        "q": "A 10 kg block is pushed with 50 N. If friction coefficient μ = 0.3 (g = 10 m/s²), acceleration is:",
        "opts": ["2 m/s²", "5 m/s²", "3 m/s²", "1 m/s²"],
        "ans": 0,
        "exp": "a = (F − μmg)/m = (50 − 30)/10 = 2 m/s²",
    },
    {
        "subject": "Physics", "topic": "Energy",
        "q": "Kinetic energy of a 2 kg body moving at 10 m/s is:",
        "opts": ["100 J", "20 J", "200 J", "50 J"],
        "ans": 0,
        "exp": "KE = ½mv² = ½ × 2 × 100 = 100 J",
    },
    {
        "subject": "Physics", "topic": "Gravitation",
        "q": "Escape velocity from Earth surface (g = 9.8 m/s², R = 6.4×10⁶ m) is approximately:",
        "opts": ["11.2 km/s", "7.9 km/s", "9.8 km/s", "8.5 km/s"],
        "ans": 0,
        "exp": "vₑ = √(2gR) = √(2 × 9.8 × 6.4×10⁶) ≈ 11,200 m/s = 11.2 km/s",
    },
    {
        "subject": "Physics", "topic": "Waves",
        "q": "A wave has frequency 500 Hz and wavelength 0.68 m. Its speed is:",
        "opts": ["340 m/s", "500 m/s", "0.68 m/s", "250 m/s"],
        "ans": 0,
        "exp": "v = fλ = 500 × 0.68 = 340 m/s",
    },
    {
        "subject": "Physics", "topic": "Optics",
        "q": "Snell's law: light enters from air (n₁=1) into glass (n₂=1.5) at 30°. Refracted angle is:",
        "opts": ["19.47°", "30°", "45°", "15°"],
        "ans": 0,
        "exp": "sin θ₂ = (n₁/n₂)sin θ₁ = (1/1.5)×0.5 = 0.333 → θ₂ ≈ 19.47°",
    },
    {
        "subject": "Physics", "topic": "Thermodynamics",
        "q": "Heat required to raise 2 kg of water by 5°C (c = 4200 J/kg·K) is:",
        "opts": ["42 000 J", "8400 J", "21 000 J", "4200 J"],
        "ans": 0,
        "exp": "Q = mcΔT = 2 × 4200 × 5 = 42 000 J",
    },
    {
        "subject": "Physics", "topic": "Electrostatics",
        "q": "Two charges of +6 µC and +6 µC are 0.3 m apart. Coulomb force between them is:",
        "opts": ["3.60 N", "36.0 N", "0.36 N", "360 N"],
        "ans": 0,
        "exp": "F = kq₁q₂/r² = 8.99×10⁹ × (6×10⁻⁶)² / 0.09 ≈ 3.60 N",
    },
    {
        "subject": "Physics", "topic": "Modern Physics",
        "q": "de Broglie wavelength of an electron (m = 9.11×10⁻³¹ kg) moving at 2×10⁶ m/s is approximately:",
        "opts": ["0.364 nm", "0.1 nm", "1.2 nm", "3.64 nm"],
        "ans": 0,
        "exp": "λ = h/mv = 6.626×10⁻³⁴ / (9.11×10⁻³¹ × 2×10⁶) ≈ 3.64×10⁻¹⁰ m = 0.364 nm",
    },
    {
        "subject": "Physics", "topic": "Circular Motion",
        "q": "A 0.5 kg ball moves in a circle of radius 1.5 m at 6 m/s. Centripetal force is:",
        "opts": ["12 N", "6 N", "18 N", "3 N"],
        "ans": 0,
        "exp": "F = mv²/r = 0.5 × 36 / 1.5 = 12 N",
    },
    # ── Chemistry ────────────────────────────────────────────────────────────
    {
        "subject": "Chemistry", "topic": "Moles",
        "q": "Number of moles in 44 g of CO₂ (M = 44 g/mol):",
        "opts": ["1 mol", "2 mol", "0.5 mol", "44 mol"],
        "ans": 0,
        "exp": "n = m/M = 44/44 = 1 mol",
    },
    {
        "subject": "Chemistry", "topic": "pH",
        "q": "pH of 0.001 M HCl solution is:",
        "opts": ["3", "11", "1", "7"],
        "ans": 0,
        "exp": "[H⁺] = 0.001 M = 10⁻³ → pH = 3",
    },
    {
        "subject": "Chemistry", "topic": "Gas Laws",
        "q": "Gas at P₁=2 atm, V₁=3 L is compressed to V₂=1 L at constant T. P₂ is (Boyle's law):",
        "opts": ["6 atm", "1.5 atm", "3 atm", "2 atm"],
        "ans": 0,
        "exp": "P₁V₁ = P₂V₂ → P₂ = 2×3/1 = 6 atm",
    },
    {
        "subject": "Chemistry", "topic": "Molar Mass",
        "q": "Molar mass of H₂SO₄ is:",
        "opts": ["98 g/mol", "96 g/mol", "100 g/mol", "94 g/mol"],
        "ans": 0,
        "exp": "2×1 + 32 + 4×16 = 2 + 32 + 64 = 98 g/mol",
    },
    {
        "subject": "Chemistry", "topic": "Electrochemistry",
        "q": "ΔG° for a cell with E° = 1.1 V and n = 2 electrons (F = 96485 C/mol) is approximately:",
        "opts": ["−212 kJ", "+212 kJ", "−106 kJ", "−424 kJ"],
        "ans": 0,
        "exp": "ΔG° = −nFE° = −2 × 96485 × 1.1 ≈ −212 267 J ≈ −212 kJ",
    },
    {
        "subject": "Chemistry", "topic": "Half-life",
        "q": "A radioactive substance has t₁/₂ = 10 days. Fraction remaining after 30 days:",
        "opts": ["1/8", "1/4", "1/2", "1/16"],
        "ans": 0,
        "exp": "n = 30/10 = 3 half-lives → (½)³ = 1/8",
    },
    # ── Mathematics ──────────────────────────────────────────────────────────
    {
        "subject": "Mathematics", "topic": "Calculus",
        "q": "∫₀¹ x² dx =",
        "opts": ["1/3", "1/2", "1/4", "2/3"],
        "ans": 0,
        "exp": "[x³/3]₀¹ = 1/3 − 0 = 1/3",
    },
    {
        "subject": "Mathematics", "topic": "Algebra",
        "q": "Roots of x² − 5x + 6 = 0 are:",
        "opts": ["2 and 3", "1 and 6", "−2 and −3", "5 and 1"],
        "ans": 0,
        "exp": "Discriminant = 25 − 24 = 1 → x = (5 ± 1)/2 = 3 or 2",
    },
    {
        "subject": "Mathematics", "topic": "Probability",
        "q": "A fair die is rolled. P(even number) =",
        "opts": ["1/2", "1/3", "2/3", "1/6"],
        "ans": 0,
        "exp": "Even outcomes: 2, 4, 6 → P = 3/6 = 1/2",
    },
    {
        "subject": "Mathematics", "topic": "Vectors",
        "q": "If |A| = 3 and |B| = 4 and A·B = 0, then |A + B| =",
        "opts": ["5", "7", "1", "12"],
        "ans": 0,
        "exp": "|A + B| = √(|A|² + |B|²) = √(9 + 16) = √25 = 5 (perpendicular vectors)",
    },
]


def _print_q(idx: int, total: int, q: dict) -> None:
    print(f"\n{Fore.YELLOW}Q{idx + 1}/{total} [{q['subject']} — {q['topic']}]{Style.RESET_ALL}")
    print(f"  {q['q']}")
    for i, opt in enumerate(q['opts']):
        label = chr(ord('A') + i)
        print(f"    {Fore.CYAN}{label}{Style.RESET_ALL}. {opt}")
    print(f"  {Fore.WHITE}Enter A/B/C/D or press Enter to skip:{Style.RESET_ALL} ", end="", flush=True)


def run_mock_test(num_questions: int = 20, time_limit_min: int = 40) -> str:
    """Run an interactive mock test. Returns a results summary string."""
    available = QUESTIONS.copy()
    random.shuffle(available)
    selected = available[:min(num_questions, len(available))]

    print(f"\n{Fore.CYAN}{'═'*56}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}  JEE ADVANCED MOCK TEST — {len(selected)} questions{Style.RESET_ALL}")
    print(f"{Fore.CYAN}  Scoring: +4 correct | −1 wrong | 0 skipped{Style.RESET_ALL}")
    print(f"{Fore.CYAN}  Time limit: {time_limit_min} minutes{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'═'*56}{Style.RESET_ALL}")
    print(f"  Press Enter to start…", end="", flush=True)
    input()

    start_time = time.time()
    deadline   = start_time + time_limit_min * 60

    results = []  # (question, user_ans_idx, correct_idx, time_sec)

    for idx, q in enumerate(selected):
        elapsed = time.time() - start_time
        remaining = deadline - time.time()
        if remaining <= 0:
            print(f"\n{Fore.RED}  ⏰ Time's up!{Style.RESET_ALL}")
            break

        mins_left = int(remaining // 60)
        secs_left = int(remaining % 60)
        print(f"\n  {Fore.WHITE}Time remaining: {mins_left:02d}:{secs_left:02d}{Style.RESET_ALL}", end="")
        _print_q(idx, len(selected), q)

        q_start = time.time()
        try:
            raw = input().strip().upper()
        except (EOFError, KeyboardInterrupt):
            raw = ""
        q_time = time.time() - q_start

        if raw in ("A", "B", "C", "D"):
            user_idx = ord(raw) - ord("A")
        else:
            user_idx = None  # skipped

        results.append({
            "q":        q,
            "user_idx": user_idx,
            "correct":  user_idx == q["ans"] if user_idx is not None else False,
            "skipped":  user_idx is None,
            "time_sec": q_time,
        })

        # Immediate feedback
        if user_idx is None:
            print(f"  {Fore.YELLOW}Skipped.{Style.RESET_ALL}")
        elif user_idx == q["ans"]:
            print(f"  {Fore.GREEN}✔ Correct! +4{Style.RESET_ALL}")
        else:
            correct_label = chr(ord("A") + q["ans"])
            print(f"  {Fore.RED}✗ Wrong. Correct: {correct_label}. −1{Style.RESET_ALL}")

    return _build_results(results, time.time() - start_time)


def _build_results(results: list, total_time_sec: float) -> str:
    n_correct = sum(1 for r in results if r["correct"])
    n_wrong   = sum(1 for r in results if not r["correct"] and not r["skipped"])
    n_skip    = sum(1 for r in results if r["skipped"])
    score     = n_correct * 4 - n_wrong * 1
    max_score = len(results) * 4
    pct       = score / max_score * 100 if max_score else 0
    total_min = int(total_time_sec // 60)
    total_sec = int(total_time_sec % 60)

    # Subject breakdown
    subj_stats: dict = {}
    for r in results:
        s = r["q"]["subject"]
        if s not in subj_stats:
            subj_stats[s] = {"correct": 0, "wrong": 0, "skip": 0}
        if r["correct"]:
            subj_stats[s]["correct"] += 1
        elif r["skipped"]:
            subj_stats[s]["skip"] += 1
        else:
            subj_stats[s]["wrong"] += 1

    lines = [
        "",
        "═"*56,
        "  MOCK TEST RESULTS",
        "═"*56,
        f"  Attempted : {len(results)} questions",
        f"  Correct   : {n_correct}  (+{n_correct*4} pts)",
        f"  Wrong     : {n_wrong}   (−{n_wrong} pts)",
        f"  Skipped   : {n_skip}",
        f"  Score     : {score}/{max_score}  ({pct:.1f}%)",
        f"  Time      : {total_min:02d}:{total_sec:02d}",
        "",
        "  Subject Breakdown:",
    ]
    for subj, st in subj_stats.items():
        tot = st['correct'] + st['wrong'] + st['skip']
        acc = st['correct'] / (tot - st['skip']) * 100 if (tot - st['skip']) > 0 else 0
        lines.append(f"    {subj:<14} ✔{st['correct']}  ✗{st['wrong']}  ○{st['skip']}  {acc:.0f}%")

    # Rough rank estimate
    if pct >= 90:
        rank = "Top 1% — Excellent! 🎯"
    elif pct >= 75:
        rank = "Top 5% — Very Good"
    elif pct >= 60:
        rank = "Top 15% — Good, keep practising"
    elif pct >= 45:
        rank = "Top 30% — Needs improvement"
    else:
        rank = "Below 30% — Focus on fundamentals"
    lines.append("")
    lines.append(f"  Performance estimate: {rank}")
    lines.append("═"*56)

    # Detailed review
    lines.append("\n  ── Review (wrong/skipped) ──────────────────────────")
    shown = 0
    for r in results:
        if not r["correct"]:
            q = r["q"]
            correct_label = chr(ord("A") + q["ans"])
            lines.append(f"\n  Q: {q['q'][:70]}")
            if r["skipped"]:
                lines.append(f"     Skipped → Answer: {correct_label}. {q['opts'][q['ans']]}")
            else:
                user_label = chr(ord("A") + r["user_idx"])
                lines.append(f"     Your answer: {user_label}  |  Correct: {correct_label}. {q['opts'][q['ans']]}")
            lines.append(f"     Explanation: {q['exp']}")
            shown += 1
            if shown >= 10:
                lines.append(f"  … (showing first 10 review items)")
                break

    if shown == 0:
        lines.append("  Perfect score — nothing to review!")

    return "\n".join(lines)
