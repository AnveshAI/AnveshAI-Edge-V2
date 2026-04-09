"""
Spaced Repetition Review (/review)
Uses SM-2 algorithm via memory.py cards.
Shows topic-focused review questions and updates intervals.
"""

import random
from typing import List, Optional

try:
    from colorama import Fore, Style
except ImportError:
    class _NC:
        def __getattr__(self, _): return ""
    Fore = Style = _NC()

from memory import get_weak_topics, update_sr_card, get_due_topics, get_progress_summary


# Review questions keyed by (subject, topic)
REVIEW_Q: dict = {
    ("Physics", "Kinematics"): [
        {
            "q": "A ball is thrown vertically upward at 20 m/s. Maximum height reached (g=10 m/s²)?",
            "ans": "20 m",
            "exp": "v²=u²−2gh → h = u²/(2g) = 400/20 = 20 m",
        },
        {
            "q": "A car travels at 60 m/s and decelerates at 3 m/s². Stopping distance?",
            "ans": "600 m",
            "exp": "v²=u²−2as → 0=3600−6s → s=600 m",
        },
    ],
    ("Physics", "Energy"): [
        {
            "q": "Potential energy of a 5 kg body at 20 m height (g=10)?",
            "ans": "1000 J",
            "exp": "PE = mgh = 5×10×20 = 1000 J",
        },
    ],
    ("Physics", "Gravitation"): [
        {
            "q": "If g on Earth is 9.8 m/s² and radius doubles, g becomes?",
            "ans": "g/4 = 2.45 m/s²",
            "exp": "g ∝ 1/R² → if R→2R, g→g/4",
        },
    ],
    ("Physics", "Electrostatics"): [
        {
            "q": "Point charge of 2 µC in vacuum: electric field at 0.3 m?",
            "ans": "≈ 2.0×10⁵ N/C",
            "exp": "E = kq/r² = 8.99×10⁹×2×10⁻⁶/0.09 ≈ 2.0×10⁵ N/C",
        },
    ],
    ("Chemistry", "pH"): [
        {
            "q": "What is pH of 10⁻⁴ M HNO₃?",
            "ans": "pH = 4",
            "exp": "[H⁺] = 10⁻⁴ M → pH = −log(10⁻⁴) = 4",
        },
    ],
    ("Chemistry", "Moles"): [
        {
            "q": "How many molecules in 0.5 mol of H₂O?",
            "ans": "3.011×10²³",
            "exp": "N = nNₐ = 0.5×6.022×10²³ = 3.011×10²³",
        },
    ],
    ("Chemistry", "Gas Laws"): [
        {
            "q": "1 mol of gas at 0°C and 1 atm occupies what volume?",
            "ans": "22.4 L",
            "exp": "V = nRT/P = 1×0.082×273/1 ≈ 22.4 L (STP)",
        },
    ],
    ("Chemistry", "Half-life"): [
        {
            "q": "If t½ = 20 min, what fraction remains after 60 min?",
            "ans": "1/8",
            "exp": "n = 60/20 = 3 half-lives → (½)³ = 1/8",
        },
    ],
    ("Mathematics", "Calculus"): [
        {
            "q": "Evaluate ∫₀^π sin(x) dx",
            "ans": "2",
            "exp": "[−cos x]₀^π = −cos π + cos 0 = 1 + 1 = 2",
        },
        {
            "q": "Differentiate f(x) = x³ + 3x² − 5x + 1",
            "ans": "3x² + 6x − 5",
            "exp": "Power rule: 3x² + 6x − 5",
        },
    ],
    ("Mathematics", "Algebra"): [
        {
            "q": "Sum of roots of x² − 7x + 12 = 0?",
            "ans": "7",
            "exp": "Sum = −(−7)/1 = 7 (Vieta's formulas)",
        },
    ],
    ("Mathematics", "Probability"): [
        {
            "q": "Two fair coins tossed. P(both heads)?",
            "ans": "1/4",
            "exp": "P = 1/2 × 1/2 = 1/4",
        },
    ],
    ("Mathematics", "Vectors"): [
        {
            "q": "Angle between A=(1,0,0) and B=(0,1,0)?",
            "ans": "90°",
            "exp": "A·B=0 → cosθ=0 → θ=90°",
        },
    ],
}

# Fallback generic questions if no specific topic match
GENERIC_Q = [
    {
        "subject": "Physics", "topic": "Kinematics",
        "q": "A stone is dropped from 80 m. Time to reach ground (g=10 m/s²)?",
        "ans": "4 s",
        "exp": "h = ½gt² → t = √(2h/g) = √16 = 4 s",
    },
    {
        "subject": "Chemistry", "topic": "Moles",
        "q": "Moles in 36 g of H₂O (M=18)?",
        "ans": "2 mol",
        "exp": "n = 36/18 = 2 mol",
    },
    {
        "subject": "Mathematics", "topic": "Calculus",
        "q": "Derivative of sin(x)·cos(x)?",
        "ans": "cos(2x)",
        "exp": "Product rule: cos²x − sin²x = cos 2x",
    },
]


def _get_question_for_topic(subject: str, topic: str) -> Optional[dict]:
    key = (subject, topic)
    candidates = REVIEW_Q.get(key)
    if candidates:
        q = random.choice(candidates).copy()
        q['subject'] = subject
        q['topic']   = topic
        return q
    return None


def run_review_session() -> str:
    """Run an interactive spaced repetition review session."""
    due = get_due_topics()
    weak = get_weak_topics(threshold=60.0, min_attempts=1)

    # Build review queue: due SR cards first, then weak topics
    review_queue: List[dict] = []
    seen_topics: set = set()

    for card in due:
        q = _get_question_for_topic("Physics",     card['topic']) or \
            _get_question_for_topic("Chemistry",   card['topic']) or \
            _get_question_for_topic("Mathematics", card['topic'])
        if q and card['topic'] not in seen_topics:
            q['_sr_topic'] = card['topic']
            review_queue.append(q)
            seen_topics.add(card['topic'])

    for subj, topic, acc in weak:
        if topic not in seen_topics:
            q = _get_question_for_topic(subj, topic)
            if q:
                q['_sr_topic'] = topic
                review_queue.append(q)
                seen_topics.add(topic)

    if not review_queue:
        # Nothing targeted — add generic review questions
        review_queue = [g.copy() for g in GENERIC_Q]
        for q in review_queue:
            q['_sr_topic'] = q['topic']

    random.shuffle(review_queue)
    review_queue = review_queue[:10]

    print(f"\n{Fore.CYAN}{'═'*56}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}  SPACED REPETITION REVIEW — {len(review_queue)} questions{Style.RESET_ALL}")
    print(f"{Fore.CYAN}  Rate yourself: 5=Easy, 4=Correct, 3=Hard but correct,{Style.RESET_ALL}")
    print(f"{Fore.CYAN}  2=Incorrect, 1=Barely recalled, 0=Blank{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'═'*56}{Style.RESET_ALL}")

    correct_count = 0
    for idx, q in enumerate(review_queue):
        print(f"\n{Fore.YELLOW}Q{idx+1}/{len(review_queue)} [{q.get('subject','?')} — {q.get('topic','?')}]{Style.RESET_ALL}")
        print(f"  {q['q']}")
        print(f"  {Fore.WHITE}Your answer (or Enter to reveal):{Style.RESET_ALL} ", end="", flush=True)
        try:
            user_ans = input().strip()
        except (EOFError, KeyboardInterrupt):
            user_ans = ""

        print(f"  {Fore.GREEN}Answer: {q['ans']}{Style.RESET_ALL}")
        print(f"  {Fore.WHITE}Explanation: {q['exp']}{Style.RESET_ALL}")

        print(f"  Rate yourself 0-5: ", end="", flush=True)
        try:
            rating_raw = input().strip()
            quality = int(rating_raw) if rating_raw.isdigit() and 0 <= int(rating_raw) <= 5 else 3
        except (EOFError, KeyboardInterrupt, ValueError):
            quality = 3

        # Update SM-2 card
        update_sr_card(q['_sr_topic'], quality)
        if quality >= 3:
            correct_count += 1
            print(f"  {Fore.GREEN}✔ Recorded.{Style.RESET_ALL}")
        else:
            print(f"  {Fore.RED}↩ Marked for earlier re-review.{Style.RESET_ALL}")

    lines = [
        f"\n{'═'*56}",
        f"  Review session complete!",
        f"  {correct_count}/{len(review_queue)} topics answered correctly.",
        f"  SM-2 intervals updated for all topics.",
        f"  Type /benchmark to see your full performance report.",
        f"{'═'*56}",
    ]
    return "\n".join(lines)
