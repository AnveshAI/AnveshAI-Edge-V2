"""
====================
AnveshAI Edge — v2
====================
Terminal-based offline-first AI tutor for JEE Advanced.
Correctness-First pipeline: deterministic engines → LLM explanation.

Routing: /commands → instant handler
         Arithmetic → math_engine (AST)
         Advanced math → SymPy + LLM
         Physics / Chemistry → deterministic solver + LLM
         Logic → inference engine + LLM
         Knowledge → BM25 KB + LLM
         Conversation → pattern rules + LLM

Commands:
    /help        → list commands
    /history     → last 10 interactions
    /clear       → clear conversation history
    /test        → JEE mock test (20 questions, scored)
    /formulas    → full formula sheet
    /formulas p  → physics formulas only
    /formulas c  → chemistry formulas only
    /formulas m  → math formulas only
    /hint        → level-1 hint for last question
    /hint2       → level-2 hint (+ formula)
    /hint3       → level-3 hint (+ first step)
    /review      → spaced repetition review session
    /progress    → topic-level progress summary
    /benchmark   → full benchmark report
    /exit        → quit
"""

import sys

try:
    from colorama import init as colorama_init, Fore, Style
    colorama_init(autoreset=True)
except ImportError:
    class _NoColor:
        def __getattr__(self, _): return ""
    Fore = Style = _NoColor()

from router              import classify_intent
from math_engine         import evaluate as math_evaluate
from advanced_math_engine import solve as advanced_math_solve
from knowledge_engine    import KnowledgeEngine
from conversation_engine import ConversationEngine
from llm_engine          import LLMEngine, MATH_SYSTEM_PROMPT, MATH_TEMPERATURE, CHAT_SYSTEM_PROMPT
from reasoning_engine    import ReasoningEngine
from inference_engine    import InferenceEngine
from physics_engine      import PhysicsEngine
from chemistry_engine    import ChemistryEngine
from memory              import (
    initialize_db, save_interaction, format_history, clear_history,
    save_progress, get_progress_summary, get_weak_topics, get_due_topics,
)
from mock_test           import run_mock_test
from formula_sheet       import get_formula_sheet
from hint_engine         import get_hints
from spaced_repetition   import run_review_session
from benchmark_report    import generate_report


BANNER = r"""
╔══════════════════════════════════════════════════════╗
║     ___                    __   ___   ____           ║
║    / _ | ___ _  _____ ___ / /  / _ | /  _/           ║
║   / __ |/ _ \ |/ / -_|_-</ _ \/ __ |_/ /             ║
║  /_/ |_/_//_/___/\__/___/_//_/_/ |_/___/  EDGE  v2   ║
║                                                      ║
╚══════════════════════════════════════════════════════╝
"""

HELP_TEXT = """
  Available commands:
    /help        — show this help message
    /history     — display last 10 conversation entries
    /clear       — clear conversation history
    /exit        — quit AnveshAI Edge

    /test        — JEE Advanced mock test (20 questions, +4/−1 scoring)
    /formulas    — full formula sheet (Physics + Chemistry + Math)
    /formulas p  — Physics formulas only
    /formulas c  — Chemistry formulas only
    /formulas m  — Math formulas only
    /hint        — conceptual hint for your last question
    /hint2       — hint + relevant formula
    /hint3       — hint + formula + first step
    /review      — spaced repetition review session (SM-2 algorithm)
    /progress    — topic-level accuracy summary
    /benchmark   — full JEE performance report

  How to use:
    • Advanced math  →  symbolic engine computes the EXACT answer,
                        LLM explains step-by-step working

      Calculus:
        "integrate x^2 sin(x)"
        "definite integral of x^2 from 0 to 3"
        "derivative of x^3 + 2x"
        "second derivative of sin(x) * e^x"
        "limit of sin(x)/x as x approaches 0"

      Algebra & equations:
        "solve x^2 - 5x + 6 = 0"
        "solve 2x + 3 = 7"

      Differential equations:
        "solve differential equation y'' + y = 0"
        "solve ode dy/dx = y"

      Series & transforms:
        "taylor series of e^x around 0 order 6"
        "laplace transform of sin(t)"
        "inverse laplace of 1/(s^2 + 1)"
        "fourier transform of exp(-x^2)"

      Matrices:
        "determinant of [[1,2],[3,4]]"
        "inverse matrix [[2,1],[5,3]]"
        "eigenvalue [[4,1],[2,3]]"
        "rank of matrix [[1,2,3],[4,5,6]]"

      Symbolic manipulation:
        "factor x^3 - 8"
        "simplify (x^2 - 1)/(x - 1)"
        "expand (x + y)^4"
        "partial fraction 1/(x^2 - 1)"

      Number theory:
        "gcd of 48 and 18"
        "lcm of 12 and 15"
        "prime factorization of 360"
        "17 mod 5"
        "modular inverse of 3 mod 7"

      Statistics:
        "mean of 2, 4, 6, 8, 10"
        "standard deviation of 1, 2, 3, 4, 5"

      Combinatorics:
        "factorial of 10"
        "binomial coefficient 10 choose 3"
        "permutation 6 P 2"

      Summations:
        "sum of k^2 for k from 1 to 10"
        "summation of 1/n^2 for n from 1 to infinity"

      Complex numbers:
        "real part of 3 + 4*I"
        "modulus of 3 + 4*I"

    • Physics  →  deterministic formula engine computes EXACT answer, LLM explains
        Kinematics:
          "A car starts from rest and accelerates at 3 m/s² for 5 seconds. Find the final velocity."
          "How far does an object travel at 20 m/s for 10 s?"
        Dynamics & Forces:
          "Find the force on a 5 kg mass with acceleration 4 m/s²"
          "Coefficient of friction 0.3, mass 10 kg — find friction force"
        Energy & Work:
          "Calculate kinetic energy of 2 kg moving at 6 m/s"
          "Work done by 50 N force over 8 m"
          "Potential energy of 3 kg mass at height 10 m"
        Electricity:
          "Voltage 12V, resistance 4Ω — find current (Ohm's law)"
          "Electric power: current 3A, voltage 9V"
        Waves & Optics:
          "Wave speed: frequency 500 Hz, wavelength 0.68 m"
          "Photon energy: frequency 6e14 Hz"
          "Snell's law: n1=1, n2=1.5, angle 30°"
        Thermodynamics:
          "Specific heat of water 4200 J/kg·K, mass 2 kg, ΔT 10°C — find heat"
        Fluid & Pressure:
          "Pressure: force 200N, area 0.5m²"
          "Density of water 1000 kg/m³ at depth 5m — find fluid pressure"
        Circular & Gravitation:
          "Centripetal force: mass 2kg, speed 4m/s, radius 0.5m"
          "Gravitational force between two 1000kg masses 100m apart"
        Momentum:
          "Momentum of 5 kg object at 12 m/s"

    • Chemistry  →  deterministic chemistry engine + LLM explanation
        Moles & Molar Mass:
          "Molar mass of H2O"
          "How many moles in 18g of H2O?"
          "Molar mass of Ca(OH)2"
        pH & Acid-Base:
          "pH of 0.01 M HCl"
          "pH of 0.1 M NaOH"
          "Ka = 1.8e-5, concentration 0.1 M — find pH of weak acid"
        Molarity & Dilution:
          "0.5 mol in 2L — find molarity"
          "Dilution: C1=2M, V1=50mL, C2=0.5M — find V2"
        Gas Laws:
          "Boyle's law: P1=1atm, V1=4L, P2=2atm — find V2"
          "Charles's law: V1=2L, T1=300K, T2=450K"
          "Ideal gas law: P=101325Pa, V=0.0224m³, T=273K — find n"
          "Combined gas law: P1=1, V1=2, T1=300, P2=2, T2=400 — find V2"
        Percent Composition:
          "Percent composition of H2SO4"
        Half-life:
          "Half-life 5 days, after 15 days — fraction remaining"
        Calorimetry:
          "q = mcΔT: mass 100g, specific heat 4.18 J/g·°C, ΔT 20°C"
        Electrochemistry:
          "ΔG: E° = 1.1V, n = 2 electrons"

    • Logic / Inference  →  formal inference engine + LLM
        "If it rains then the ground is wet. It is raining. Therefore?"
        "All mammals are warm-blooded. A whale is a mammal. Therefore?"
        "Modus tollens: If P then Q. Not Q. What follows?"
        "Is the argument valid: if A implies B and B implies C, does A imply C?"

    • Arithmetic     →  computed instantly
        e.g.  "2 + 3 * (4 ^ 2)"

    • Knowledge      →  BM25 knowledge base first, then LLM with KB context
        e.g.  "What is quantum computing?"
              "Explain the Central Limit Theorem"
              "What causes climate change?"

    • Chat           →  pattern rules, then reasoning-guided LLM
        e.g.  "Hello!"
"""


def _print(text: str, color: str = "") -> None:
    print(f"{color}{text}{Style.RESET_ALL}" if color else text)


def _prompt() -> str:
    try:
        return input(f"\n{Fore.CYAN}You{Style.RESET_ALL} › ").strip()
    except (EOFError, KeyboardInterrupt):
        return "/exit"


def _respond(label: str, text: str) -> None:
    print(
        f"\n{Fore.GREEN}AnveshAI{Style.RESET_ALL} "
        f"[{Fore.YELLOW}{label}{Style.RESET_ALL}] › {text}"
    )


def _system(text: str) -> None:
    print(f"{Fore.MAGENTA}  {text}{Style.RESET_ALL}")


def compose_response(
    user_input: str,
    intent: str,
    knowledge_engine: KnowledgeEngine,
    conversation_engine: ConversationEngine,
    llm_engine: LLMEngine,
    reasoning_engine: ReasoningEngine,
    inference_engine: InferenceEngine,
    physics_engine: PhysicsEngine,
    chemistry_engine: ChemistryEngine,
) -> tuple[str, str]:
    """
    Route input through the full hierarchy.
    Returns (label, response_text).
    """

    # ── Simple arithmetic ─────────────────────────────────────────────────────
    if intent == "math":
        return "Math", math_evaluate(user_input)

    # ── Advanced math ─────────────────────────────────────────────────────────
    if intent == "advanced_math":
        success, result_str, _latex = advanced_math_solve(user_input)

        if success:
            _system(f"SymPy → {result_str}")
            _system("Reasoning engine v2: decomposing problem…")
            plan = reasoning_engine.analyze(user_input, intent, has_symbolic_result=True)
            _system(plan.summary())
            if plan.warnings:
                for w in plan.warnings:
                    _system(f"  ⚠  {w}")
            _system(f"Mode: {plan.reasoning_mode} — building prompt → LLM…")
            prompt = reasoning_engine.build_math_prompt(user_input, result_str, plan)
            explanation = llm_engine.generate(
                prompt,
                system_prompt=MATH_SYSTEM_PROMPT,
                temperature=MATH_TEMPERATURE,
            )
            full_response = (
                f"{result_str}\n\n"
                f"[{plan.reasoning_mode} | {plan.problem_type} | "
                f"confidence: {plan.confidence}]\n\n"
                f"{explanation}"
            )
            return "AdvMath+CoT+LLM", full_response

        else:
            _system(f"SymPy error: {result_str}")
            _system("Reasoning engine v2: building fallback chain-of-thought…")
            plan = reasoning_engine.analyze(user_input, intent)
            _system(plan.summary())
            prompt = reasoning_engine.build_math_fallback_prompt(
                user_input, plan, error_context=result_str
            )
            llm_response = llm_engine.generate(prompt)
            return "AdvMath+CoT", llm_response

    # ── Physics ───────────────────────────────────────────────────────────────
    if intent == "physics":
        success, result_str, formula_type = physics_engine.solve(user_input)

        if success:
            _system(f"Physics engine → {result_str}")
            _system("Reasoning engine: building physics explanation prompt…")
            plan = reasoning_engine.analyze(user_input, intent, has_symbolic_result=True)
            _system(plan.summary())
            prompt = reasoning_engine.build_physics_prompt(user_input, result_str, formula_type, plan)
            explanation = llm_engine.generate(
                prompt,
                system_prompt=MATH_SYSTEM_PROMPT,
                temperature=MATH_TEMPERATURE,
            )
            full_response = (
                f"{result_str}\n\n"
                f"[{formula_type} | confidence: {plan.confidence}]\n\n"
                f"{explanation}"
            )
            return "Physics+CoT+LLM", full_response

        else:
            _system(f"Physics engine: {result_str}")
            _system("Reasoning engine: building fallback physics prompt…")
            plan = reasoning_engine.analyze(user_input, intent)
            _system(plan.summary())
            prompt = reasoning_engine.build_physics_fallback_prompt(
                user_input, plan, error_context=result_str
            )
            return "Physics+CoT", llm_engine.generate(prompt)

    # ── Chemistry ─────────────────────────────────────────────────────────────
    if intent == "chemistry":
        success, result_str, chem_type = chemistry_engine.solve(user_input)

        if success:
            _system(f"Chemistry engine → {result_str}")
            _system("Reasoning engine: building chemistry explanation prompt…")
            plan = reasoning_engine.analyze(user_input, intent, has_symbolic_result=True)
            _system(plan.summary())
            prompt = reasoning_engine.build_chemistry_prompt(user_input, result_str, chem_type, plan)
            explanation = llm_engine.generate(
                prompt,
                system_prompt=MATH_SYSTEM_PROMPT,
                temperature=MATH_TEMPERATURE,
            )
            full_response = (
                f"{result_str}\n\n"
                f"[{chem_type} | confidence: {plan.confidence}]\n\n"
                f"{explanation}"
            )
            return "Chemistry+CoT+LLM", full_response

        else:
            _system(f"Chemistry engine: {result_str}")
            _system("Reasoning engine: building fallback chemistry prompt…")
            plan = reasoning_engine.analyze(user_input, intent)
            _system(plan.summary())
            prompt = reasoning_engine.build_chemistry_fallback_prompt(
                user_input, plan, error_context=result_str
            )
            return "Chemistry+CoT", llm_engine.generate(prompt)

    # ── Logic / Inference ─────────────────────────────────────────────────────
    if intent == "logic":
        _system("Inference engine: parsing logical structure…")
        inf_result = inference_engine.infer(user_input)

        if inf_result.valid:
            _system(f"  ✔ Rule: {inf_result.rule_applied}")
            _system(f"  → Conclusion: {inf_result.conclusion}")
            return "Logic+Inference", inf_result.to_response()

        _system("  Inference engine: no rule matched — reasoning-guided LLM…")
        plan = reasoning_engine.analyze(user_input, intent)
        _system(plan.summary())
        kb_context = knowledge_engine.get_context(user_input)
        prompt = reasoning_engine.build_general_prompt(
            user_input, intent, kb_context, plan
        )
        return "Logic+CoT+LLM", llm_engine.generate(prompt)

    # ── Knowledge ─────────────────────────────────────────────────────────────
    if intent == "knowledge":
        _system("Knowledge engine v2: BM25 retrieval…")
        kb_response, kb_found = knowledge_engine.query(user_input)

        if kb_found:
            _system("  ✔ KB match found (BM25)")
            return "Knowledge", kb_response

        _system("  KB: no confident match — reasoning engine + LLM (with KB context)…")
        plan = reasoning_engine.analyze(user_input, intent)
        _system(plan.summary())
        kb_context = knowledge_engine.get_context(user_input, top_k=2)
        prompt = reasoning_engine.build_general_prompt(
            user_input, intent, kb_context, plan
        )
        return "LLM+CoT-KB", llm_engine.generate(prompt)

    # ── Conversation ──────────────────────────────────────────────────────────
    chat_response, pattern_matched = conversation_engine.respond(user_input)
    if pattern_matched:
        return "Chat", chat_response

    # Guard: very short or clearly nonsensical input — answer without LLM
    stripped = user_input.strip()
    words = [w for w in stripped.split() if w.isalpha()]
    if len(stripped) < 4 or (len(words) == 0 and len(stripped) < 20):
        return "Chat", "I'm not sure what you mean — could you rephrase or ask me a JEE question?"

    # Fallback: use a lightweight conversational prompt (no CoT / reasoning plan)
    _system("No pattern match — LLM chat fallback…")
    return "Chat", llm_engine.generate(
        user_input,
        system_prompt=CHAT_SYSTEM_PROMPT,
        temperature=0.85,
    )


def main() -> None:
    _print(BANNER, Fore.CYAN)
    _system("Initialising modules…")

    initialize_db()
    _system("✔  Memory (SQLite) ready")

    knowledge_engine    = KnowledgeEngine()
    _system(
        "✔  Knowledge base loaded (BM25, multi-passage synthesis)"
        if knowledge_engine.is_loaded()
        else "⚠  knowledge.txt not found"
    )

    conversation_engine = ConversationEngine()
    inference_engine    = InferenceEngine()
    physics_engine_obj  = PhysicsEngine()
    chemistry_engine_obj = ChemistryEngine()
    _system("✔  Conversation engine ready")
    _system("✔  Math engine ready (AST safe-eval)")
    _system("✔  Advanced math engine ready (SymPy — 31+ operations)")
    _system("✔  Physics engine ready (deterministic formula solver — 20 domains)")
    _system("✔  Chemistry engine ready (deterministic solver — moles, pH, gas laws, …)")
    _system("✔  Reasoning engine v2 ready (CoT + Tree-of-Thought + self-consistency)")
    _system("✔  Inference engine ready (modus ponens/tollens, syllogisms, propositional logic)")
    _system("✔  Intent router v2 ready (8-way classification)")

    llm_engine    = LLMEngine()
    reasoning_eng = ReasoningEngine()
    _system("✔  LLM engine ready (Qwen2.5-1.5B loads on first use)")

    _print(f"\n{Fore.WHITE}Type /help for commands or just start chatting!{Style.RESET_ALL}")

    last_question: str = ""   # track last question for /hint

    while True:
        user_input = _prompt()
        if not user_input:
            continue

        intent = classify_intent(user_input)

        # ── System commands ───────────────────────────────────────────────────
        if intent == "system":
            parts = user_input.lower().split()
            cmd   = parts[0]

            if cmd == "/exit":
                _print(f"\n{Fore.CYAN}Goodbye! Session closed.{Style.RESET_ALL}")
                sys.exit(0)

            elif cmd == "/history":
                _print(f"\n{Fore.YELLOW}── Conversation History ─────────────────────{Style.RESET_ALL}")
                _print(format_history())
                _print(f"{Fore.YELLOW}─────────────────────────────────────────────{Style.RESET_ALL}")

            elif cmd == "/clear":
                clear_history()
                _system("✔  Conversation history cleared.")

            elif cmd == "/help":
                _print(f"\n{Fore.YELLOW}── Help ──────────────────────────────────────{Style.RESET_ALL}")
                _print(HELP_TEXT)
                _print(f"{Fore.YELLOW}─────────────────────────────────────────────{Style.RESET_ALL}")

            # ── Mock test ─────────────────────────────────────────────────
            elif cmd == "/test":
                results = run_mock_test()
                _print(results, Fore.WHITE)

            # ── Formula sheet ─────────────────────────────────────────────
            elif cmd == "/formulas":
                subj_arg = parts[1] if len(parts) > 1 else None
                subj_map = {
                    "p": "physics", "ph": "physics", "physics": "physics",
                    "c": "chemistry", "ch": "chemistry", "chem": "chemistry", "chemistry": "chemistry",
                    "m": "math", "ma": "math", "math": "math", "maths": "math",
                }
                subject = subj_map.get(subj_arg, None) if subj_arg else None
                _print(get_formula_sheet(subject), Fore.WHITE)

            # ── Hint system ───────────────────────────────────────────────
            elif cmd in ("/hint", "/hint1"):
                if last_question:
                    _print(get_hints(last_question, level=1), Fore.YELLOW)
                else:
                    _respond("Hint", "Ask a question first, then use /hint for a hint.")

            elif cmd == "/hint2":
                if last_question:
                    _print(get_hints(last_question, level=2), Fore.YELLOW)
                else:
                    _respond("Hint", "Ask a question first, then use /hint2.")

            elif cmd == "/hint3":
                if last_question:
                    _print(get_hints(last_question, level=3), Fore.YELLOW)
                else:
                    _respond("Hint", "Ask a question first, then use /hint3.")

            # ── Spaced repetition review ──────────────────────────────────
            elif cmd == "/review":
                result_msg = run_review_session()
                _print(result_msg, Fore.WHITE)

            # ── Progress dashboard ────────────────────────────────────────
            elif cmd == "/progress":
                summary = get_progress_summary()
                if not summary:
                    _respond("Progress", "No progress data yet. Use /test or ask JEE questions.")
                else:
                    lines = [f"\n{Fore.YELLOW}── Progress Summary ───────────────────────{Style.RESET_ALL}"]
                    for subj, data in sorted(summary.items()):
                        a   = data['attempted']
                        c   = data['correct']
                        acc = data['accuracy']
                        lines.append(f"  {subj:<14} {c}/{a} correct  ({acc:.1f}%)")
                        for topic, td in sorted(data['topics'].items()):
                            tacc = td['correct']/td['attempted']*100 if td['attempted'] else 0
                            marker = "✔" if tacc >= 60 else "⚠"
                            lines.append(f"    {marker} {topic:<25} {td['correct']}/{td['attempted']}  ({tacc:.0f}%)")
                    _print("\n".join(lines))

            # ── Benchmark report ──────────────────────────────────────────
            elif cmd == "/benchmark":
                summary    = get_progress_summary()
                due_topics = get_due_topics()
                weak       = get_weak_topics()
                _print(generate_report(summary, due_topics, weak), Fore.WHITE)

            else:
                _respond("System", f"Unknown command '{user_input}'. Type /help.")
            continue

        # ── Compose response ──────────────────────────────────────────────────
        last_question = user_input   # track for /hint
        label, response = compose_response(
            user_input, intent, knowledge_engine,
            conversation_engine, llm_engine, reasoning_eng, inference_engine,
            physics_engine_obj, chemistry_engine_obj,
        )
        _respond(label, response)
        save_interaction(user_input, response)

        # ── Auto-track progress for physics/chemistry/math ────────────────
        if intent in ("physics", "chemistry", "advanced_math"):
            subj_map = {"physics": "Physics", "chemistry": "Chemistry", "advanced_math": "Mathematics"}
            subj  = subj_map[intent]
            topic = label.split("+")[0] if "+" in label else label
            is_correct = (
                "Could not" not in response
                and "Provide" not in response[:80]
                and response.strip() != ""
            )
            save_progress(subj, topic, is_correct, question=user_input)


if __name__ == "__main__":
    main()
