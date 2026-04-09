"""
Benchmark Report — JEE performance analytics.
/benchmark — full report
"""

from typing import List


def _bar(value: float, max_val: float = 100, width: int = 20) -> str:
    filled = int(value / max_val * width) if max_val > 0 else 0
    return "[" + "█" * filled + "░" * (width - filled) + f"] {value:.1f}%"


def generate_report(summary: dict, due_topics: List[dict], weak_topics: list) -> str:
    lines = [
        "",
        "═"*60,
        "  ANVESHAI BENCHMARK REPORT",
        "═"*60,
    ]

    if not summary:
        lines.append("\n  No progress data yet.")
        lines.append("  Use /test to attempt questions, or ask JEE-style problems.")
        lines.append("  Performance will be tracked automatically.\n")
        lines.append("═"*60)
        return "\n".join(lines)

    # ── Subject overview ───────────────────────────────────────────────────
    lines.append("\n  Subject Performance")
    lines.append("  " + "─"*56)
    total_attempted = 0
    total_correct   = 0
    for subj, data in sorted(summary.items()):
        a   = data['attempted']
        c   = data['correct']
        acc = data['accuracy']
        total_attempted += a
        total_correct   += c
        mins = int(data['time_sec'] // 60)
        secs = int(data['time_sec'] % 60)
        lines.append(f"\n  {subj}")
        lines.append(f"    Attempted : {a} questions")
        lines.append(f"    Correct   : {c}  ({acc:.1f}%)")
        lines.append(f"    Time      : {mins:02d}:{secs:02d}")
        lines.append(f"    {_bar(acc)}")

    # ── Overall ───────────────────────────────────────────────────────────
    overall_acc = total_correct / total_attempted * 100 if total_attempted else 0
    lines.append(f"\n  {'─'*56}")
    lines.append(f"  OVERALL: {total_attempted} attempted, {total_correct} correct, {overall_acc:.1f}% accuracy")
    lines.append(f"  {_bar(overall_acc)}")

    # ── Topic-level breakdown ─────────────────────────────────────────────
    lines.append(f"\n\n  Topic Breakdown")
    lines.append("  " + "─"*56)
    for subj, data in sorted(summary.items()):
        if data['topics']:
            lines.append(f"\n  {subj}:")
            for topic, tdata in sorted(data['topics'].items()):
                a   = tdata['attempted']
                c   = tdata['correct']
                acc = c / a * 100 if a > 0 else 0
                bar_chars = int(acc / 100 * 10)
                bar = "█" * bar_chars + "░" * (10 - bar_chars)
                lines.append(f"    {topic:<25} [{bar}] {acc:5.1f}%  ({c}/{a})")

    # ── Weak topics ───────────────────────────────────────────────────────
    if weak_topics:
        lines.append(f"\n\n  Weak Topics (< 60% accuracy)")
        lines.append("  " + "─"*56)
        for subj, topic, acc in weak_topics[:8]:
            lines.append(f"    ⚠  {subj} / {topic:<25} {acc:.1f}%")
        lines.append("\n  Tip: Use /review to practise these topics with spaced repetition.")
    else:
        lines.append("\n  No weak topics (< 60%) — well done!")

    # ── Spaced repetition due ─────────────────────────────────────────────
    if due_topics:
        lines.append(f"\n\n  Spaced Repetition — {len(due_topics)} topic(s) due for review")
        lines.append("  " + "─"*56)
        for card in due_topics[:5]:
            lines.append(f"    {card['topic']:<30} last: {card['last_review']}  interval: {card['interval']}d")
        lines.append("\n  Use /review to start today's review session.")

    # ── JEE readiness estimate ─────────────────────────────────────────────
    lines.append(f"\n\n  JEE Readiness Estimate")
    lines.append("  " + "─"*56)
    if overall_acc >= 80:
        level = "Advanced / High rank"
        advice = "Keep this pace; focus on time management."
    elif overall_acc >= 65:
        level = "Good foundation"
        advice = "Revise weak topics and practise timed mocks."
    elif overall_acc >= 50:
        level = "Developing"
        advice = "Consolidate fundamentals, attempt /test daily."
    elif total_attempted > 0:
        level = "Needs focused effort"
        advice = "Start from formulas (/formulas), then solve step-by-step."
    else:
        level = "Not enough data"
        advice = "Attempt /test to generate your first report."
    lines.append(f"  Level  : {level}")
    lines.append(f"  Advice : {advice}")

    lines.append(f"\n{'═'*60}\n")
    return "\n".join(lines)
