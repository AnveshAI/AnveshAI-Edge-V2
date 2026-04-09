"""
LLM Engine — local Qwen2.5-1.5B-Instruct fallback via llama-cpp-python.

This is the bottom layer of the AnveshAI hierarchy:

    Math          →  math_engine         (instant, rule-based)
    Knowledge     →  knowledge_engine    (keyword retrieval from knowledge.txt)
      └─ no match →  LLMEngine.generate  (Qwen2.5-1.5B)
    Conversation  →  conversation_engine (pattern matching from conversation.txt)
      └─ no match →  LLMEngine.generate  (Qwen2.5-1.5B)

Model: Qwen/Qwen2.5-1.5B-Instruct  (Q4_K_M GGUF, ~1 GB)
    ─ 3× more parameters than 0.5B — noticeably better reasoning & fluency
    ─ Runs entirely on CPU via llama.cpp
    ─ Downloaded once into ~/.cache/huggingface/ on first use
    ─ Loaded LAZILY: the model only loads when first needed,
      keeping startup instant.
"""

MODEL_REPO   = "Qwen/Qwen2.5-1.5B-Instruct-GGUF"
MODEL_FILE   = "qwen2.5-1.5b-instruct-q4_k_m.gguf"

SYSTEM_PROMPT = (
    "You are AnveshAI Edge, an expert AI tutor specialising in JEE Advanced "
    "(Physics, Chemistry, and Mathematics). "
    "Your answers must meet the rigour and depth expected at JEE Advanced level. "
    "Always show complete, step-by-step working. Use standard JEE notation. "
    "State the relevant formula or principle before applying it. "
    "Do not repeat the question back. "
    "If you are unsure about something, say so clearly. "
    "Prefer concise, exam-focused explanations over verbose prose."
)

MATH_SYSTEM_PROMPT = (
    "You are an expert JEE Advanced mathematics tutor. "
    "You will be given a VERIFIED ANSWER computed by a symbolic engine. "
    "That answer is 100% correct — do NOT change it, do NOT recompute it. "
    "Your ONLY job is to explain, step by step, HOW a JEE Advanced student "
    "would work through the problem and arrive at that exact answer. "
    "Use JEE Advanced standard techniques (e.g. substitution, IBP, "
    "L'Hopital, cofactor expansion, characteristic equation, etc.). "
    "Every step must lead logically toward the verified answer. "
    "State the verified answer word-for-word at the end of your explanation. "
    "Keep the explanation concise and exam-focused."
)

CHAT_SYSTEM_PROMPT = (
    "You are AnveshAI, a friendly JEE Advanced study assistant. "
    "When the user sends a casual or non-academic message, reply briefly and "
    "naturally in 1-2 sentences — like a helpful study buddy, not an academic paper. "
    "Never analyse casual phrases or exclamations as if they are problems. "
    "If you cannot understand the message, politely ask what they need help with."
)

MAX_TOKENS  = 1024   # enough for detailed explanations and step-by-step answers
TEMPERATURE = 0.7
MATH_TEMPERATURE = 0.1   # near-deterministic for math explanations
TOP_P       = 0.9
N_CTX       = 16384  # match model's trained context (supports up to 32768)


class LLMEngine:
    """
    Lazy-loading wrapper around Qwen2.5-1.5B-Instruct (GGUF via llama.cpp).

    Usage:
        engine = LLMEngine()
        response = engine.generate("What is photosynthesis?")

    The GGUF model is downloaded from HuggingFace on the first call to
    generate() and cached locally. Every subsequent call reuses the
    in-memory model — no re-loading.
    """

    def __init__(self) -> None:
        self._llm   = None
        self._loaded: bool = False
        self._failed: bool = False
        self._fail_reason: str = ""

    def is_available(self) -> bool:
        """True once the model has loaded without error."""
        return self._loaded and not self._failed

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Download (first run only) and load the GGUF model into memory."""
        if self._loaded or self._failed:
            return

        try:
            print(
                f"\n  [LLM] Loading {MODEL_FILE} … "
                "(first run downloads ~1 GB, then cached locally)",
                flush=True,
            )

            from llama_cpp import Llama

            self._llm = Llama.from_pretrained(
                repo_id=MODEL_REPO,
                filename=MODEL_FILE,
                n_ctx=N_CTX,
                n_threads=4,   # use up to 4 CPU threads
                verbose=False,
            )

            self._loaded = True
            print("  [LLM] Qwen2.5-1.5B-Instruct ready\n", flush=True)

        except Exception as exc:
            self._failed = True
            self._fail_reason = str(exc)
            print(f"  [LLM] Failed to load: {exc}\n", flush=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        user_input: str,
        context: str = "",
        system_prompt: str = "",
        temperature: float = TEMPERATURE,
    ) -> str:
        """
        Generate a response using the local LLM.

        Args:
            user_input    : The user's message or question.
            context       : Optional retrieved text to inject as background.
            system_prompt : Override the default system prompt (e.g. for math).
            temperature   : Sampling temperature; use low values for math.

        Returns:
            The model's reply as a plain string.
        """
        self._load()

        if self._failed:
            return (
                "The local LLM is currently unavailable "
                f"({self._fail_reason}). "
                "Ensure 'llama-cpp-python' is installed and the model "
                "could be downloaded."
            )

        try:
            system_content = system_prompt if system_prompt else SYSTEM_PROMPT
            if context:
                system_content += f"\n\nRelevant background:\n{context}"

            messages = [
                {"role": "system", "content": system_content},
                {"role": "user",   "content": user_input},
            ]

            output = self._llm.create_chat_completion(
                messages=messages,
                max_tokens=MAX_TOKENS,
                temperature=temperature,
                top_p=TOP_P,
            )

            response: str = output["choices"][0]["message"]["content"]
            return response.strip()

        except Exception as exc:
            return f"LLM generation error: {exc}"
