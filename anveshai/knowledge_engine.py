"""
Knowledge Engine v2 — BM25-ranked multi-passage retrieval.

Improvements over v1:
  · BM25 ranking (Okapi BM25) instead of raw keyword overlap
  · Multi-passage synthesis — retrieves top-K paragraphs and merges them
  · Phrase-level matching with bonus scoring
  · TF-IDF-style IDF weighting for rare/common term discrimination
  · Dynamic MIN_RELEVANCE threshold based on query complexity
  · Zero external dependencies (pure stdlib + math)
"""

from __future__ import annotations

import math
import os
import re
from collections import Counter
from typing import List, Tuple


KNOWLEDGE_FILE = os.path.join(os.path.dirname(__file__), "knowledge.txt")

# BM25 hyper-parameters
K1: float = 1.5   # term saturation
B:  float = 0.75  # length normalisation

# How many top passages to retrieve & synthesise
TOP_K: int = 3

# Minimum BM25 score for a passage to be considered a real match
MIN_BM25_SCORE: float = 1.0

STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "shall",
    "should", "may", "might", "must", "can", "could", "to", "of", "in",
    "on", "at", "by", "for", "with", "about", "against", "between", "into",
    "through", "during", "before", "after", "above", "below", "from",
    "up", "down", "out", "off", "over", "under", "again", "and", "but",
    "or", "nor", "so", "yet", "both", "either", "neither", "not", "no",
    "what", "which", "who", "whom", "this", "that", "these", "those",
    "i", "me", "my", "myself", "we", "our", "you", "your", "he", "she",
    "it", "they", "them", "their", "tell", "explain", "describe", "give",
    "me", "some", "information", "about", "please", "could", "how", "when",
    "where", "why", "let",
}

# ─────────────────────────────────────────────────────────────────────────────
# Text processing helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_paragraphs(filepath: str) -> List[str]:
    if not os.path.exists(filepath):
        return []
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    raw = re.split(r"\n\s*\n", content.strip())
    return [p.strip() for p in raw if len(p.strip()) > 10]


def _tokenize(text: str) -> List[str]:
    words = re.findall(r"\b[a-z]{2,}\b", text.lower())
    return [w for w in words if w not in STOP_WORDS]


def _extract_phrases(text: str, n: int = 2) -> List[str]:
    tokens = _tokenize(text)
    return [" ".join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]


def _strip_question_prefixes(text: str) -> str:
    prefixes = [
        "what is", "what are", "who is", "who are", "explain", "define",
        "tell me about", "describe", "how does", "why is", "when was",
        "where is", "history of", "meaning of", "knowledge:", "knowledge :",
        "learn about", "facts about", "information about", "can you explain",
        "could you explain", "please explain", "give me information about",
    ]
    lowered = text.lower().strip()
    for prefix in prefixes:
        if lowered.startswith(prefix):
            remainder = text[len(prefix):].strip(" ?.,")
            if remainder:
                return remainder
    return text


# ─────────────────────────────────────────────────────────────────────────────
# BM25 index
# ─────────────────────────────────────────────────────────────────────────────

class BM25Index:
    """Okapi BM25 retrieval index over a list of paragraphs."""

    def __init__(self, paragraphs: List[str]):
        self.paragraphs   = paragraphs
        self.n_docs       = len(paragraphs)
        self._token_lists = [_tokenize(p) for p in paragraphs]
        self._tf_lists    = [Counter(tl) for tl in self._token_lists]
        self._doc_lengths = [len(tl) for tl in self._token_lists]
        self._avgdl       = (
            sum(self._doc_lengths) / self.n_docs if self.n_docs else 1.0
        )
        self._idf         = self._compute_idf()

    def _compute_idf(self) -> dict[str, float]:
        idf: dict[str, float] = {}
        for token_list in self._token_lists:
            for tok in set(token_list):
                idf[tok] = idf.get(tok, 0) + 1
        result: dict[str, float] = {}
        for tok, df in idf.items():
            result[tok] = math.log(
                (self.n_docs - df + 0.5) / (df + 0.5) + 1
            )
        return result

    def score(self, query_tokens: List[str], doc_idx: int) -> float:
        tf_map = self._tf_lists[doc_idx]
        dl     = self._doc_lengths[doc_idx]
        score  = 0.0
        for tok in query_tokens:
            if tok not in tf_map:
                continue
            idf_val = self._idf.get(tok, 0.0)
            tf_val  = tf_map[tok]
            numerator   = tf_val * (K1 + 1)
            denominator = tf_val + K1 * (1 - B + B * dl / self._avgdl)
            score += idf_val * (numerator / denominator)
        return score

    def phrase_bonus(self, phrase: str, doc_idx: int) -> float:
        """Extra score if a phrase appears verbatim in the paragraph."""
        if phrase in self.paragraphs[doc_idx].lower():
            return 2.0
        return 0.0

    def retrieve(
        self, query: str, top_k: int = TOP_K
    ) -> List[Tuple[float, str]]:
        clean     = _strip_question_prefixes(query)
        q_tokens  = _tokenize(clean)
        q_phrases = _extract_phrases(clean, 2)

        if not q_tokens:
            return []

        scores: List[Tuple[float, int]] = []
        for idx in range(self.n_docs):
            s = self.score(q_tokens, idx)
            for phrase in q_phrases:
                s += self.phrase_bonus(phrase, idx)
            scores.append((s, idx))

        scores.sort(reverse=True)
        return [
            (s, self.paragraphs[idx])
            for s, idx in scores[:top_k]
            if s >= MIN_BM25_SCORE
        ]


# ─────────────────────────────────────────────────────────────────────────────
# Multi-passage synthesiser
# ─────────────────────────────────────────────────────────────────────────────

def _synthesise(passages: List[Tuple[float, str]], query: str) -> str:
    """
    Merge the top-K retrieved passages into a coherent response.
    Deduplicates sentences that appear across passages.
    """
    if not passages:
        return ""

    if len(passages) == 1:
        return passages[0][1]

    seen_sentences: set[str] = set()
    merged_sentences: List[str] = []

    for _score, para in passages:
        sentences = re.split(r"(?<=[.!?])\s+", para.strip())
        for sent in sentences:
            normalised = sent.strip().lower()
            if normalised not in seen_sentences and len(sent.strip()) > 10:
                seen_sentences.add(normalised)
                merged_sentences.append(sent.strip())

    return " ".join(merged_sentences)


# ─────────────────────────────────────────────────────────────────────────────
# Public interface
# ─────────────────────────────────────────────────────────────────────────────

class KnowledgeEngine:
    """
    BM25-powered local knowledge retrieval.

    Usage:
        ke = KnowledgeEngine()
        response, found = ke.query("What is quantum computing?")
    """

    def __init__(self, knowledge_file: str = KNOWLEDGE_FILE):
        self._paragraphs = _load_paragraphs(knowledge_file)
        self._loaded     = len(self._paragraphs) > 0
        self._index      = BM25Index(self._paragraphs) if self._loaded else None

    def is_loaded(self) -> bool:
        return self._loaded

    def query(self, user_input: str, top_k: int = TOP_K) -> Tuple[str, bool]:
        """
        Find the most relevant passage(s) for the given query using BM25.

        Returns:
            (response_text, found)
            found=True  → high-confidence match(es) retrieved
            found=False → no confident match; caller should escalate to LLM
        """
        if not self._loaded or self._index is None:
            return ("Knowledge base unavailable. Ensure 'knowledge.txt' exists.", False)

        hits = self._index.retrieve(user_input, top_k=top_k)

        if not hits:
            return ("", False)

        response = _synthesise(hits, user_input)
        return (response, True)

    def get_context(self, user_input: str, top_k: int = 2) -> str:
        """
        Return up to top_k relevant passages as a context block (for LLM prompts).
        Returns empty string if nothing relevant is found.
        """
        if not self._loaded or self._index is None:
            return ""
        hits = self._index.retrieve(user_input, top_k=top_k)
        if not hits:
            return ""
        return "\n\n".join(para for _, para in hits)
