"""The RAG pipeline: question -> retrieval -> grounded prompt -> answer + citations."""

from __future__ import annotations

import time
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field

from app.llm import LLM
from app.retrieval import Retriever
from app.vectorstore import SearchHit

SYSTEM_PROMPT = (
    "Ты — ассистент поддержки интернет-магазина «ТехноМаркет». "
    "Отвечай на вопрос покупателя ТОЛЬКО на основе фрагментов базы знаний, приведённых ниже. "
    "Если во фрагментах нет ответа — честно скажи, что информация не найдена, "
    "и предложи обратиться к оператору поддержки. "
    "Не выдумывай факты, сроки и суммы. "
    "Ссылайся на использованные фрагменты в формате [1], [2]. "
    "Отвечай кратко и по делу, на русском языке."
)

REFUSAL_TEXT = (
    "К сожалению, в базе знаний «ТехноМаркета» нет информации по этому вопросу. "
    "Я передам обращение оператору поддержки — он поможет разобраться."
)

# Pronouns and demonstratives that point at the previous turn
# ("а сколько это стоит?"). Deliberately excludes common question words such as
# "сколько" or conjunctions like "а": they appear in fully independent questions
# ("Сколько идёт доставка?"), and treating those as follow-ups would drag an
# unrelated topic into the search query.
_ANAPHORA = frozenset(
    {
        "это", "этот", "эта", "эти", "тот", "та", "те", "такой", "такая",
        "такие", "он", "она", "оно", "они", "его", "её", "ее", "их", "там",
        "туда", "тогда",
    }
)
# A question this short rarely carries a topic of its own ("А если брак?").
_SHORT_QUESTION_WORDS = 3


@dataclass(frozen=True)
class ChatTurn:
    """One completed message of the dialogue."""

    role: str  # "user" | "assistant"
    content: str


@dataclass(frozen=True)
class AnswerResult:
    answer: str
    hits: list[SearchHit]
    latency_ms: int
    refused: bool = False
    used_history: bool = False


def build_retrieval_query(question: str, history: Sequence[ChatTurn] = ()) -> str:
    """Decide what to search for, taking the previous turn into account.

    Follow-up questions ("а сколько это стоит?") are unsearchable on their own:
    the embedding has no topic to work with. Two cheap, deterministic signals
    say the question depends on context — it is very short, or it contains
    anaphora. In that case the previous user question is prepended so the search
    has a topic. Otherwise the question is used as-is, because blindly gluing
    turns together would drag an unrelated topic into the query.
    """
    if not history:
        return question
    words = question.lower().split()
    depends_on_context = len(words) <= _SHORT_QUESTION_WORDS or any(
        word.strip("?,.!«»()") in _ANAPHORA for word in words
    )
    if not depends_on_context:
        return question
    previous_user_turns = [turn.content for turn in history if turn.role == "user"]
    if not previous_user_turns:
        return question
    return f"{previous_user_turns[-1]} {question}"


def build_user_prompt(
    question: str,
    hits: list[SearchHit],
    history: Sequence[ChatTurn] = (),
) -> str:
    """Compose the user prompt: dialogue so far + question + numbered fragments."""
    blocks: list[str] = []
    for ref, hit in enumerate(hits, start=1):
        chunk = hit.chunk
        blocks.append(
            f"[{ref}] {chunk.title} (раздел: {chunk.category}, обновлено: {chunk.updated_at})\n"
            f"{chunk.text}"
        )
    context = "\n\n".join(blocks) if blocks else "(пусто)"

    parts: list[str] = []
    if history:
        dialogue = "\n".join(
            f"{'Покупатель' if turn.role == 'user' else 'Ассистент'}: {turn.content}"
            for turn in history
        )
        parts.append(f"ПРЕДЫДУЩИЙ ДИАЛОГ:\n{dialogue}\n")
    parts.append(f"ВОПРОС ПОКУПАТЕЛЯ: {question}\n\nФРАГМЕНТЫ БАЗЫ ЗНАНИЙ:\n{context}\n")
    parts.append("Сформулируй ответ покупателю.")
    return "\n".join(parts)


@dataclass
class _StreamState:
    """Bookkeeping for one streamed answer."""

    pieces: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "".join(self.pieces)



class RAGPipeline:
    """Ties retrieval and generation together. Pure logic, no I/O of its own."""

    def __init__(self, retriever: Retriever, llm: LLM, default_top_k: int = 5) -> None:
        self._retriever = retriever
        self._llm = llm
        self._default_top_k = default_top_k

    async def answer(
        self,
        question: str,
        top_k: int | None = None,
        audience: str | None = "customer",
        history: Sequence[ChatTurn] = (),
    ) -> AnswerResult:
        started = time.perf_counter()
        hits = await self._retrieve(question, top_k, audience, history)
        if not hits:
            # Relevance gate (or an empty index): refuse without paying for an LLM call.
            latency_ms = int((time.perf_counter() - started) * 1000)
            return AnswerResult(
                answer=REFUSAL_TEXT,
                hits=[],
                latency_ms=latency_ms,
                refused=True,
                used_history=bool(history),
            )
        prompt = build_user_prompt(question, hits, history)
        answer = await self._llm.complete(SYSTEM_PROMPT, prompt)
        latency_ms = int((time.perf_counter() - started) * 1000)
        return AnswerResult(
            answer=answer,
            hits=hits,
            latency_ms=latency_ms,
            used_history=bool(history),
        )

    async def answer_stream(
        self,
        question: str,
        top_k: int | None = None,
        audience: str | None = "customer",
        history: Sequence[ChatTurn] = (),
    ) -> AsyncIterator[dict]:
        """Yield the answer as it is generated.

        Event contract (consumed by the chat UI and by API clients):

        * ``{"type": "sources", "sources": [...]}`` — sent first, so the user
          sees which articles were found before the text starts arriving;
        * ``{"type": "token", "text": "..."}`` — one piece of the answer;
        * ``{"type": "done", "answer": ..., "refused": ..., "latency_ms": ...}``.

        A refusal is streamed as a single token event, so clients need only one
        rendering path.
        """
        started = time.perf_counter()
        hits = await self._retrieve(question, top_k, audience, history)
        yield {"type": "sources", "sources": hits, "used_history": bool(history)}

        if not hits:
            yield {"type": "token", "text": REFUSAL_TEXT}
            yield {
                "type": "done",
                "answer": REFUSAL_TEXT,
                "refused": True,
                "latency_ms": int((time.perf_counter() - started) * 1000),
            }
            return

        state = _StreamState()
        prompt = build_user_prompt(question, hits, history)
        async for piece in self._llm.stream(SYSTEM_PROMPT, prompt):
            state.pieces.append(piece)
            yield {"type": "token", "text": piece}

        yield {
            "type": "done",
            "answer": state.text,
            "refused": False,
            "latency_ms": int((time.perf_counter() - started) * 1000),
        }

    async def _retrieve(
        self,
        question: str,
        top_k: int | None,
        audience: str | None,
        history: Sequence[ChatTurn],
    ) -> list[SearchHit]:
        query = build_retrieval_query(question, history)
        return await self._retriever.retrieve(
            query,
            top_k or self._default_top_k,
            audience=audience,
        )

