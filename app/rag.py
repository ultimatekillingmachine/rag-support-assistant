"""The RAG pipeline: question -> retrieval -> grounded prompt -> answer + citations."""

from __future__ import annotations

import time
from dataclasses import dataclass

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


@dataclass(frozen=True)
class AnswerResult:
    answer: str
    hits: list[SearchHit]
    latency_ms: int
    refused: bool = False


def build_user_prompt(question: str, hits: list[SearchHit]) -> str:
    """Compose the user prompt: question + numbered context fragments."""
    blocks: list[str] = []
    for ref, hit in enumerate(hits, start=1):
        chunk = hit.chunk
        blocks.append(
            f"[{ref}] {chunk.title} (раздел: {chunk.category}, обновлено: {chunk.updated_at})\n"
            f"{chunk.text}"
        )
    context = "\n\n".join(blocks) if blocks else "(пусто)"
    return (
        f"ВОПРОС ПОКУПАТЕЛЯ: {question}\n\n"
        f"ФРАГМЕНТЫ БАЗЫ ЗНАНИЙ:\n{context}\n\n"
        "Сформулируй ответ покупателю."
    )


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
        audience: str = "customer",
    ) -> AnswerResult:
        started = time.perf_counter()
        hits = await self._retriever.retrieve(
            question,
            top_k or self._default_top_k,
            audience=audience,
        )
        if not hits:
            # Relevance gate (or an empty index): refuse without paying for an LLM call.
            latency_ms = int((time.perf_counter() - started) * 1000)
            return AnswerResult(answer=REFUSAL_TEXT, hits=[], latency_ms=latency_ms, refused=True)
        answer = await self._llm.complete(SYSTEM_PROMPT, build_user_prompt(question, hits))
        latency_ms = int((time.perf_counter() - started) * 1000)
        return AnswerResult(answer=answer, hits=hits, latency_ms=latency_ms)

