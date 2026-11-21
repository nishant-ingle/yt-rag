"""Multi-mode RAG engine (Lookup, Opinion, Evolution)."""
from __future__ import annotations

import asyncio
import re
from typing import Any, Literal

from openai import AsyncOpenAI

import backend.db as db
from backend.config import settings

QueryMode = Literal["lookup", "opinion", "evolution"]


def format_timestamp(seconds: int) -> str:
    """Format total seconds into MM:SS or HH:MM:SS."""
    s = max(0, int(seconds))
    hours = s // 3600
    minutes = (s % 3600) // 60
    rem_seconds = s % 60
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{rem_seconds:02d}"
    return f"{minutes:02d}:{rem_seconds:02d}"


def format_upload_date(date_str: str) -> str:
    """Format YYYYMMDD string into YYYY-MM-DD if possible."""
    clean = (date_str or "").strip()
    if len(clean) == 8 and clean.isdigit():
        return f"{clean[:4]}-{clean[4:6]}-{clean[6:8]}"
    return clean or "Unknown Date"


def build_source_dict(chunk: dict[str, Any]) -> dict[str, Any]:
    """Convert chunk dictionary to standard structured source item."""
    vid = chunk["video_id"]
    start_sec = int(chunk["start_sec"])
    end_sec = int(chunk["end_sec"])
    return {
        "video_id": vid,
        "title": chunk.get("title", ""),
        "upload_date": chunk.get("upload_date", ""),
        "start_sec": start_sec,
        "end_sec": end_sec,
        "timestamp_str": format_timestamp(start_sec),
        "url": f"https://youtu.be/{vid}?t={start_sec}",
        "snippet": chunk.get("text", ""),
    }


async def classify_query(question: str, provider: str = "openai", chat_key: str = "") -> QueryMode:
    """Classify user question into lookup, opinion, or evolution."""
    q_lower = question.lower()

    # 1. Evolution regex heuristics
    evolution_patterns = [
        r"\b(evolv|evolution)\b",
        r"\bchange[d]?\b",
        r"\bshift(ed|ing)?\b",
        r"\bover time\b",
        r"\bfrom \d{4} to \d{4}\b",
        r"\bbetween \d{4} and \d{4}\b",
        r"\bearlier vs\b",
        r"\btimeline\b",
        r"\bhow has his (view|opinion|stance|position)\b",
    ]
    if any(re.search(pat, q_lower) for pat in evolution_patterns):
        return "evolution"

    # 2. Opinion regex heuristics
    opinion_patterns = [
        r"\boverall opinion\b",
        r"\boverall view\b",
        r"\boverall stance\b",
        r"\bgeneral (opinion|view|stance|perspective)\b",
        r"\bwhat is his (view|opinion|stance|philosophy|perspective) on\b",
        r"\bwhat does he think (about|of)\b",
    ]
    if any(re.search(pat, q_lower) for pat in opinion_patterns):
        return "opinion"

    # 3. LLM classifier fallback
    client, model = get_chat_client_and_model(provider, chat_key)
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a query router for a YouTube video transcript RAG system. "
                        "Classify the question into exactly one category: lookup, opinion, or evolution.\n"
                        "- lookup: Specific facts, exact quotes, definitions, single events, or timestamp lookup.\n"
                        "- opinion: Broad viewpoint, philosophy, thesis, or general stance on a topic.\n"
                        "- evolution: How views, stance, or predictions changed or developed over time or across years.\n"
                        "Reply with ONLY one lowercase word: lookup, opinion, or evolution."
                    ),
                },
                {"role": "user", "content": question},
            ],
            temperature=0.0,
            max_tokens=10,
        )
        content = (response.choices[0].message.content or "").strip().lower()
        if "evolution" in content:
            return "evolution"
        if "opinion" in content:
            return "opinion"
        return "lookup"
    except Exception:
        return "lookup"


def get_chat_client_and_model(provider: str = "openai", chat_key: str = "") -> tuple[AsyncOpenAI, str]:
    """Return AsyncOpenAI client and model for the selected provider."""
    prov_cfg = settings.CHAT_PROVIDERS.get(provider) or settings.CHAT_PROVIDERS["openai"]
    key = chat_key or settings.LLM_API_KEY
    client = AsyncOpenAI(base_url=prov_cfg["base_url"], api_key=key)
    return client, prov_cfg["model"]


def get_embed_client(embed_key: str = "") -> AsyncOpenAI:
    """Return AsyncOpenAI client configured for embedding models."""
    key = embed_key or settings.EMBED_API_KEY or settings.LLM_API_KEY
    return AsyncOpenAI(base_url=settings.EMBED_BASE_URL, api_key=key)


async def get_query_embedding(text: str, embed_key: str = "") -> list[float]:
    """Generate vector embedding for the query."""
    client = get_embed_client(embed_key)
    response = await client.embeddings.create(
        model=settings.EMBED_MODEL,
        input=[text],
    )
    return response.data[0].embedding


async def run_lookup_pipeline(
    question: str,
    playlist_id: str | None = None,
    provider: str = "openai",
    chat_key: str = "",
    embed_key: str = "",
) -> dict[str, Any]:
    """Lookup mode: fact and timestamp retrieval with top-8 chunks."""
    query_emb = await get_query_embedding(question, embed_key=embed_key)
    raw_chunks = await asyncio.to_thread(
        db.hybrid_search,
        query_embedding=query_emb,
        query_text=question,
        playlist_id=playlist_id,
        limit=8,
    )

    if not raw_chunks:
        return {
            "answer": "No relevant transcript chunks found for this query in the indexed videos.",
            "mode": "lookup",
            "sources": [],
        }

    sources = [build_source_dict(c) for c in raw_chunks]

    context_blocks = []
    for s in sources:
        context_blocks.append(
            f"Source: [{s['title']} @ {s['timestamp_str']}]({s['url']})\n"
            f"Transcript: {s['snippet']}"
        )
    context_text = "\n\n---\n\n".join(context_blocks)

    system_prompt = (
        "You are an expert assistant answering questions based strictly on YouTube video transcripts.\n"
        "Instructions:\n"
        "1. Answer concisely, directly addressing the question using only the provided context.\n"
        "2. Cite exact timestamps inline in markdown format: [Video Title @ MM:SS](https://youtu.be/{video_id}?t={start_sec}).\n"
        "3. Only use the provided citations; never fabricate URLs or timestamps.\n"
        "4. If the context does not contain the answer, state that clearly."
    )

    client, model = get_chat_client_and_model(provider, chat_key)
    completion = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n\n{context_text}\n\nQuestion: {question}"},
        ],
        temperature=0.2,
    )
    answer = completion.choices[0].message.content or ""

    return {
        "answer": answer,
        "mode": "lookup",
        "sources": sources,
    }


async def run_opinion_pipeline(
    question: str,
    playlist_id: str | None = None,
    provider: str = "openai",
    chat_key: str = "",
    embed_key: str = "",
) -> dict[str, Any]:
    """Opinion mode: broad synthesis across distinct videos (capped at 2 chunks per video)."""
    query_emb = await get_query_embedding(question, embed_key=embed_key)
    raw_chunks = await asyncio.to_thread(
        db.hybrid_search,
        query_embedding=query_emb,
        query_text=question,
        playlist_id=playlist_id,
        limit=24,
    )

    if not raw_chunks:
        return {
            "answer": "No relevant transcript chunks found for this query in the indexed videos.",
            "mode": "opinion",
            "sources": [],
        }

    # Deduplicate: cap to maximum 2 chunks per video_id, max 12 chunks total
    deduped_chunks: list[dict[str, Any]] = []
    video_chunk_counts: dict[str, int] = {}

    for c in raw_chunks:
        vid = c["video_id"]
        count = video_chunk_counts.get(vid, 0)
        if count < 2:
            video_chunk_counts[vid] = count + 1
            deduped_chunks.append(c)
        if len(deduped_chunks) >= 12:
            break

    sources = [build_source_dict(c) for c in deduped_chunks]

    context_blocks = []
    for s in sources:
        context_blocks.append(
            f"Video: {s['title']} | Date: {format_upload_date(s['upload_date'])}\n"
            f"Timestamp Link: [{s['title']} @ {s['timestamp_str']}]({s['url']})\n"
            f"Content: {s['snippet']}"
        )
    context_text = "\n\n---\n\n".join(context_blocks)

    system_prompt = (
        "You are synthesizing a YouTube creator's overall perspective and stance on a topic across multiple videos.\n"
        "Instructions:\n"
        "1. Synthesize the speaker's core thesis, nuanced positions, and key arguments based on the transcript evidence.\n"
        "2. Contrast or connect points across different videos to present a cohesive overview.\n"
        "3. Cite supporting statements inline with timestamp markdown links: [Video Title @ MM:SS](https://youtu.be/{video_id}?t={start_sec}).\n"
        "4. Remain objective and faithful to what the speaker actually articulated."
    )

    client, model = get_chat_client_and_model(provider, chat_key)
    completion = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Evidence from Transcripts:\n\n{context_text}\n\nQuestion: {question}"},
        ],
        temperature=0.3,
    )
    answer = completion.choices[0].message.content or ""

    return {
        "answer": answer,
        "mode": "opinion",
        "sources": sources,
    }


async def run_evolution_pipeline(
    question: str,
    playlist_id: str | None = None,
    provider: str = "openai",
    chat_key: str = "",
    embed_key: str = "",
) -> dict[str, Any]:
    """Evolution mode: chronological synthesis analyzing how views evolved over time."""
    query_emb = await get_query_embedding(question, embed_key=embed_key)
    raw_chunks = await asyncio.to_thread(
        db.hybrid_search,
        query_embedding=query_emb,
        query_text=question,
        playlist_id=playlist_id,
        limit=30,
    )

    if not raw_chunks:
        return {
            "answer": "No relevant transcript chunks found for this query in the indexed videos.",
            "mode": "evolution",
            "sources": [],
        }

    # Sort retrieved chunks strictly ascending by upload_date, then start_sec
    sorted_chunks = sorted(
        raw_chunks,
        key=lambda c: (c.get("upload_date") or "", int(c.get("start_sec") or 0)),
    )

    sources = [build_source_dict(c) for c in sorted_chunks]

    # Group chunks chronologically by upload year/date
    grouped_context: dict[str, list[dict[str, Any]]] = {}
    for s in sources:
        year_key = (s["upload_date"][:4]) if s["upload_date"] else "Undated"
        grouped_context.setdefault(year_key, []).append(s)

    chronological_blocks = []
    for period, period_sources in grouped_context.items():
        period_items = []
        for s in period_sources:
            date_fmt = format_upload_date(s["upload_date"])
            period_items.append(
                f"- [{s['title']} ({date_fmt}) @ {s['timestamp_str']}]({s['url']}):\n  \"{s['snippet']}\""
            )
        chronological_blocks.append(f"### Period: {period}\n" + "\n".join(period_items))

    evidence_text = "\n\n".join(chronological_blocks)

    system_prompt = (
        "You are an expert analyst studying the chronological evolution of a speaker's ideas and opinions over time.\n"
        "Instructions:\n"
        "1. Compare earlier statements versus later statements chronologically based on the provided evidence.\n"
        "2. Explicitly highlight: early positions, later/recent positions, continuities, and any shifts or refinements in opinion.\n"
        "3. Cite dates and timestamps inline as: [Video Title (YYYY-MM-DD) @ MM:SS](https://youtu.be/{video_id}?t={start_sec}).\n"
        "4. Structure your response clearly with chronological progression."
    )

    client, model = get_chat_client_and_model(provider, chat_key)
    completion = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Chronological Evidence:\n\n{evidence_text}\n\nQuestion: {question}"},
        ],
        temperature=0.3,
    )
    answer = completion.choices[0].message.content or ""

    return {
        "answer": answer,
        "mode": "evolution",
        "sources": sources,
    }


async def query_rag(
    question: str,
    playlist_id: str | None = None,
    mode: Literal["auto", "lookup", "opinion", "evolution"] = "auto",
    provider: str = "openai",
    chat_key: str = "",
    embed_key: str = "",
) -> dict[str, Any]:
    """Unified entry point for multi-mode RAG queries."""
    chosen_mode: QueryMode
    if mode == "auto":
        chosen_mode = await classify_query(question, provider=provider, chat_key=chat_key)
    else:
        chosen_mode = mode

    if chosen_mode == "lookup":
        return await run_lookup_pipeline(
            question, playlist_id=playlist_id, provider=provider, chat_key=chat_key, embed_key=embed_key
        )
    elif chosen_mode == "opinion":
        return await run_opinion_pipeline(
            question, playlist_id=playlist_id, provider=provider, chat_key=chat_key, embed_key=embed_key
        )
    elif chosen_mode == "evolution":
        return await run_evolution_pipeline(
            question, playlist_id=playlist_id, provider=provider, chat_key=chat_key, embed_key=embed_key
        )
    else:
        return await run_lookup_pipeline(
            question, playlist_id=playlist_id, provider=provider, chat_key=chat_key, embed_key=embed_key
        )
