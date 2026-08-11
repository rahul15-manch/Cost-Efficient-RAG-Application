import time
import groq
from groq import Groq
from app.config import settings
from app.models import RetrievedChunk, RetrievalResponse
from app.telemetry import logger

_client = None

def get_groq_client() -> Groq:
    global _client
    if _client is None:
        from app.config import is_groq_configured
        if not is_groq_configured():
            raise ValueError("Groq API key not configured")
        _client = Groq(api_key=settings.GROQ_API_KEY)
    return _client

def build_context(chunks: list[RetrievedChunk]) -> str:
    context_parts = []
    for chunk in chunks:
        part = f"--- Chunk ID: {chunk.chunk_id} ---\n"
        part += f"Source: {chunk.source}\n"
        if chunk.page is not None:
            part += f"Page: {chunk.page}\n"
        if chunk.section is not None:
            part += f"Section: {chunk.section}\n"
        part += f"Text: {chunk.text}\n"
        context_parts.append(part)
    return "\n".join(context_parts)

def build_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    context = build_context(chunks)
    prompt = f"""You are a grounded QA assistant. Answer ONLY from the provided context. Do not use outside knowledge. Cite supporting chunk ids inline like [chunk:abc123]. If the context is insufficient, reply exactly: "No relevant context found." Do not invent facts. Do not paraphrase unsupported claims.

Context:
{context}

Question: {question}
Answer:"""
    return prompt

def generate_answer(question: str, retrieval_response: RetrievalResponse) -> tuple[str, dict, float]:
    start_time = time.perf_counter()
    
    prompt = build_prompt(question, retrieval_response.chunks)
    
    try:
        client = get_groq_client()
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model=settings.LLM_MODEL,
            temperature=0,
            max_tokens=512,
        )
        answer = response.choices[0].message.content or ""
        token_usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens
        }
    except Exception as e:
        logger.error(f"Generation failed: {str(e)}")
        answer = f"Error during generation: {str(e)}"
        token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    
    end_time = time.perf_counter()
    generation_time_ms = round((end_time - start_time) * 1000, 2)
    
    return answer, token_usage, generation_time_ms
