"""
Recursive sub-LLM service.
The root LLM can call sub-LLMs on context snippets to drill down recursively.
Each call is tracked for depth limiting and observability.
"""
import logging
import time
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from core.config import get_settings

logger = logging.getLogger("rlm_agent")


SUB_LLM_PROMPT = """You are a focused analytical assistant. 
Your job is to answer a specific instruction based ONLY on the provided context snippet.
Be concise and precise. If the answer is not in the snippet, say "NOT FOUND IN SNIPPET".

Instruction: {instruction}

Context Snippet:
{context_snippet}

Answer:"""


def call_sub_llm(
    instruction: str,
    context_snippet: str,
    current_depth: int = 0,
    call_log: list = None,
) -> str:
    """
    Call a sub-LLM on a context snippet.
    Tracks recursion depth and logs call metadata.
    
    Args:
        instruction: What to find/analyze in the snippet
        context_snippet: The text to analyze
        current_depth: Current recursion depth (for limit enforcement)
        call_log: Mutable list to append call metadata to
    
    Returns:
        LLM response string
    """
    settings = get_settings()

    if current_depth >= settings.rlm_max_recursion_depth:
        logger.warning(f"Max recursion depth {settings.rlm_max_recursion_depth} reached")
        return f"[MAX DEPTH REACHED at depth {current_depth}]"

    # Truncate snippet if needed
    snippet = context_snippet[:settings.rlm_snippet_size]

    llm = ChatOpenAI(
        model=settings.llm_model,
        openai_api_key=settings.openrouter_api_key,
        openai_api_base=settings.openrouter_base_url,
        temperature=0.1,
        max_tokens=settings.rlm_max_tokens_per_call,
    )

    prompt = ChatPromptTemplate.from_template(SUB_LLM_PROMPT)
    chain = prompt | llm | StrOutputParser()

    start = time.time()
    try:
        response = chain.invoke({
            "instruction": instruction,
            "context_snippet": snippet,
        })
        latency_ms = (time.time() - start) * 1000

        if call_log is not None:
            call_log.append({
                "depth": current_depth,
                "instruction": instruction[:100],
                "context_length": len(snippet),
                "response_summary": response[:200],
                "latency_ms": latency_ms,
            })

        logger.info(f"Sub-LLM call at depth {current_depth} | {latency_ms:.0f}ms | {len(snippet)} chars")
        return response

    except Exception as e:
        logger.error(f"Sub-LLM call failed at depth {current_depth}: {e}")
        return f"[SUB-LLM ERROR: {str(e)}]"


def split_and_call(
    instruction: str,
    full_text: str,
    n_splits: int = 4,
    current_depth: int = 0,
    call_log: list = None,
) -> list[str]:
    """
    Divide full_text into n_splits segments and call sub-LLM on each.
    This is the key RLM divide-and-conquer primitive.
    """
    if not full_text.strip():
        return []

    chunk_size = len(full_text) // n_splits
    results = []

    for i in range(n_splits):
        start_idx = i * chunk_size
        end_idx = start_idx + chunk_size if i < n_splits - 1 else len(full_text)
        snippet = full_text[start_idx:end_idx]

        if snippet.strip():
            result = call_sub_llm(
                instruction=instruction,
                context_snippet=snippet,
                current_depth=current_depth,
                call_log=call_log,
            )
            results.append(result)

    return results