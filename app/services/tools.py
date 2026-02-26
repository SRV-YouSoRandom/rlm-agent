"""
LangChain tools available to the RLM root agent.
The agent picks the right tool for each step in its reasoning loop.
"""
import logging
from langchain_core.tools import tool
from services.embedder import embed_query
from services.vector_store import search
from services.repl import REPLEnvironment
from services.sub_llm import call_sub_llm, split_and_call
from core.config import get_settings

logger = logging.getLogger("rlm_agent")

# Module-level REPL instance (reset per request in pipeline)
_repl: REPLEnvironment | None = None


def set_repl(repl: REPLEnvironment):
    global _repl
    _repl = repl


def get_repl() -> REPLEnvironment:
    global _repl
    if _repl is None:
        _repl = REPLEnvironment()
    return _repl


# --- Tool definitions ---

@tool
def vector_search(query: str) -> str:
    """
    Search the vector database for semantically similar document chunks.
    Use this for quick targeted retrieval when you know what you're looking for.
    Input: a search query string.
    """
    try:
        settings = get_settings()
        query_vector = embed_query(query)
        docs = search(query_vector, top_k=settings.top_k)
        if not docs:
            return "No results found in vector database."
        return "\n\n---\n\n".join(
            [f"[Score: {d['score']:.3f} | File: {d['filename']}]\n{d['text']}" for d in docs]
        )
    except Exception as e:
        return f"Vector search error: {e}"


@tool
def repl_execute(code: str) -> str:
    """
    Execute Python code in the sandboxed REPL environment.
    The variable 'context' contains all ingested document text.
    Use Python string operations (split, find, regex) to explore it.
    Print results — they will be captured and returned.
    Example: print(context[:500])
    Example: print([line for line in context.split('\\n') if 'revenue' in line.lower()])
    """
    repl = get_repl()
    result = repl.execute(code)
    if result["success"]:
        return result["output"] or "[Code executed, no output printed]"
    return f"[REPL ERROR]: {result['error']}"


@tool
def sub_llm_analyze(instruction_and_snippet: str) -> str:
    """
    Call a focused sub-LLM on a specific text snippet.
    Input format: 'INSTRUCTION|||SNIPPET'
    Use '|||' as the separator between the instruction and the text snippet.
    Example: 'What is the total revenue?|||Q3 revenue was $4.2M, Q4 was $5.1M...'
    """
    try:
        if "|||" not in instruction_and_snippet:
            return "[ERROR]: Input must be formatted as 'INSTRUCTION|||SNIPPET'"
        instruction, snippet = instruction_and_snippet.split("|||", 1)
        return call_sub_llm(instruction.strip(), snippet.strip(), current_depth=1)
    except Exception as e:
        return f"[Sub-LLM error]: {e}"


@tool
def divide_and_analyze(instruction_and_text: str) -> str:
    """
    Divide a large text into segments and run a sub-LLM on each segment in parallel.
    Use this for broad questions over large documents where you don't know where the answer is.
    Input format: 'INSTRUCTION|||TEXT'
    Use '|||' as the separator.
    Returns aggregated findings from all segments.
    """
    try:
        if "|||" not in instruction_and_text:
            return "[ERROR]: Input must be formatted as 'INSTRUCTION|||TEXT'"
        instruction, text = instruction_and_text.split("|||", 1)
        results = split_and_call(instruction.strip(), text.strip(), n_splits=4, current_depth=1)
        hits = [r for r in results if "NOT FOUND IN SNIPPET" not in r]
        if not hits:
            return "No relevant information found across all segments."
        return "\n\n---SEGMENT---\n\n".join(hits)
    except Exception as e:
        return f"[Divide-and-analyze error]: {e}"


@tool
def grep_context(pattern: str) -> str:
    """
    Search for lines in the document context containing a keyword or phrase.
    Faster than sub-LLM for exact keyword lookups.
    Input: a keyword or short phrase to search for.
    Returns matching lines with surrounding context.
    """
    repl = get_repl()
    context = repl.get_variable("context") or ""
    if not context:
        return "No document context loaded."

    lines = context.split("\n")
    matches = []
    pattern_lower = pattern.lower()
    for i, line in enumerate(lines):
        if pattern_lower in line.lower():
            start = max(0, i - 1)
            end = min(len(lines), i + 2)
            matches.append("\n".join(lines[start:end]))

    if not matches:
        return f"No lines found containing '{pattern}'."
    return f"Found {len(matches)} match(es):\n\n" + "\n\n---\n\n".join(matches[:10])


def get_all_tools():
    return [vector_search, repl_execute, sub_llm_analyze, divide_and_analyze, grep_context]