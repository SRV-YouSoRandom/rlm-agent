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
    ALWAYS try this tool first for any question — it is fast and reliable.
    Input: a search query string describing what you want to find.
    Example: vector_search("unreliable networks distributed systems")
    """
    try:
        settings = get_settings()
        query_vector = embed_query(query)
        # Use the collection_name stored in REPL context if available
        collection = _repl.get_variable("collection_name") if _repl else None
        docs = search(query_vector, top_k=settings.top_k, collection_name=collection)
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
    Execute Python code in the REPL environment.
    The variable 'context' contains ALL ingested document text as a string.
    Always print your results — output is captured and returned.

    Good examples:
      print(context[:2000])
      lines = [l for l in context.split('\\n') if 'replication' in l.lower()]
      print('\\n'.join(lines[:20]))

    DO NOT just print(len(context)) — always extract and print actual content.
    """
    repl = get_repl()
    result = repl.execute(code)
    if result["success"]:
        output = result["output"]
        if not output or not output.strip():
            return "[Code executed but no output was printed. Make sure to use print() to show results.]"
        return output
    return f"[REPL ERROR]: {result['error']}"


@tool
def sub_llm_analyze(instruction_and_snippet: str) -> str:
    """
    Call a focused sub-LLM to analyze a specific piece of text.
    Input format: 'YOUR QUESTION|||THE TEXT TO ANALYZE'
    Use '|||' as the separator. Provide real text after |||, not a variable name.

    Example:
      sub_llm_analyze("What strategies are mentioned for handling network failures?|||[paste actual text here]")
    """
    try:
        if "|||" not in instruction_and_snippet:
            return "[ERROR]: Input must be formatted as 'INSTRUCTION|||TEXT'. Make sure to include actual text after |||, not a variable name."
        instruction, snippet = instruction_and_snippet.split("|||", 1)
        snippet = snippet.strip()
        if not snippet or snippet == "context":
            repl = get_repl()
            snippet = repl.get_variable("context") or ""
            if not snippet:
                return "[ERROR]: No document text available. Please ingest documents first."
        return call_sub_llm(instruction.strip(), snippet, current_depth=1)
    except Exception as e:
        return f"[Sub-LLM error]: {e}"


@tool
def divide_and_analyze(instruction_and_text: str) -> str:
    """
    Split document text into segments and analyze each one with a sub-LLM.
    Use this for broad questions that require scanning the whole document.
    Input format: 'YOUR QUESTION|||THE TEXT TO ANALYZE'
    Use '|||' as the separator. Provide real text after |||, not a variable name.

    To analyze the full document, use:
      divide_and_analyze("your question|||FULL_CONTEXT")
    The keyword FULL_CONTEXT will be automatically replaced with the document text.
    """
    try:
        if "|||" not in instruction_and_text:
            return "[ERROR]: Input must be formatted as 'INSTRUCTION|||TEXT'."
        instruction, text = instruction_and_text.split("|||", 1)
        text = text.strip()

        # Handle cases where agent passes variable name or keyword instead of actual text
        if not text or text in ("context", "FULL_CONTEXT", "full_context"):
            repl = get_repl()
            text = repl.get_variable("context") or ""
            if not text:
                return "[ERROR]: No document text available. Please ingest documents first."

        results = split_and_call(instruction.strip(), text, n_splits=4, current_depth=1)
        hits = [r for r in results if "NOT FOUND IN SNIPPET" not in r]
        if not hits:
            return "No relevant information found across all segments."
        return "\n\n---SEGMENT---\n\n".join(hits)
    except Exception as e:
        return f"[Divide-and-analyze error]: {e}"


@tool
def grep_context(pattern: str) -> str:
    """
    Search for lines in the document containing a keyword or phrase.
    Fast exact keyword search. Use for finding specific terms or topics.
    Input: a keyword or short phrase to search for.
    Returns matching lines with surrounding context (up to 10 matches).
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