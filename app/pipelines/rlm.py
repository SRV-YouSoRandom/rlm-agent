"""
Core RLM pipeline.
Root LLM agent loop with access to REPL, sub-LLM calls, vector search, and grep tools.
"""
import logging
import time
import uuid
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage
from services.repl import REPLEnvironment
from services.document_store import get_all_documents_text
from services.tools import get_all_tools, set_repl
from services.memory import get_or_create_session, add_to_memory
from core.config import get_settings
from core.metrics import get_metrics_collector, QueryMetrics

logger = logging.getLogger("rlm_agent")

ROOT_SYSTEM_PROMPT = """You are an intelligent document analysis assistant with access to powerful tools.

You have the following tools available:
- **vector_search**: Fast semantic search over document chunks
- **repl_execute**: Execute Python code to explore documents (variable 'context' has all document text)
- **sub_llm_analyze**: Call a focused AI on a specific text snippet
- **divide_and_analyze**: Split a large text and analyze each segment
- **grep_context**: Keyword search across all document text

Strategy:
1. For simple questions: use vector_search or grep_context first (fast)
2. For complex questions requiring deep understanding: use repl_execute to explore, then sub_llm_analyze
3. For broad questions over long documents: use divide_and_analyze
4. Combine tools — grep to find a section, then sub_llm_analyze to understand it
5. Always cite which document/section your answer comes from

If documents have not been ingested, say so clearly."""


def build_rlm_agent(settings):
    llm = ChatOpenAI(
        model=settings.llm_model,
        openai_api_key=settings.openrouter_api_key,
        openai_api_base=settings.openrouter_base_url,
        temperature=0.2,
        max_tokens=settings.rlm_max_tokens_per_call,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", ROOT_SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    tools = get_all_tools()
    agent = create_openai_tools_agent(llm, tools, prompt)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        max_iterations=settings.rlm_agent_max_iterations,
        verbose=True,
        return_intermediate_steps=True,
        handle_parsing_errors=True,
    )


def run_rlm(
    question: str,
    collection_name: str = None,
    session_id: str = None,
    max_depth: int = None,
) -> dict:
    settings = get_settings()
    metrics_collector = get_metrics_collector()

    metrics = QueryMetrics(
        session_id=session_id or "none",
        question=question,
        collection_name=collection_name or settings.qdrant_collection,
    )
    start_time = time.time()

    try:
        # Setup session
        session_id, memory = get_or_create_session(session_id)

        # Load document context into REPL
        doc_text = get_all_documents_text(collection_name)
        repl = REPLEnvironment(document_text=doc_text)
        set_repl(repl)  # Make REPL available to tools

        # Build and run agent
        agent_executor = build_rlm_agent(settings)
        chat_history = memory.load_memory_variables({})["chat_history"]

        result = agent_executor.invoke({
            "input": question,
            "chat_history": chat_history,
        })

        answer = result.get("output", "No answer generated.")
        intermediate_steps = result.get("intermediate_steps", [])

        # Parse agent steps for response
        agent_steps = []
        for i, (action, observation) in enumerate(intermediate_steps):
            agent_steps.append({
                "step_number": i + 1,
                "tool_used": action.tool,
                "input_summary": str(action.tool_input)[:150],
                "output_summary": str(observation)[:150],
                "recursion_depth": 0,
            })

        # Save to memory
        add_to_memory(session_id, question, answer)

        metrics.total_latency_ms = (time.time() - start_time) * 1000
        metrics.num_docs_retrieved = len(intermediate_steps)
        metrics.success = True
        metrics_collector.record_query(metrics)

        return {
            "answer": answer,
            "sources": [],  # Extracted from agent steps in production
            "collection_name": collection_name or settings.qdrant_collection,
            "session_id": session_id,
            "agent_steps": agent_steps,
            "recursion_depth_reached": len(intermediate_steps),
            "pipeline_used": "rlm",
        }

    except Exception as e:
        metrics.total_latency_ms = (time.time() - start_time) * 1000
        metrics.success = False
        metrics.error = str(e)
        metrics_collector.record_query(metrics)
        logger.error(f"RLM pipeline failed: {e}", exc_info=True)
        raise


def run_rlm_streaming(
    question: str,
    collection_name: str = None,
    session_id: str = None,
):
    """Streaming RLM — yields agent step events then final answer."""
    settings = get_settings()

    session_id, memory = get_or_create_session(session_id)
    doc_text = get_all_documents_text(collection_name)
    repl = REPLEnvironment(document_text=doc_text)
    set_repl(repl)

    llm = ChatOpenAI(
        model=settings.llm_model,
        openai_api_key=settings.openrouter_api_key,
        openai_api_base=settings.openrouter_base_url,
        temperature=0.2,
        max_tokens=settings.rlm_max_tokens_per_call,
        streaming=True,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", ROOT_SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    tools = get_all_tools()
    agent = create_openai_tools_agent(llm, tools, prompt)
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        max_iterations=settings.rlm_agent_max_iterations,
        verbose=False,
        return_intermediate_steps=True,
        handle_parsing_errors=True,
    )

    chat_history = memory.load_memory_variables({})["chat_history"]
    full_answer = ""

    for event in executor.stream({"input": question, "chat_history": chat_history}):
        if "actions" in event:
            for action in event["actions"]:
                yield f"[TOOL: {action.tool}] {str(action.tool_input)[:100]}\n"
        elif "steps" in event:
            for step in event["steps"]:
                obs = str(step.observation)[:200]
                yield f"[RESULT]: {obs}\n"
        elif "output" in event:
            full_answer = event["output"]
            yield f"\n[ANSWER]: {full_answer}"

    if full_answer:
        add_to_memory(session_id, question, full_answer)