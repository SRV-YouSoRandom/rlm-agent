from fastapi import APIRouter, HTTPException
from pipelines.rlm import run_rlm
from services.embedder import embed_query
from services.vector_store import search
from services.memory import get_or_create_session, add_to_memory
from schemas.chat import ChatRequest, ChatResponse
from core.config import get_settings
import logging

logger = logging.getLogger("rlm_agent")
router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        if request.use_rlm:
            # Full RLM pipeline
            result = run_rlm(
                request.question,
                collection_name=request.collection_name,
                session_id=request.session_id,
                max_depth=request.rlm_max_depth,
            )
        else:
            # Lightweight vector fallback (no full RLM agent loop)
            result = _vector_fallback(request)

    except Exception as e:
        logger.error(f"Chat pipeline failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {str(e)}")

    return ChatResponse(**result)


def _vector_fallback(request: ChatRequest) -> dict:
    """Simple vector search fallback when use_rlm=False."""
    from langchain_openai import ChatOpenAI
    from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_core.output_parsers import StrOutputParser

    settings = get_settings()
    session_id, memory = get_or_create_session(request.session_id)

    q_vec = embed_query(request.question)
    docs = search(q_vec, top_k=settings.top_k, collection_name=request.collection_name)

    if not docs:
        return {
            "answer": "No relevant documents found.",
            "sources": [],
            "collection_name": request.collection_name or settings.qdrant_collection,
            "session_id": session_id,
            "pipeline_used": "vector_fallback",
        }

    context = "\n\n".join([d["text"] for d in docs])
    llm = ChatOpenAI(
        model=settings.llm_model,
        openai_api_key=settings.openrouter_api_key,
        openai_api_base=settings.openrouter_base_url,
        temperature=0.2,
        max_tokens=1024,
    )
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "Context:\n{context}\n\nQuestion: {question}"),
    ])
    chain = prompt | llm | StrOutputParser()
    chat_history = memory.load_memory_variables({})["chat_history"]
    answer = chain.invoke({"context": context, "question": request.question, "chat_history": chat_history})
    add_to_memory(session_id, request.question, answer)

    return {
        "answer": answer,
        "sources": [{"filename": d["filename"], "score": d["score"]} for d in docs],
        "collection_name": request.collection_name or settings.qdrant_collection,
        "session_id": session_id,
        "pipeline_used": "vector_fallback",
    }