from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pipelines.rlm import run_rlm_streaming
from schemas.chat import ChatRequest

router = APIRouter()


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """Streaming RLM chat — streams tool calls and final answer."""
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        return StreamingResponse(
            run_rlm_streaming(
                request.question,
                collection_name=request.collection_name,
                session_id=request.session_id,
            ),
            media_type="text/event-stream",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Streaming failed: {str(e)}")