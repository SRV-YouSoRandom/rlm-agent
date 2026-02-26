from pydantic import BaseModel
from typing import Optional, List, Any


class ChatRequest(BaseModel):
    question: str
    collection_name: Optional[str] = None
    session_id: Optional[str] = None
    use_rlm: bool = True
    rlm_max_depth: Optional[int] = None
    stream_agent_steps: bool = False


class SourceDoc(BaseModel):
    filename: str
    score: float
    rerank_score: Optional[float] = None


class AgentStep(BaseModel):
    step_number: int
    tool_used: str
    input_summary: str
    output_summary: str
    recursion_depth: int


class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceDoc]
    collection_name: str
    session_id: str
    agent_steps: Optional[List[AgentStep]] = None
    recursion_depth_reached: Optional[int] = None
    pipeline_used: str = "rlm"


class ChatHistoryItem(BaseModel):
    role: str
    content: str


class SessionHistoryResponse(BaseModel):
    session_id: str
    history: List[ChatHistoryItem]