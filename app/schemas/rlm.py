from pydantic import BaseModel
from typing import Optional, List, Any
from datetime import datetime


class REPLExecution(BaseModel):
    """Tracks a single REPL code execution."""
    code: str
    output: str
    success: bool
    execution_time_ms: float
    error: Optional[str] = None


class SubLLMCall(BaseModel):
    """Tracks a single recursive sub-LLM invocation."""
    depth: int
    instruction: str
    context_length: int
    response_summary: str
    latency_ms: float


class RLMTrace(BaseModel):
    """Full execution trace of an RLM pipeline run."""
    session_id: str
    question: str
    collection_name: str
    total_latency_ms: float
    recursion_depth_reached: int
    repl_executions: List[REPLExecution]
    sub_llm_calls: List[SubLLMCall]
    pipeline_used: str
    success: bool
    timestamp: datetime