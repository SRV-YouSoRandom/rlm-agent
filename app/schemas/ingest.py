from pydantic import BaseModel


class IngestResponse(BaseModel):
    filename: str
    doc_hash: str
    total_chunks: int
    new_chunks_indexed: int
    message: str
    collection_name: str
    raw_text_stored: bool = False       # Confirms raw text was persisted for RLM
    raw_text_length: int = 0            # Character count of stored raw text