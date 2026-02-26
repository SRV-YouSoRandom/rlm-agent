from services.parser import parse_pdf
from services.chunker import chunk_text
from services.embedder import embed_texts
from services.hasher import hash_chunk, hash_document
from services.vector_store import ensure_collection, upsert_chunks
from services.document_store import store_document_text
import logging

logger = logging.getLogger("rlm_agent")


def run_ingest(file_bytes: bytes, filename: str, collection_name: str = None) -> dict:
    logger.info(f"Starting ingest for: {filename}")

    doc_hash = hash_document(file_bytes)
    raw_text = parse_pdf(file_bytes)
    if not raw_text.strip():
        raise ValueError("PDF appears to be empty or unreadable.")

    chunks = chunk_text(raw_text)
    logger.info(f"Produced {len(chunks)} chunks")

    chunk_hashes = [hash_chunk(c) for c in chunks]
    embeddings = embed_texts(chunks)

    ensure_collection(collection_name)
    upserted = upsert_chunks(chunks, embeddings, chunk_hashes, doc_hash, filename, collection_name)

    # Store raw text for RLM REPL access
    raw_stored = store_document_text(doc_hash, filename, raw_text)

    return {
        "filename": filename,
        "doc_hash": doc_hash,
        "total_chunks": len(chunks),
        "new_chunks_indexed": upserted,
        "collection_name": collection_name or "default",
        "raw_text_stored": raw_stored,          
        "raw_text_length": len(raw_text),        
    }