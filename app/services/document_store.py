"""
Raw document text storage for RLM.
Stores full document text so the REPL environment can access it as a variable.
File-backed with JSON index. Upgradeable to Redis.
"""
import json
import os
import logging
from pathlib import Path
from core.config import get_settings

logger = logging.getLogger("rlm_agent")


def _store_path() -> Path:
    settings = get_settings()
    p = Path(settings.document_storage_path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _index_path() -> Path:
    return _store_path() / "index.json"


def _load_index() -> dict:
    idx = _index_path()
    if not idx.exists():
        return {}
    with open(idx, "r") as f:
        return json.load(f)


def _save_index(index: dict):
    with open(_index_path(), "w") as f:
        json.dump(index, f, indent=2)


def store_document_text(doc_hash: str, filename: str, text: str) -> bool:
    """Persist raw document text keyed by doc_hash."""
    try:
        store = _store_path()
        text_file = store / f"{doc_hash}.txt"
        with open(text_file, "w", encoding="utf-8") as f:
            f.write(text)

        index = _load_index()
        index[doc_hash] = {
            "filename": filename,
            "length": len(text),
            "path": str(text_file),
        }
        _save_index(index)
        logger.info(f"Stored raw text for {filename} ({len(text)} chars)")
        return True
    except Exception as e:
        logger.error(f"Failed to store document text: {e}")
        return False


def get_document_text(doc_hash: str) -> str | None:
    """Retrieve raw text by doc_hash."""
    index = _load_index()
    if doc_hash not in index:
        return None
    try:
        with open(index[doc_hash]["path"], "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        logger.error(f"Failed to read document text: {e}")
        return None


def get_all_documents_text(collection_name: str = None) -> str:
    """
    Concatenate all stored document texts.
    In production you'd filter by collection_name.
    Returns combined text for REPL injection.
    """
    index = _load_index()
    if not index:
        return ""
    parts = []
    for doc_hash, meta in index.items():
        text = get_document_text(doc_hash)
        if text:
            parts.append(f"=== Document: {meta['filename']} ===\n{text}")
    return "\n\n".join(parts)


def list_stored_documents() -> list[dict]:
    """List all stored documents with metadata."""
    index = _load_index()
    return [
        {"doc_hash": k, "filename": v["filename"], "length": v["length"]}
        for k, v in index.items()
    ]


def delete_document_text(doc_hash: str) -> bool:
    """Delete stored raw text."""
    index = _load_index()
    if doc_hash not in index:
        return False
    try:
        os.remove(index[doc_hash]["path"])
        del index[doc_hash]
        _save_index(index)
        return True
    except Exception as e:
        logger.error(f"Failed to delete document text: {e}")
        return False