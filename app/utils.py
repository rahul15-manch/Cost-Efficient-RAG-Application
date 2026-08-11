import hashlib
import os
import re
import json

def sha256_file(path: str) -> str:
    """Calculate the SHA256 hash of a file."""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()

def clean_text(text: str) -> str:
    """Normalize whitespace and strip repeated blank lines."""
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def detect_file_type(path: str) -> str:
    """Detect file type based on extension."""
    _, ext = os.path.splitext(path)
    return ext.lower().lstrip('.')

def make_chunk_id(source: str, page: int | None, section: str | None, text: str, chunk_index: int | None = None) -> str:
    """Generate a deterministic SHA256 chunk ID."""
    hasher = hashlib.sha256()
    hasher.update(source.encode('utf-8'))
    if page is not None:
        hasher.update(str(page).encode('utf-8'))
    if section is not None:
        hasher.update(section.encode('utf-8'))
    if chunk_index is not None:
        hasher.update(str(chunk_index).encode('utf-8'))
    hasher.update(text.encode('utf-8'))
    return hasher.hexdigest()

def load_seen_chunk_ids(path: str) -> set[str]:
    """Load seen chunk IDs from a JSON file."""
    if not os.path.exists(path):
        return set()
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return set(data)
    except (json.JSONDecodeError, ValueError):
        return set()

def save_seen_chunk_ids(path: str, ids: set[str]) -> None:
    """Save seen chunk IDs to a JSON file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temp_path = path + ".tmp"
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(list(ids), f)
    os.replace(temp_path, path)
