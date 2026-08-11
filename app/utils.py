import hashlib
import os
import re

def sha256_file(path: str) -> str:
    """Calculate the SHA256 hash of a file."""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()

def clean_text(text: str) -> str:
    """Normalize whitespace and strip repeated blank lines."""
    # Replace multiple spaces/tabs with a single space
    text = re.sub(r'[ \t]+', ' ', text)
    # Replace 3 or more newlines with 2 newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def detect_file_type(path: str) -> str:
    """Detect file type based on extension."""
    _, ext = os.path.splitext(path)
    return ext.lower().lstrip('.')

def make_chunk_id(source: str, page: int | None, section: str | None, text: str) -> str:
    """Generate a deterministic SHA256 chunk ID."""
    hasher = hashlib.sha256()
    hasher.update(source.encode('utf-8'))
    if page is not None:
        hasher.update(str(page).encode('utf-8'))
    if section is not None:
        hasher.update(section.encode('utf-8'))
    hasher.update(text.encode('utf-8'))
    return hasher.hexdigest()
