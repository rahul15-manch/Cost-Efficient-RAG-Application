import os
from datetime import datetime
import fitz
import markdown
from bs4 import BeautifulSoup, NavigableString
from app.models import DocumentChunk
from app.utils import sha256_file, clean_text, detect_file_type, make_chunk_id

def extract_chunks_from_soup(soup: BeautifulSoup, file_hash: str, source: str) -> list[DocumentChunk]:
    for tag in soup(["script", "style"]):
        tag.decompose()

    chunks = []
    current_section = None
    current_text = []

    for descendant in soup.descendants:
        if getattr(descendant, "name", None) in ['h1', 'h2', 'h3']:
            # Flush existing text
            text = clean_text('\n'.join(current_text))
            if text:
                chunk_id = make_chunk_id(source, None, current_section, text)
                chunks.append(DocumentChunk(
                    chunk_id=chunk_id,
                    source=source,
                    page=None,
                    section=current_section,
                    text=text,
                    file_hash=file_hash,
                    created_at=datetime.utcnow()
                ))
            current_text = []
            current_section = clean_text(descendant.get_text(separator=' ', strip=True))
        elif isinstance(descendant, NavigableString):
            if getattr(descendant.parent, "name", None) in ['h1', 'h2', 'h3']:
                continue
            clean_str = descendant.strip()
            if clean_str:
                current_text.append(clean_str)

    text = clean_text('\n'.join(current_text))
    if text:
        chunk_id = make_chunk_id(source, None, current_section, text)
        chunks.append(DocumentChunk(
            chunk_id=chunk_id,
            source=source,
            page=None,
            section=current_section,
            text=text,
            file_hash=file_hash,
            created_at=datetime.utcnow()
        ))

    return chunks

def read_pdf(path: str, file_hash: str, source: str) -> list[DocumentChunk]:
    chunks = []
    doc = fitz.open(path)
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = clean_text(page.get_text())
        if not text:
            continue
            
        chunk_id = make_chunk_id(source, page_num + 1, None, text)
        chunks.append(DocumentChunk(
            chunk_id=chunk_id,
            source=source,
            page=page_num + 1,
            section=None,
            text=text,
            file_hash=file_hash,
            created_at=datetime.utcnow()
        ))
    doc.close()
    return chunks

def read_html(path: str, file_hash: str, source: str) -> list[DocumentChunk]:
    with open(path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f.read(), "html.parser")
    return extract_chunks_from_soup(soup, file_hash, source)

def read_markdown(path: str, file_hash: str, source: str) -> list[DocumentChunk]:
    with open(path, "r", encoding="utf-8") as f:
        html = markdown.markdown(f.read())
    soup = BeautifulSoup(html, "html.parser")
    return extract_chunks_from_soup(soup, file_hash, source)

def load_document(path: str) -> tuple[str, list[DocumentChunk]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
        
    if os.path.getsize(path) == 0:
        raise ValueError(f"File is empty: {path}")
        
    file_type = detect_file_type(path)
    file_hash = sha256_file(path)
    source = os.path.basename(path)
    
    if file_type == 'pdf':
        chunks = read_pdf(path, file_hash, source)
    elif file_type in ['html', 'htm']:
        chunks = read_html(path, file_hash, source)
        file_type = 'html'
    elif file_type in ['md', 'markdown']:
        chunks = read_markdown(path, file_hash, source)
        file_type = 'markdown'
    else:
        raise ValueError(f"Unsupported file type: {file_type}")
        
    if not chunks:
        raise ValueError(f"Document contains no text: {path}")
        
    return file_type, chunks
