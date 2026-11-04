#Enterprise RAG - Document ingestion (incremental indexing)
from config import DOCS_DIR, CHUNK_SIZE, CHUNK_OVERLAP, ALLOWED_LEVELS
import os
import hashlib
from rich import print
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from store import vectorstore
from database import SessionLocal, DocumentRecord

splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

# 🧰Helpers
def get_file_hash(path : str) -> str:
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8192), b""):
            sha.update(block)
    return sha.hexdigest()

def load_file(path : str) -> list:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return PyPDFLoader(path).load()
    if ext == ".docx":
        return Docx2txtLoader(path).load()
    if ext in [".txt", ".md"]:
        return TextLoader(path, encoding="utf-8").load()
    raise ValueError(f"Unsupported file type : {ext}")

# 📥Ingest one file -> returns "skipped" / "indexed" / "updated" / "empty"
def ingest_file(path : str, department : str = "all", access_level : str = "internal", uploaded_by : str = "system") -> str:
    filename = os.path.basename(path)
    file_hash = get_file_hash(path)
    db = SessionLocal()
    try:
        record = db.query(DocumentRecord).filter(DocumentRecord.filename == filename).first()

        # Same file + same metadata -> nothing to do
        if record and record.file_hash == file_hash and record.department == department and record.access_level == access_level:
            return "skipped"

        pages = load_file(path)
        chunks = splitter.split_documents(pages)
        if not chunks:
            return "empty"

        status = "indexed"
        if record is None:
            record = DocumentRecord(filename=filename, file_hash=file_hash)
            db.add(record)
            db.flush()
        else:
            # File changed -> remove its old chunks, then re-index
            old_ids = [f"{record.id}_{i}" for i in range(record.chunk_count)]
            vectorstore.delete(ids=old_ids)
            status = "updated"

        ids = []
        for i, chunk in enumerate(chunks):
            chunk_id = f"{record.id}_{i}"
            page = chunk.metadata.get("page")
            chunk.metadata = {
                "doc_id" : record.id,
                "chunk_id" : chunk_id,
                "source" : filename,
                "page" : int(page) + 1 if page is not None else 0,
                "department" : department,
                "access_level" : access_level,
            }
            ids.append(chunk_id)

        vectorstore.add_documents(chunks, ids=ids)

        record.file_hash = file_hash
        record.department = department
        record.access_level = access_level
        record.chunk_count = len(chunks)
        record.uploaded_by = uploaded_by
        db.commit()
        return status
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()

# 📂Folder layout : data/documents/<department>/<access_level>/file.pdf
if __name__ == "__main__":
    for root, dirs, files in os.walk(DOCS_DIR):
        rel = os.path.relpath(root, DOCS_DIR)
        parts = [] if rel == "." else rel.split(os.sep)
        department = parts[0].lower() if len(parts) > 0 else "all"
        access_level = parts[1].lower() if len(parts) > 1 else "internal"
        if access_level not in ALLOWED_LEVELS:
            access_level = "internal"
        for name in sorted(files):
            if name.lower().endswith((".pdf", ".docx", ".txt", ".md")):
                status = ingest_file(os.path.join(root, name), department, access_level)
                print(f"📄 {name} -> {status} (dept={department}, level={access_level})")
