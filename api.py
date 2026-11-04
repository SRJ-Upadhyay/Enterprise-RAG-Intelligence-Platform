#Enterprise RAG - FastAPI service
from config import UPLOAD_DIR, ALLOWED_LEVELS, ROLE_LEVELS
import os
import time
import json
import shutil
from fastapi import FastAPI, UploadFile, File, Form, Header, HTTPException, Depends
from pydantic import BaseModel
from database import SessionLocal, UserRecord, DocumentRecord, QueryLog
from ingest import ingest_file
from rag_chain import ask
from sqlalchemy import func

app = FastAPI(title="Enterprise RAG Intelligence Platform")

class AskRequest(BaseModel):
    question : str
    department : str | None = None

# 🔐Auth : every request carries an X-API-Key header
def get_current_user(x_api_key : str = Header(...)) -> dict:
    db = SessionLocal()
    try:
        record = db.query(UserRecord).filter(UserRecord.api_key == x_api_key).first()
        if record is None:
            raise HTTPException(status_code=401, detail="Invalid API key")
        return {"username" : record.username, "role" : record.role, "department" : record.department}
    finally:
        db.close()

@app.get("/health")
def health():
    return {"status" : "ok"}

# 💬Ask a question
@app.post("/ask")
def ask_question(req : AskRequest, user : dict = Depends(get_current_user)):
    start = time.time()
    result = ask(req.question, user, req.department)
    latency = round((time.time() - start) * 1000, 1)

    db = SessionLocal()
    try:
        db.add(QueryLog(
            username=user["username"],
            question=req.question,
            answer=result["answer"],
            sources=json.dumps(result["sources"]),
            latency_ms=latency,
        ))
        db.commit()
    finally:
        db.close()
    return {"answer" : result["answer"], "sources" : result["sources"], "latency_ms" : latency}

# 📚List the documents this user can see
@app.get("/documents")
def list_documents(user : dict = Depends(get_current_user)):
    db = SessionLocal()
    try:
        query = db.query(DocumentRecord).filter(DocumentRecord.access_level.in_(ROLE_LEVELS[user["role"]]))
        if user["role"] != "admin":
            query = query.filter(DocumentRecord.department.in_([user["department"], "all"]))
        return [
            {"id" : d.id, "filename" : d.filename, "department" : d.department,
             "access_level" : d.access_level, "chunks" : d.chunk_count}
            for d in query.all()
        ]
    finally:
        db.close()

# 📤Upload a document (managers and admins only)
@app.post("/upload")
def upload_document(
    file : UploadFile = File(...),
    department : str = Form("all"),
    access_level : str = Form("internal"),
    user : dict = Depends(get_current_user),
):
    if user["role"] not in ["manager", "admin"]:
        raise HTTPException(status_code=403, detail="Only managers and admins can upload")
    if access_level not in ALLOWED_LEVELS:
        raise HTTPException(status_code=400, detail=f"access_level must be one of {ALLOWED_LEVELS}")
    if user["role"] == "manager" and department != user["department"]:
        raise HTTPException(status_code=403, detail="Managers can only upload to their own department")

    filename = os.path.basename(file.filename)
    path = os.path.join(UPLOAD_DIR, filename)
    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        status = ingest_file(path, department, access_level, user["username"])
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"filename" : filename, "status" : status}


# 🙋Who am I (the Streamlit UI calls this after sign in)
@app.get("/me")
def me(user : dict = Depends(get_current_user)):
    return user

# 📊Usage stats (admin only)
@app.get("/stats")
def stats(user : dict = Depends(get_current_user)):
    if user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Only admins can view analytics")
    db = SessionLocal()
    try:
        total = db.query(func.count(QueryLog.id)).scalar() or 0
        average = db.query(func.avg(QueryLog.latency_ms)).scalar() or 0
        by_user = db.query(QueryLog.username, func.count(QueryLog.id)).group_by(QueryLog.username).all()
        recent = db.query(QueryLog).order_by(QueryLog.created_at.desc()).limit(20).all()
        return {
            "total_queries" : total,
            "avg_latency_ms" : round(average, 1),
            "by_user" : [{"username" : name, "queries" : count} for name, count in by_user],
            "recent" : [
                {"username" : r.username, "question" : r.question, "latency_ms" : r.latency_ms, "created_at" : r.created_at.isoformat()}
                for r in recent
            ],
        }
    finally:
        db.close()