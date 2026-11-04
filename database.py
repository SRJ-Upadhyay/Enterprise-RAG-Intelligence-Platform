#Enterprise RAG - PostgreSQL models
from config import DATABASE_URL
import secrets
from datetime import datetime
from rich import print
from sqlalchemy import create_engine, Column, Integer, String, Text, Float, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker

# 🗄️Engine and session
engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

# 👤Users (role decides which access levels they can read)
class UserRecord(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    username = Column(String(100), unique=True, nullable=False)
    api_key = Column(String(64), unique=True, nullable=False)
    role = Column(String(20), nullable=False)
    department = Column(String(50), nullable=False)

# 📄Documents (file_hash is used for incremental indexing)
class DocumentRecord(Base):
    __tablename__ = "documents"
    id = Column(Integer, primary_key=True)
    filename = Column(String(255), unique=True, nullable=False)
    file_hash = Column(String(64), nullable=False)
    department = Column(String(50), default="all")
    access_level = Column(String(20), default="internal")
    chunk_count = Column(Integer, default=0)
    uploaded_by = Column(String(100), default="system")
    uploaded_at = Column(DateTime, default=datetime.utcnow)

# 📝Query logs
class QueryLog(Base):
    __tablename__ = "query_logs"
    id = Column(Integer, primary_key=True)
    username = Column(String(100))
    question = Column(Text)
    answer = Column(Text)
    sources = Column(Text)
    latency_ms = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(engine)

# 🌱Demo users
def seed_demo_users():
    db = SessionLocal()
    try:
        if db.query(UserRecord).count() > 0:
            print("Users already exist - skipping seed")
            return
        demo_users = [
            ("admin", "admin", "all"),
            ("hr_manager", "manager", "hr"),
            ("dev_employee", "employee", "engineering"),
        ]
        for username, role, department in demo_users:
            key = secrets.token_hex(16)
            db.add(UserRecord(username=username, api_key=key, role=role, department=department))
            print(f"🔑 {username} ({role}, {department}) -> API key : {key}")
        db.commit()
    finally:
        db.close()

if __name__ == "__main__":
    seed_demo_users()
