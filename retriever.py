#Enterprise RAG - Hybrid retrieval + reranking + access control
from config import ROLE_LEVELS, TOP_K_CANDIDATES, RERANK_TOP_N, RERANK_MODEL
import re
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from store import vectorstore

# 🔁Reranker
reranker = CrossEncoder(RERANK_MODEL)

# 🔐Metadata filter built from the user's role + department
def build_filter(user : dict, department : str | None = None) -> dict:
    conditions = [{"access_level" : {"$in" : ROLE_LEVELS[user["role"]]}}]
    if user["role"] != "admin":
        conditions.append({"department" : {"$in" : [user["department"], "all"]}})
    if department:
        conditions.append({"department" : department})
    return {"$and" : conditions} if len(conditions) > 1 else conditions[0]

def tokenize(text : str) -> list:
    return re.findall(r"\w+", text.lower())

# 🧠Semantic search
def vector_search(question : str, where : dict) -> list:
    return vectorstore.similarity_search(question, k=TOP_K_CANDIDATES, filter=where)

# 🔎Keyword search (BM25 over the chunks this user is allowed to see)
def bm25_search(question : str, where : dict) -> list:
    data = vectorstore.get(where=where, include=["documents", "metadatas"])
    if not data["documents"]:
        return []
    corpus = [tokenize(text) for text in data["documents"]]
    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(tokenize(question))
    top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:TOP_K_CANDIDATES]
    return [Document(page_content=data["documents"][i], metadata=data["metadatas"][i]) for i in top if scores[i] > 0]

# 🔀Reciprocal Rank Fusion (merges vector + BM25 results)
def rrf_fuse(result_lists : list, k : int = 60) -> list:
    scores = {}
    docs = {}
    for results in result_lists:
        for rank, doc in enumerate(results):
            key = doc.metadata["chunk_id"]
            docs[key] = doc
            scores[key] = scores.get(key, 0) + 1 / (k + rank + 1)
    ordered = sorted(scores, key=scores.get, reverse=True)
    return [docs[key] for key in ordered]

# 🏆Cross-encoder reranking
def rerank(question : str, docs : list) -> list:
    if not docs:
        return []
    scores = reranker.predict([(question, doc.page_content) for doc in docs])
    ranked = sorted(zip(docs, scores), key=lambda pair: pair[1], reverse=True)
    for doc, score in ranked:
        doc.metadata["score"] = float(score)
    return [doc for doc, score in ranked[:RERANK_TOP_N]]

# 🚀Full pipeline : filter -> vector + BM25 -> fuse -> rerank
def retrieve(question : str, user : dict, department : str | None = None) -> list:
    where = build_filter(user, department)
    fused = rrf_fuse([vector_search(question, where), bm25_search(question, where)])
    return rerank(question, fused)
