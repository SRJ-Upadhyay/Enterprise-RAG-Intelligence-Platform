# Enterprise RAG Intelligence Platform

A company knowledge assistant that answers questions from internal documents, cites the exact page it used, and enforces role-based access control. It is built with LangChain, ChromaDB, hybrid retrieval, a cross-encoder reranker, FastAPI, PostgreSQL, Streamlit and RAGAS evaluation, and runs locally or in Docker.

## Features

- **Ingestion with incremental indexing.** PDF, DOCX, TXT and MD files are chunked and embedded. A file is re-indexed only when its content or access settings change.
- **Hybrid retrieval.** Semantic search (ChromaDB with MiniLM embeddings) and keyword search (BM25) are merged with reciprocal rank fusion, then reranked with a cross-encoder.
- **Grounded answers with citations.** Answers are written only from retrieved passages, each cited as a numbered source with file, page, department and access level.
- **Abstention.** Questions the documents do not answer get a clear "could not find it" response rather than a guess.
- **Access control.** Every search is filtered by the user's role and department before any text reaches the model. An employee cannot see confidential HR content, even by asking directly.
- **API with key authentication.** FastAPI endpoints for asking questions, listing documents, uploading files and viewing usage.
- **Streamlit interface.** Sign in with an API key, ask questions, browse documents, upload files and view analytics. The menu adapts to the user's role.
- **Evaluation with RAGAS.** Faithfulness, answer relevancy, context precision and context recall are scored on a labelled question set.

## Architecture

```
 Documents ──► Ingest ──► Chunk ──► Embed ──► ChromaDB
 (PDF/DOCX/TXT/MD)                              │
                                                ▼
 User ──► Streamlit UI ──► FastAPI ──► Access filter ──► Vector + BM25 search
                              │                                 │
                              │                        Reciprocal rank fusion
                              │                                 │
                              │                         Cross-encoder rerank
                              │                                 │
                              ▼                                 ▼
                        PostgreSQL ◄── Query logs ◄── Gemini answer + citations
                     (users, documents)
```

Ingestion runs offline. Questions run online: the access filter is applied before retrieval, so unauthorised passages never reach the prompt.

## Project structure

```
├── api.py              FastAPI endpoints and API key authentication
├── app.py              Streamlit interface
├── chat.py             Console chat for testing
├── config.py           Models, chunking, and role permissions
├── database.py         PostgreSQL tables and demo users
├── ingest.py           Document loading, chunking and incremental indexing
├── rag_chain.py        Answer generation with citations
├── retriever.py        Access filter, hybrid search and reranking
├── store.py            Embeddings and vector store
├── evaluate.py         RAGAS evaluation
├── eval_dataset.json   Labelled questions for evaluation
├── docker-compose.yml  API and database services
└── Dockerfile
```

## Setup

### Prerequisites

- Python 3.11 or newer, with [uv](https://docs.astral.sh/uv/)
- Docker Desktop (for PostgreSQL, and optionally the full stack)
- A Google Gemini API key

### 1. Install dependencies

```bash
uv venv
uv pip install -r requirements.txt
```

### 2. Configure environment

Copy `.env.example` to `.env` and add your key:

```
GOOGLE_API_KEY=your_gemini_key
DATABASE_URL=postgresql+psycopg2://postgres:postgres@localhost:5433/enterprise_rag
```

### 3. Start PostgreSQL

```bash
docker compose up db -d
```

### 4. Create tables and demo users

```bash
python database.py
```

This prints one API key per demo user: `admin`, `hr_manager` and `dev_employee`. Save them, since they print only once.

### 5. Add documents

Place files in a folder named for the department and access level:

```
data/documents/engineering/internal/handbook.pdf
data/documents/hr/confidential/hr_policy.txt
```

Then index them:

```bash
python ingest.py
```

Supported access levels are `public`, `internal` and `confidential`.

### 6. Run

```bash
python chat.py                      # console chat
uvicorn api:app --reload            # API at http://127.0.0.1:8000/docs
uv run streamlit run app.py         # UI at http://localhost:8501
```

### Docker

```bash
docker compose up --build -d
```

This starts PostgreSQL and the API on port 8000. Your `.env` file is injected at runtime and is never copied into the image.

## Access control

| Role | Access levels | Departments | Can upload |
|---|---|---|---|
| Employee | public, internal | own department and `all` | No |
| Manager | public, internal, confidential | own department and `all` | Own department |
| Admin | all | all | Any department |

The filter is applied inside the search itself, not by asking the model to behave. Tests confirm that an employee asking for a manager's bonus policy receives no HR passage, while a manager receives the answer with its source.

### Getting an API key

API keys are issued by an administrator. Each key is tied to one user, role and department.

- **Demo setup:** `python database.py` creates three demo users and prints their keys once. Use these only for local testing.
- **Production (planned):** an admin creates users through a protected `/users` endpoint, which returns a random key to hand over privately. Keys should be stored as hashes, support revocation, and be sent over HTTPS only. Single sign-on with the company's identity provider is the recommended long-term option.

Keys are never stored in the repository or the image. Share them through a secure channel, such as a password manager.

## Evaluation

Scores from RAGAS on a 12-question set drawn from the source textbook, including one out-of-scope question and one access-controlled question:

| Metric | Score | What it measures |
|---|---|---|
| Faithfulness | 0.90 | Answers are supported by the retrieved text |
| Answer relevancy | 0.87 | Answers address the question asked |
| Context precision | 0.70 | Share of retrieved passages that are useful |
| Context recall | 1.00 | Retrieved passages contain what the reference answers need |

**Known limitation:** context precision is the weakest score. Some retrieved passages are table-of-contents or glossary text rather than explanations. Reranking with section-aware filtering is the next improvement.

To reproduce:

```bash
python evaluate.py
```

Results are written to `eval_results.csv`.

## Tech stack

LangChain · Google Gemini · ChromaDB · sentence-transformers (MiniLM, cross-encoder) · BM25 · FastAPI · PostgreSQL · SQLAlchemy · Streamlit · RAGAS · Docker

## Limitations

- Gemini free-tier rate limits make full evaluation runs slow.
- Embeddings and reranking run on CPU, so first requests take longer while models load.
- The API key system is suited to a demo. A production deployment needs SSO and key rotation.
- Demo credentials are for local testing only and must be replaced before any real use.

## Licence

Add your preferred licence here, for example MIT.