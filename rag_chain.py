#Enterprise RAG - Answer generation with citations
from config import LLM_MODEL
from langchain_google_genai import ChatGoogleGenerativeAI
from retriever import retrieve

# 🧠LLM Setup
llm = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0.1)

def get_text(response) -> str:
    return response.text

def build_context(docs : list) -> str:
    blocks = []
    for i, doc in enumerate(docs, start=1):
        blocks.append(f"[{i}] (Source: {doc.metadata['source']}, page {doc.metadata['page']})\n{doc.page_content}")
    return "\n\n".join(blocks)

# 📌Pick only the sources the model actually cited, e.g. [1] or [1, 3]
def get_cited_numbers(answer : str, total : int) -> list:
    import re
    cited = set()
    for group in re.findall(r"\[([\d,\s]+)\]", answer):
        for n in group.split(","):
            if n.strip().isdigit() and 1 <= int(n) <= total:
                cited.add(int(n))
    return sorted(cited)

# 💬Main function used by the API, chat and evaluation scripts
def ask(question : str, user : dict, department : str | None = None) -> dict:
    docs = retrieve(question, user, department)
    if not docs:
        return {
            "answer" : "I could not find this in the documents you have access to.",
            "sources" : [],
            "contexts" : [],
        }

    prompt = (
        "You are an enterprise knowledge assistant. "
        "Answer ONLY from the context below. "
        "After every fact, cite the source number like [1] or [2]. "
        "If the answer is not in the context, say you could not find it in the documents.\n\n"
        f"Context:\n{build_context(docs)}\n\n"
        f"Question: {question}\n\n"
        "Give a clear, friendly, and precise answer."
    )
    answer = get_text(llm.invoke(prompt))

    numbers = get_cited_numbers(answer, len(docs)) or list(range(1, len(docs) + 1))
    sources = []
    for n in numbers:
        doc = docs[n - 1]
        sources.append({
            "id" : n,
            "source" : doc.metadata["source"],
            "page" : doc.metadata["page"],
            "department" : doc.metadata["department"],
            "access_level" : doc.metadata["access_level"],
            "snippet" : doc.page_content[:200],
        })
    return {"answer" : answer, "sources" : sources, "contexts" : [doc.page_content for doc in docs]}
