#Enterprise RAG - RAGAS evaluation
import config
import json
from rich import print
from ragas import evaluate, EvaluationDataset
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from rag_chain import ask, llm
from store import embeddings

# 🧑 Evaluate as admin so every document is visible
admin_user = {"username" : "evaluator", "role" : "admin", "department" : "all"}

# Step 1 - Load the test questions
with open("eval_dataset.json", "r", encoding="utf-8") as f:
    eval_items = json.load(f)

# Step 2 - Run the RAG pipeline on every question
samples = []
for item in eval_items:
    result = ask(item["question"], admin_user)
    print(f"✅ {item['question']}")
    samples.append({
        "user_input" : item["question"],
        "retrieved_contexts" : result["contexts"],
        "response" : result["answer"],
        "reference" : item["reference"],
    })
dataset = EvaluationDataset.from_list(samples)

# Step 3 - Score with RAGAS
ragas_llm = LangchainLLMWrapper(llm)
ragas_embeddings = LangchainEmbeddingsWrapper(embeddings)
answer_relevancy.strictness = 1   # Gemini does not support multiple generations per call

result = evaluate(
    dataset=dataset,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    llm=ragas_llm,
    embeddings=ragas_embeddings,
)

# Step 4 - Show and save
print("\n[bold green]📊 RAGAS Scores[/bold green]")
print(result)
result.to_pandas().to_csv("eval_results.csv", index=False)
print("Saved to eval_results.csv")
