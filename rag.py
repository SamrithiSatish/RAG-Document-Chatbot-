import os
from typing import TypedDict, List
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import chromadb
import anthropic
from langgraph.graph import StateGraph, START, END

load_dotenv()


CHROMA_DIR = "chroma_db"
TOP_K = 3
MAX_RETRIES = 2


embedder = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = chroma_client.get_or_create_collection("documents")
claude_client = anthropic.Anthropic()



class RAGState(TypedDict):
    question: str
    search_query: str
    retrieved_docs: List[str]
    answer: str
    retry_count: int



def retrieve(state: RAGState) -> RAGState:
    query = state.get("search_query") or state["question"]
    print(f"  [retrieve] Searching for: {query}")
    query_embedding = embedder.encode([query]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=TOP_K,
    )
    docs = results["documents"][0] if results["documents"] else []
    return {**state, "retrieved_docs": docs}


def grade_documents(state: RAGState) -> str:
    """Decide whether retrieved docs are relevant enough to answer, or whether to retry."""
    if state.get("retry_count", 0) >= MAX_RETRIES:
        print("  [grading] Max retries reached, generating anyway.")
        return "generate"

    if not state["retrieved_docs"]:
        print("  [grading] No docs retrieved, retrying.")
        return "retrieve_again"

    context = "\n\n".join(state["retrieved_docs"])
    grading_prompt = f"""You are grading whether retrieved context is relevant enough to answer a question.

Question: {state['question']}

Retrieved context:
{context}

Does this context contain information that could help answer the question? Reply with only one word: "yes" or "no"."""

    response = claude_client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=5,
        messages=[{"role": "user", "content": grading_prompt}],
    )
    verdict = response.content[0].text.strip().lower()
    print(f"  [grading] Verdict: {verdict}")

    return "generate" if "yes" in verdict else "retrieve_again"


def retrieve_again(state: RAGState) -> RAGState:
    """Rewrite the query to try to get better retrieval results, then loop back."""
    current_retry = state.get("retry_count", 0) + 1
    print(f"  [retry {current_retry}] Rewriting query...")

    rewrite_prompt = f"""The following search query did not retrieve relevant results from a document database.
Rewrite it as a clearer, more specific search query that might match the source material better.

Original question: {state['question']}
Previous search query: {state.get('search_query') or state['question']}

Reply with only the rewritten query, nothing else."""

    response = claude_client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=50,
        messages=[{"role": "user", "content": rewrite_prompt}],
    )
    new_query = response.content[0].text.strip()
    print(f"  [retry {current_retry}] New query: {new_query}")

    return {
        **state,
        "search_query": new_query,
        "retry_count": current_retry,
    }


def generate(state: RAGState) -> RAGState:
    context = "\n\n".join(state["retrieved_docs"]) if state["retrieved_docs"] else "No relevant context found."

    prompt = f"""Answer the question based only on the context below. If the context doesn't contain the answer, say so.

Context:
{context}

Question: {state['question']}"""

    response = claude_client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}],
    )
    answer = response.content[0].text
    return {**state, "answer": answer}



graph = StateGraph(RAGState)
graph.add_node("retrieve", retrieve)
graph.add_node("generate", generate)
graph.add_node("retrieve_again", retrieve_again)

graph.add_edge(START, "retrieve")
graph.add_conditional_edges(
    "retrieve",
    grade_documents,
    {
        "generate": "generate",
        "retrieve_again": "retrieve_again",
    },
)
graph.add_edge("retrieve_again", "retrieve")
graph.add_edge("generate", END)

app_graph = graph.compile()



if __name__ == "__main__":
    print("RAG chatbot ready. Type 'quit' to exit.\n")
    while True:
        question = input("Ask a question: ")
        if question.lower() in ("quit", "exit"):
            break
        result = app_graph.invoke({"question": question, "retry_count": 0})
        print("\n--- ANSWER ---")
        print(result["answer"])
        print("\n--- SOURCES USED ---")
        for i, doc in enumerate(result["retrieved_docs"]):
            print(f"[{i}] {doc[:100]}...")
        print()