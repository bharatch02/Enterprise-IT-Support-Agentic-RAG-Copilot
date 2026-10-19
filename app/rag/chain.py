
from langchain_ollama import ChatOllama
from app.rag.vectorstore import get_retriever

llm = ChatOllama(
    model="qwen3:8b",
    temperature=0,
)

def ask_question(question: str) -> dict:
    retriever = get_retriever()
    documents = retriever.invoke(question)

    unique_documents = []
    seen = set()

    for doc in documents:
        key = (
            doc.metadata.get("source"),
            doc.metadata.get("start_index"),
            doc.page_content,
        )
        if key not in seen:
            seen.add(key)
            unique_documents.append(doc)

    documents = unique_documents

    if not documents:
        return {
            "question": question,
            "answer": "I couldn't find relevant information in the IT handbook.",
            "sources": [],
        }

    context = "\n\n".join(
        doc.page_content for doc in documents
    )

    prompt = f"""
You are an Enterprise IT Support Assistant.
Answer the question using ONLY the context below.
If the answer is not available in the context, say so.
Do not invent company policies.

Context:
{context}

Question:
{question}

Answer:
"""

    response = llm.invoke(prompt)

    return {
        "question": question,
        "answer": response.content.strip(),
        "sources": [
            doc.metadata.get("source", "Unknown")
            for doc in documents
        ],
    }
