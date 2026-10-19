
from app.rag.vectorstore import get_retriever

def main():
    retriever = get_retriever()

    question = "What should an employee do after five failed login attempts?"

    results = retriever.invoke(question)

    print(f"\nQuestion: {question}")
    print(f"Retrieved chunks: {len(results)}\n")

    for i, doc in enumerate(results, start=1):
        print(f"--- Result {i} ---")
        print(doc.page_content)
        print("Metadata:", doc.metadata)
        print()

if __name__ == "__main__":
    main()