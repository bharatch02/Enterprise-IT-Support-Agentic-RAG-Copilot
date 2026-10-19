from pathlib import Path
from app.services.ingestion import load_file, chunk_documents
from app.rag.vectorstore import add_documents

docs = load_file(Path("data/sample_kb/company_it_handbook.md"))
chunks = chunk_documents(docs)

print(f"Documents loaded: {len(docs)}")
print(f"Chunks created: {len(chunks)}")

add_documents(chunks)

print("Documents added to Pinecone successfully!")