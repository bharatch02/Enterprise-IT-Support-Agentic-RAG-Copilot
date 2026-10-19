
from app.rag.chain import ask_question

question = "What should an employee do after five failed login attempts?"

result = ask_question(question)

print("\nQuestion:", result["question"])
print("\nAnswer:", result["answer"])
print("\nSources:")

for source in result["sources"]:
    print("-", source)
