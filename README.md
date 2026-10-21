# Enterprise IT Support Agentic RAG Copilot

An agentic retrieval-augmented generation (RAG) support assistant for internal IT documentation.

An end-to-end Forward Deployed Engineer (FDE) project that demonstrates how an organization can turn internal IT documentation into a conversational support assistant. The application provides a web UI, a FastAPI backend, a LangGraph workflow, private knowledge retrieval, and optional external web search.
> **Project status:** Working development baseline. The API, local Qwen/Ollama response generation, private knowledge retrieval, direct greeting handling, and safe insufficient-evidence fallback have been manually exercised. Automated tests, full production hardening, and deployment have not yet been verified.

## Table of Contents
- Business Problem
- Features
- Architecture
- Workflow
- Technology Stack
- Project Structure
- Prerequisites
- Local Setup
- Run the Application
- API Endpoints
- Demo Scenarios
- Security Notes
- Testing
- Known Limitations and Roadmap
## 1. Business Problem
Consider NovaRetail, a fictional company whose IT team maintains documentation for VPN access, password policies, MFA, software installation, laptop troubleshooting, and service-desk procedures.

Employees may struggle to find the right document, and a general-purpose chatbot can provide unsupported answers. Some questions also require information that is not available in internal documentation, such as current vendor incidents.

The goal is to build an IT Support Copilot that:
1. Retrieves relevant internal documentation before answering IT questions.
2. Generates answers grounded in retrieved knowledge when evidence is sufficient.
3. Attempts external search when the private knowledge base cannot support an answer and Tavily is configured.
4. Returns a safe fallback when reliable evidence is unavailable.
5. Shows a trace of the workflow's decisions.
6. Allows authorized users to upload supported documents for indexing.
## 2. Current Features
- Web interface: Chat UI with an agent trace panel and document-upload interface.
- REST API: FastAPI endpoints for health checks, chat, and document ingestion.
- Local language model: Qwen 3 8B served through Ollama, avoiding a required paid OpenAI API.
- Private knowledge retrieval: Hugging Face sentence-transformers/all-MiniLM-L6-v2 embeddings and Pinecone vector search.
- Workflow orchestration: LangGraph routes greetings directly and IT questions through private retrieval.
- External search fallback: Tavily search is attempted when the private knowledge base does not produce an answer, provided a Tavily API key is configured.
- Safe fallback: If useful evidence is unavailable or answer generation fails, the application returns an insufficient-evidence message instead of intentionally inventing an answer.
- Document ingestion: Supported files are loaded, chunked, embedded, and indexed.
- Basic audit logging: SQLite-based audit logging for chat requests.
Features such as a complete evidence-grading loop, query rewriting/retries, comprehensive automated tests, and production deployment are not claimed as implemented in the current workflow.
## 3. Architecture
Employee
   |
   v
HTML / CSS / JavaScript
   |
   | POST /api/chat
   v
FastAPI
   |
   v
LangGraph workflow
   |
   +--> Direct response for recognized greetings
   |
   +--> Retrieve private knowledge
   |        |
   |        v
   |   MiniLM embeddings + Pinecone
   |        |
   |        v
   |   Local Qwen 3 via Ollama
   |
   +--> If private evidence is insufficient
            |
            v
       Tavily web search (optional)
            |
            v
       Local Qwen 3 via Ollama
            |
            v
       Answer with web citations or safe fallback
## 4. Current Agent Workflow
The current workflow follows this decision path:
Question
   |
   v
Rule-based routing
   |
   +--> Recognized greeting --> Direct response
   |
   +--> IT/support question
              |
              v
       Retrieve private KB
              |
              v
       Generate answer from KB
              |
        Answer sufficient?
          /          \
        Yes           No
         |             |
         v             v
   Return KB answer   Attempt Tavily search
                            |
                       Usable evidence?
                         /       \
                       Yes        No
                        |          |
                        v          v
                 Generate answer  Safe fallback
                 from web evidence
The trace returned by the API helps demonstrate which path ran. The current implementation uses the model's response marker to identify insufficient evidence; it does not yet implement a separate, structured evidence-grading node or query-rewrite/retry loop.
## 5. Technology Stack
Layer	Technology	Purpose
Workflow orchestration	LangGraph	Stateful workflow and conditional routing
Local LLM	Qwen 3 8B via Ollama	Local answer generation
Embeddings	Hugging Face sentence-transformers/all-MiniLM-L6-v2	Local 384-dimensional text embeddings
Vector database	Pinecone	Store and retrieve document chunks
External search	Tavily	Optional fallback search
API	FastAPI	REST endpoints
Frontend	HTML, CSS, JavaScript	Chat and document-upload UI
Audit	SQLite	Basic request audit records
Language	Python	Application and workflow implementation


## 6. Project Structure

| Path | Purpose |
| --- | --- |
| `app/api/` | FastAPI routes for chat, health checks, and document ingestion |
| `app/core/` | Application configuration and logging |
| `app/rag/` | Retrieval, vector store, workflow state, and LangGraph orchestration |
| `app/services/` | Document ingestion and audit logging |
| `data/sample_kb/` | Sample IT handbook and service-desk runbook |
| `static/` | Frontend CSS and JavaScript |
| `templates/` | HTML page template |
| `tests/` | Retrieval and RAG test modules |
| `uploads/` | Uploaded documents (created/used by the application) |
| `ingest_sample_kb.py` | Indexes the sample knowledge base |
| `run.py` | Starts the application |
| `requirements.txt` | Python dependencies |
| `.gitignore` | Files excluded from version control |

The project also includes the `LICENSE` and this `README.md` at its root.
## 7. Prerequisites
- Windows, macOS, or Linux
- Python 3.11 recommended
- Conda or Python virtual environment
- Ollama installed and running
- Qwen model downloaded in Ollama (for example, qwen3:8b)
- Pinecone account and API key
- Tavily API key if external web fallback is required
The local LLM and embeddings do not require an OpenAI API key. Pinecone and Tavily are external services and may have their own usage limits or pricing; check their current plans before relying on them.
## 8. Local Setup
### Step 1 — Open the project directory
cd C:\Users\KHUSHI\NewFolder\Enterprise-IT-Support-Agentic-RAG-Copilot
Use the actual project directory if yours is different.
### Step 2 — Activate your Conda environment
If you already created the project environment:
conda activate project
Alternatively, create a new environment with Python 3.11:
conda create -n project python=3.11 -y
conda activate project
### Step 3 — Install dependencies
python -m pip install -r requirements.txt
### Step 4 — Configure environment variables
Create a local .env file in the project root. Use the variable names expected by app/core/config.py. The exact settings and defaults in that file are authoritative.

Example template (replace placeholders with your own values; never commit actual secrets):
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=enterprise-it-support-local-new
PINECONE_NAMESPACE=fde-it-support-rag

EMBEDDING_MODEL=all-MiniLM-L6-v2

TAVILY_API_KEY=your_tavily_api_key_if_used
ADMIN_API_KEY=replace_with_a_long_random_secret

APP_ENV=development
Do not add real credentials to this README or commit them to version control. If the configuration code uses different names, follow app/core/config.py and your existing .env rather than blindly copying this example.
### Step 5 — Start Ollama and verify the model
Ensure Ollama is running, then check:
ollama list
If Qwen 3 8B is not downloaded and your computer has sufficient memory:
ollama pull qwen3:8b
### Step 6 — Ingest sample knowledge
python ingest_sample_kb.py
Run this when you need to index or refresh the sample documents. Avoid repeatedly ingesting the same files unless your ingestion logic handles duplicate document IDs.
## 9. Run the Application
Start the API and UI:
python run.py
Open the application:
- Web UI: http://127.0.0.1:8000
- FastAPI Swagger docs: http://127.0.0.1:8000/docs
- Health endpoint: http://127.0.0.1:8000/api/health
The local Qwen model can take several seconds to respond, depending on hardware and model settings.
## 10. API Endpoints
GET /api/health
Checks whether the API is running.
POST /api/chat
Example request:
{
  "question": "How do I connect to the company VPN?"
}
The response includes:
- answer: generated answer or safe fallback
- source_used: response source label
- trace: workflow events
- citations: private knowledge or web references, when available
- rewritten_query: current query used by the workflow (currently usually the original question)
POST /api/ingest
Uploads and indexes a supported document. The request requires the X-Admin-Key header to match the configured admin key. Supported file types and size limits are enforced by the ingestion endpoint.
## 11. Demo Scenarios
Demo A — Private knowledge base
Question:
How do I connect to the company VPN?

Expected behavior: the system retrieves internal handbook content and answers from the private KB when the retrieved evidence is sufficient.
Demo B — Direct greeting
Question:
Hi

Expected behavior: the rule-based router returns a direct greeting without retrieval or an LLM call.
Demo C — Current external information
Question:
What is the latest Microsoft Teams outage guidance?

Expected behavior: the system attempts Tavily search if configured. If the search results do not provide sufficient usable evidence, it returns a safe fallback rather than claiming to know the latest status.
Demo D — Protected document ingestion
Try uploading a supported text document without an admin key.

Expected behavior: the endpoint rejects the request with 401 Unauthorized.
## 12. Security Notes
- Keep .env out of version control.
- Never hardcode API keys or paste them into issues, screenshots, or public repositories.
- Use a long, unique admin key; do not use placeholder values in production.
- Restrict the ingestion endpoint to trusted administrators.
- Validate uploaded content and consider request-size limits at the server or reverse proxy.
- Do not expose stack traces or secret values to API clients.
- Review audit data handling and retention before using real employee questions.
- Treat retrieved documents and web content as untrusted input.
- This project has not yet undergone a full security audit and should not be used with sensitive production data without further review.
## 13. Testing
The API and several workflow paths have been manually exercised during development. Automated tests have not yet been confirmed to run successfully.

Once test functions are present in the tests/ directory, run:
python -m pytest -q
A previous run reported no tests ran, so automated test coverage remains a work item. Before production use, add tests for retrieval, ingestion, routing, fallback behavior, API validation, and unauthorized uploads.
## 14. Known Limitations and Roadmap
Current limitations:
- Rule-based routing recognizes a fixed set of greetings; it is not a general intent classifier.
- The workflow does not yet have a dedicated structured evidence-grading node.
- Query rewriting and retry loops are not implemented in the current workflow.
- External search quality depends on Tavily configuration and result relevance.
- Citation deduplication, source quality, and freshness need additional validation.
- Automated tests and deployment have not been completed or verified.
- The current application is a development baseline, not a production-hardened service.
Suggested next steps:
1. Add deterministic automated tests for the workflow and API.
2. Add structured evidence grading and a bounded query-rewrite/retrieval retry.
3. Improve source citation quality and remove duplicate citations.
4. Add upload and authentication hardening, rate limiting, and clearer operational logging.
5. Measure latency and retrieval quality with a small evaluation dataset.
6. Add screenshots, a short demo recording, and verified deployment instructions.
7. Evaluate Docker packaging and deployment only after the local setup is reproducible.
## License
See the LICENSE file in this repository.
