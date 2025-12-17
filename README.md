# Enterprise RAG System

A production-grade Retrieval-Augmented Generation (RAG) system implementing best practices for enterprise deployments.

## 🌟 Features

### 📚 Document Processing
- **Smart Ingestion**: Support for PDF, Text, HTML, DOCX, and web URLs
- **Semantic Chunking**: Split documents at semantic boundaries using embeddings
- **Recursive Chunking**: Hierarchical text splitting with customizable separators
- **Markdown-Aware**: Special handling for markdown documents

### 🔍 Advanced Retrieval
- **Hybrid Search**: Combine vector search with BM25 keyword search
- **MMR Diversity**: Maximum Marginal Relevance for diverse results
- **Query Rewriting**: LLM-powered query improvement
- **Multi-Query Generation**: Generate query variations for better recall
- **HyDE**: Hypothetical Document Embeddings for better retrieval
- **Domain Routing**: Route queries to appropriate knowledge domains

### 🎯 Reranking & Compression
- **Cohere Reranker**: Production-grade reranking API
- **Cross-Encoder**: Local transformer-based reranking
- **LLM Reranker**: Flexible LLM-based relevance scoring
- **Contextual Compression**: Reduce noise in retrieved context

### ✨ Generation
- **Citations**: Automatic source attribution
- **Guardrails**: Input/output validation, PII filtering
- **Streaming**: Real-time response streaming
- **Multiple LLMs**: OpenAI, Anthropic, Groq support

### 📊 Evaluation
- **RAGAS Metrics**: Faithfulness, relevancy, correctness
- **Custom Metrics**: Extensible evaluation framework
- **Batch Evaluation**: Evaluate across test sets

### 🤖 Agentic Workflows
- **LangGraph Integration**: Multi-step reasoning agents
- **Adaptive RAG**: Dynamic retrieval strategies
- **Self-Correction**: Iterative query refinement
- **Memory Systems**: Buffer, summary, and vector-based memory

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone <repository-url>
cd RAG

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt
```

### Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit with your API keys
nano .env
```

Required API keys:
- `OPENAI_API_KEY` - For embeddings and LLM (required)
- `COHERE_API_KEY` - For Cohere reranking (optional)
- `ANTHROPIC_API_KEY` - For Claude models (optional)

### Usage

#### Command Line

```bash
# Run demo
python main.py --demo

# Ingest documents
python main.py --ingest ./documents/

# Query the system
python main.py --query "What is hybrid search?"

# Use agentic workflow
python main.py --query "Explain RAG best practices" --agent

# Start API server
python main.py --serve --port 8000
```

#### Python API

```python
from main import setup_rag_system, ingest_documents, query_rag

# Initialize system
system = setup_rag_system()

# Ingest documents
ingest_documents(system, "./documents/", source_type="directory")

# Query
result = query_rag(system, "What is semantic chunking?")
print(result["answer"])
print(result["citations"])
```

#### REST API

```bash
# Start server
uvicorn src.api.app:create_app --factory --reload

# Query endpoint
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is hybrid search?"}'

# Ingest endpoint
curl -X POST http://localhost:8000/api/v1/ingest \
  -H "Content-Type: application/json" \
  -d '{"source": "./documents/", "source_type": "directory"}'
```

## 📁 Project Structure

```
RAG/
├── main.py                  # Main entry point
├── requirements.txt         # Dependencies
├── pyproject.toml          # Project metadata
├── .env.example            # Environment template
├── README.md               # Documentation
└── src/
    ├── config/
    │   └── settings.py     # Centralized configuration
    ├── ingestion/
    │   ├── loaders.py      # Document loaders
    │   ├── chunkers.py     # Text chunking strategies
    │   └── pipeline.py     # Ingestion pipeline
    ├── vectorstore/
    │   ├── embeddings.py   # Embedding factory
    │   ├── vector_stores.py # Vector store factory
    │   └── hybrid_store.py # Hybrid search
    ├── retrieval/
    │   ├── retriever.py    # Production retriever
    │   ├── reranker.py     # Reranking strategies
    │   └── compressor.py   # Context compression
    ├── query/
    │   └── query_processor.py # Query enhancement
    ├── generation/
    │   ├── llm_factory.py  # LLM creation
    │   ├── prompts.py      # Prompt templates
    │   ├── rag_chain.py    # RAG chain with citations
    │   └── guardrails.py   # Safety guardrails
    ├── evaluation/
    │   ├── evaluator.py    # RAG evaluator
    │   └── metrics.py      # Custom metrics
    ├── agents/
    │   ├── memory.py       # Conversation memory
    │   └── agent.py        # Agentic workflows
    └── api/
        ├── app.py          # FastAPI application
        ├── routes.py       # API endpoints
        └── schemas.py      # Request/response models
```

## 🔧 Configuration

### Retrieval Settings

| Setting | Default | Description |
|---------|---------|-------------|
| `RETRIEVAL_TOP_K` | 10 | Number of documents to retrieve |
| `USE_MMR` | true | Enable MMR diversity |
| `MMR_LAMBDA` | 0.5 | Diversity vs relevance tradeoff |
| `HYBRID_ALPHA` | 0.7 | Vector vs keyword weight |

### Chunking Settings

| Setting | Default | Description |
|---------|---------|-------------|
| `CHUNK_STRATEGY` | recursive | Chunking strategy |
| `CHUNK_SIZE` | 1000 | Target chunk size |
| `CHUNK_OVERLAP` | 200 | Overlap between chunks |

### Generation Settings

| Setting | Default | Description |
|---------|---------|-------------|
| `LLM_MODEL` | gpt-4o-mini | LLM model |
| `LLM_TEMPERATURE` | 0.0 | Response temperature |
| `INCLUDE_CITATIONS` | true | Add source citations |

## 🧪 Evaluation

```python
from main import setup_rag_system, evaluate_rag

system = setup_rag_system()

test_data = [
    {
        "question": "What is hybrid search?",
        "ground_truth": "Hybrid search combines vector and keyword search."
    },
    # Add more test cases...
]

results = evaluate_rag(system, test_data)
print(f"Avg Faithfulness: {results['aggregate']['avg_faithfulness']:.2f}")
print(f"Avg Relevancy: {results['aggregate']['avg_answer_relevancy']:.2f}")
```

## 🤖 Agentic Workflows

The system supports LangGraph-based agentic workflows:

```python
from main import setup_rag_system, query_rag

system = setup_rag_system()

# Use agent for complex queries
result = query_rag(
    system,
    "Compare different chunking strategies and their tradeoffs",
    use_agent=True
)
```

The agent can:
- Route queries to appropriate strategies
- Iteratively refine searches
- Self-correct based on retrieved context
- Maintain conversation memory

## 📊 Monitoring with LangSmith

Enable LangSmith for tracing:

```bash
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your-key
LANGCHAIN_PROJECT=enterprise-rag
```

## 🛡️ Guardrails

The system includes safety guardrails:

- **Input Validation**: Length limits, blocked topics
- **PII Filtering**: Detect and mask personal information
- **Output Validation**: Hallucination detection, factuality checks
- **Content Safety**: Block harmful content generation

## 🔄 Memory Systems

Available memory types:

- **Buffer Memory**: Keep last N message pairs
- **Summary Memory**: Summarize old conversations
- **Vector Memory**: Semantic retrieval of relevant history
- **Entity Memory**: Track entities across conversation

## 📝 License

MIT License - see [LICENSE](LICENSE) for details.

## 🤝 Contributing

Contributions welcome! Please read our contributing guidelines.

## 📚 References

- [LangChain Documentation](https://python.langchain.com/)
- [RAGAS Framework](https://docs.ragas.io/)
- [LangGraph Guide](https://langchain-ai.github.io/langgraph/)