from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.embeddings import JinaEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

load_dotenv()

embeddings = JinaEmbeddings(model="jina-embeddings-v2-base-en")

documents = [
    Document(
        page_content="Product SKU-7742X is our flagship router. It supports gigabit speeds and advanced QoS features.",
        metadata={"type": "product"},
    ),
    Document(
        page_content="For network connectivity issues, first check the ethernet cable and router status lights.",
        metadata={"type": "troubleshooting"},
    ),
    Document(
        page_content="Error code E_CONN_REFUSED indicates the server rejected the connection. Check firewall settings.",
        metadata={"type": "error"},
    ),
    Document(
        page_content="QoS ensures critical applications get priority bandwidth to reduce latency and packet loss.",
        metadata={"type": "info"},
    ),
    Document(
        page_content="If the router is not powering on, verify that the power adapter is securely connected and the power outlet is working.",
        metadata={"type": "troubleshooting"},
    ),
    Document(
        page_content="Error code E_TIMEOUT indicates that the connection request exceeded the allowed response time. Check network stability.",
        metadata={"type": "error"},
    ),
]

print(f"Loaded {len(documents)} documents")

vectorstore = Chroma.from_documents(
    documents, embeddings, collection_name="hybrid_test"
)

# create vector retriever
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

print("Vector retriever ready")

bm25_retriever = BM25Retriever.from_documents(documents, k=3)

print("BM25 retriever ready")

ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever], weights=[0.5, 0.5]
)

print("Hybrid retriever ready")


def test_query(query, name, retriever):
    """Test a query and show result"""
    results = retriever.invoke(query)
    print(f'\\n{name} - Query : "{query}"')
    for i, doc in enumerate(results[:3]):
        preview = doc.page_content[:80] + "..."
        print(f" {i+1}. {preview}")
    return results


test_quries = [
    "SKU-7742X specifications",
    "E_CONN_REFUSED error",
    "What to do if the router is not powering on",
    "QoS ensures",
    "Network connectivity issue",
]

for query in test_quries:
    print("=" * 60)

    vector_results = test_query(query, "VECTOR", vector_retriever)
    bm25_results = test_query(query, "BM25", bm25_retriever)
    hybrid_results = test_query(query, "HYBRID", ensemble_retriever)
