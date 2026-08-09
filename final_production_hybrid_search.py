from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.embeddings import JinaEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

load_dotenv()

# Documents with both semantic content AND specific identifiers
documents = [
    Document(
        page_content="Product SKU-7742X is our flagship router. It supports "
        "gigabit speeds and advanced QoS features.",
        metadata={"type": "product"},
    ),
    Document(
        page_content="For network connectivity issues, first check the "
        "ethernet cable and router status lights.",
        metadata={"type": "troubleshooting"},
    ),
    Document(
        page_content="Error code E_CONN_REFUSED indicates the server "
        "rejected the connection. Check firewall settings.",
        metadata={"type": "error"},
    ),
    Document(
        page_content="The authentication process requires valid credentials. "
        "Use OAuth2 for secure API access.",
        metadata={"type": "auth"},
    ),
    Document(
        page_content="Router configuration guide: Access the admin panel "
        "at 192.168.1.1 to modify settings.",
        metadata={"type": "config"},
    ),
    Document(
        page_content="WCAG 2.1 compliance requires all images to have "
        "alt text and sufficient color contrast.",
        metadata={"type": "compliance"},
    ),
]


class HybridRetriever:
    """Production hybrid retriever"""

    def __init__(self, documents: list[Document], bm25_weight: float = 0.5, k: int = 5):
        self.k = k
        self.bm25_weight = bm25_weight
        self.vector_weight = 1 - bm25_weight

        # initialize embeddings
        self.embeddings = JinaEmbeddings(model="jina-embedding-v2-base-en")

        # create vector store
        self.vector_store = Chroma.from_documents(
            documents, self.embeddings, collection_name="hybrid_search"
        )
        self.vector_retriever = self.vector_store.as_retriever(search_kwargs={"k": k})

        # create bm25 retriever
        self.bm25_retriever = BM25Retriever.from_documents(documents, k=k)

    def search(self, query: str) -> list[Document]:
        """Run hybrid search using weighted RRF"""
        hybrid_retriever = EnsembleRetriever(
            retrievers=[self.bm25_retriever, self.vector_retriever],
            weights=[0.5, 0.5],
            c=60,
        )
        result = hybrid_retriever.invoke(query)
        return result

    def add_documents(self, document: list[Document]):
        """Add new doc to both retriever"""
        # Add to vector store
        self.vector_store.add_documents(documents)

        # recreate BM25
        all_docs = self.vector_store.get()
        self.bm25_retriever = BM25Retriever.from_documents(
            [Document(page_content=doc) for doc in all_docs["documents"]]
        )


retriever = HybridRetriever(documents, bm25_weight=0.5, k=4)
results = retriever.search("SKU-7742X specifications")

for doc in results:
    print(doc.page_content[:100])
