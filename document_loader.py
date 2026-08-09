import os
import tempfile
import warnings
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    TextLoader,
    WebBaseLoader,
)
from langchain_core.documents import Document

warnings.filterwarnings("ignore", category=DeprecationWarning)

load_dotenv()


def load_text_file(file_path: str) -> str:
    # create a temporary file for demonstration
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as temp_file:
        temp_file.write(
            b"Hello, this is a sample text file.\n This is used for testing the TextLoader."
        )
        temp_file_path = temp_file.name

    try:
        # load text file using text loader
        loader = TextLoader(temp_file_path)
        documents = loader.load()

        for doc in documents:
            print("Document Content")
            print(doc)
            print(doc.page_content)
    finally:
        os.remove(temp_file_path)


def web_loader():
    loader = WebBaseLoader(
        "https://en.wikipedia.org/wiki/Web_scraping", bs_kwargs={"parse_only": None}
    )
    documents = loader.load()
    print(f"Loaded {len(documents)} document(s) from web")
    print(f"Source: {documents[0].metadata.get('source','N/A')}")
    print(f"Content Length: {len(documents[0].page_content)} characters")
    print(f"Preview: {documents[0].page_content[:200]}...")


def lazy_loader():

    # Create temporary directory woth sample files
    with tempfile.TemporaryDirectory() as tmpdir:
        # create a sample file
        for i in range(5):
            path = Path(tmpdir) / f"doc_{i}.txt"
            path.write_text(
                f"This is the content of document {i}. It contains sample content"
            )

        loader = DirectoryLoader(tmpdir, glob="*.txt", loader_cls=TextLoader)

        print("Initialized lazy loader for directory:", tmpdir)
        for doc in loader.lazy_load():
            print("Document content preview:", doc.page_content[:50], "...")
            print("Metadata:", doc.metadata["source"])


def doc_structure():
    doc = Document(
        page_content="This is a sample document.",
        metadata={
            "source": "maual_creation.txt",
            "author": "Vikas",
            "lenth": len("This is a sample document."),
            "tags": ["sample", "document"],
            "created_at": "2024-06-20T12:00:00Z",
        },
    )

    print("Document Structure:")
    print(f"  page_content (type): {type(doc.page_content)}")
    print(f"  page_content (value): {doc.page_content}")
    print(f"  metadata : {doc.metadata}")


def pdf_loader(pdf_path: str):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    print(f"Loaded {len(documents)} document(s) from PDF")
    for i, doc in enumerate(documents):
        print(f"Document {i+1} Content Length: {len(doc.page_content)} characters")
        print(f"Preview: {doc.page_content[:200]}...")
        print(f"Metadata: {doc.metadata}")
        print("-" * 50)


if __name__ == "__main__":
    # load_text_file("D:\\projects\\production_rag")
    # web_loader()
    # lazy_loader()
    # doc_structure()
    pdf_loader("Docs\Demo.pdf")
