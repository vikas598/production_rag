from dotenv import load_dotenv
from langchain_community.embeddings import JinaEmbeddings

load_dotenv()
embeddings = JinaEmbeddings(model_name="jina-embeddings-v2-base-en")

# embeddings = HuggingFaceBgeEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# # single text
# text = "This is a sample text for embedding."
# embedding = embeddings.embed_query(text)
# print(f"Embedding for single text: {embedding}")

# print(len(embedding))

# multiple texts
embeds = embeddings.embed_documents(
    ["This is the first document.", "This is the second document."]
)
print(f"Embeddings for multiple texts: {embeds}")
print(f"Number of embeddings returned: {len(embeds)}")  # Should print 2
print(f"Length of each embedding: {len(embeds[0])}")
