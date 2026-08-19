# Check if chunks have proper metadata
import os
from langchain_chroma import Chroma
from sentence_transformers import SentenceTransformer
from langchain_core.embeddings import Embeddings
from typing import List

class LocalSentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name="BAAI/bge-small-en-v1.5"):
        self.model = SentenceTransformer(model_name)
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, show_progress_bar=False).tolist()
    def embed_query(self, text: str) -> List[float]:
        instructed = f"Represent this sentence for searching relevant passages: {text}"
        return self.model.encode(instructed).tolist()

embeddings = LocalSentenceTransformerEmbeddings("BAAI/bge-small-en-v1.5")

# Load from disk
db_books = Chroma(
    collection_name="books",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)

# Test a search
recs = db_books.similarity_search("A book to teach children about nature", k=5)
print(f"Got {len(recs)} results")
for i, rec in enumerate(recs):
    print(f"\nResult {i}:")
    print(f"  Has metadata: {'isbn13' in rec.metadata}")
    if 'isbn13' in rec.metadata:
        print(f"  Metadata ISBN13: {rec.metadata['isbn13']}")
    print(f"  Content starts with: {rec.page_content[:100]}")
