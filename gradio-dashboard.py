import pandas as pd
import numpy as np
from dotenv import load_dotenv
from typing import List

from langchain_classic import text_splitter
from langchain_text_splitters import CharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_community.document_loaders import TextLoader
import gradio as gr

load_dotenv()

# Custom embedding class (Completely free, no API limits)
class LocalSentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name="BAAI/bge-small-en-v1.5"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, show_progress_bar=False).tolist()

    def embed_query(self, text: str) -> List[float]:
        instructed = f"Represent this sentence for searching relevant passages: {text}"
        return self.model.encode(instructed).tolist()

# Initialize embeddings
embeddings = LocalSentenceTransformerEmbeddings("BAAI/bge-small-en-v1.5")

# Load existing vector store from disk
db_books = Chroma(
    collection_name="books",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)

books = pd.read_csv('books_with_emotions.csv')
books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"
books["large_thumbnail"] = np.where(
    books["large_thumbnail"].isna(),
    "cover-not-found.jpg",
    books["large_thumbnail"],
)


def retrieve_semantic_recommendations(
        query: str,
        category: str,
        tone: str = None,
        initial_top_k: int = 50,
        final_top_k: int = 16,
) -> pd.DataFrame:

    recs = db_books.similarity_search(query, k=initial_top_k)
    books_list = []
    for rec in recs:
        if "isbn13" in rec.metadata:
            books_list.append(rec.metadata["isbn13"])
    books_list_str = [str(isbn) for isbn in books_list]
    book_recs = books[books["isbn13"].astype(str).isin(books_list_str)].head(final_top_k)

    if category != "All":
        book_recs = book_recs[book_recs["simple_categories"] == category][:final_top_k]
    else:
        book_recs = book_recs[:final_top_k]

    if tone == "Happy":
        book_recs.sort_values(by="joy", ascending=False, inplace=True)
    elif tone == "Surprising":
        book_recs.sort_values(by="surprise", ascending=False, inplace=True)
    elif tone == "Angry":
        book_recs.sort_values(by="anger", ascending=False, inplace=True)
    elif tone == "Sad":
        book_recs.sort_values(by="sadness", ascending=False, inplace=True)
    elif tone == "Suspenseful":
        book_recs.sort_values(by="fear", ascending=False, inplace=True)

    return book_recs


def recommend_books(
        query: str,
        category: str,
        tone: str
):
    recommendations = retrieve_semantic_recommendations(query, category, tone)
    results = []

    for _, row in recommendations.iterrows():
        description = row["description"]
        truncated_des_split = description.split()
        truncated_description = " ".join(truncated_des_split[:30]) + "..."

        authors = row["authors"]
        if pd.isna(authors):
            authors_str = "Unknown Author"
        else:
            authors_split = authors.split(";")
            if len(authors_split) == 2:
                authors_str = f"{authors_split[0]} & {authors_split[1]}"
            elif len(authors_split) > 2:
                authors_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
            else:
                authors_str = authors

        caption = f"{row['title']} by {authors_str}: {truncated_description}"
        results.append((row['large_thumbnail'], caption))
    return results


categories = ["All"] + sorted(books["simple_categories"].unique())
tones = ["All"] + ["Happy", "Surprising", "Angry", "Sad", "Suspenseful"]

with gr.Blocks(theme = gr.themes.Glass()) as dashboard:
    gr.Markdown("# Semantic Book Recommender")

    with gr.Row():
        user_query = gr.Textbox(label="Please enter a description of a book:",
                                placeholder="e.g., A story about forgiveness")
        category_dropdown = gr.Dropdown(
            choices=categories,
            label="Select a category:",
            value="All"
        )

        tone_dropdown = gr.Dropdown(
            choices=tones,
            label="Select an emotional tone:",
            value="All"
        )

        submit_button = gr.Button("Find recommendations")

    gr.Markdown("## Recommendations")
    output = gr.Gallery(label = "Recommended Books", columns = 8, rows = 2)

    submit_button.click(
        fn = recommend_books,
        inputs = [user_query, category_dropdown, tone_dropdown],
        outputs = output
    )

if __name__ == "__main__":
    dashboard.launch()