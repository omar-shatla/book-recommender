# Semantic Book Recommender

A book recommendation app that finds books by *meaning*, not keywords, then lets you filter results by category and emotional tone. Describe the kind of book you're in the mood for, e.g. *"a story about forgiveness"*, and get back a visual gallery of matches, ranked by semantic similarity to your description.

Built as an end-to-end NLP pipeline: from raw metadata, through embedding generation, category classification, and emotion scoring, to an interactive Gradio dashboard.

![Dashboard demo](docs/demo.gif)

## How it works

1. **Data**: ~7,000 books with metadata (title, author, description, category, ratings) from the [7k Books with Metadata dataset on Kaggle](https://www.kaggle.com/datasets/dylanjcastillo/7k-books-with-metadata).
2. **Embedding & semantic search**: Book descriptions are embedded locally using [`BAAI/bge-small-en-v1.5`](https://huggingface.co/BAAI/bge-small-en-v1.5) via `sentence-transformers`, with no external API calls, no rate limits, and everything running entirely on-device. Embeddings are stored and queried using [ChromaDB](https://www.trychroma.com/) through LangChain.
3. **Category classification**: Book categories are simplified/predicted where missing, to support clean filtering (see `text-classification.ipynb`).
4. **Emotion scoring**: Each book's description is scored across emotional dimensions (joy, sadness, anger, fear, surprise) to enable tone-based sorting of results (see `sentiment-analysis.ipynb`).
5. **Dashboard**: A [Gradio](https://www.gradio.app/) interface (`gradio-dashboard.py`) ties it together: enter a natural-language query, optionally filter by category and tone, and get a visual gallery of the top matching books with cover art and descriptions.

## Project structure

```
book-recommender/
├── gradio-dashboard.py          # Main app — run this
├── data-exploration.ipynb       # Initial dataset cleaning & exploration
├── text-classification.ipynb    # Category prediction/simplification
├── sentiment-analysis.ipynb     # Emotion scoring pipeline
├── vector-search.ipynb          # Embedding & vector store experimentation
├── books_cleaned.csv            # Cleaned book metadata
├── books_with_categories.csv    # + predicted/simplified categories
├── books_with_emotions.csv      # + emotion scores (used by the dashboard)
├── emotions.csv                 # Raw emotion scoring output
├── tagged_descriptions.txt      # Descriptions tagged with ISBN for embedding
├── requirements.txt
└── .env.example                 # Template for optional API keys
```

## Setup

**1. Clone and install dependencies**

```bash
git clone https://github.com/omar-shatla/book-recommender.git
cd book-recommender
python -m venv .venv
.venv\Scripts\activate       # Windows
source .venv/bin/activate    # macOS/Linux
pip install -r requirements.txt
```

**2. (Optional) Set up environment variables**

The dashboard runs fully locally and doesn't require any API keys. `.env.example` is included as a template in case you want to extend the project using Google's Gemini API or a gated HuggingFace model, copy it to `.env` and fill in your own keys if so.

```bash
cp .env.example .env
```

**3. Build the vector store**

The dashboard expects a populated ChromaDB vector store at `./chroma_db`. Run through `vector-search.ipynb` to generate it from `tagged_descriptions.txt`.

**4. Run the dashboard**

```bash
python gradio-dashboard.py
```

Open the local URL Gradio prints in your terminal (typically `http://127.0.0.1:7860`).

## Tech stack

`Python` · `LangChain` · `ChromaDB` · `sentence-transformers` · `Gradio` · `pandas` · `scikit-learn`

## Notes

- The embedding model runs entirely locally: no API costs, no external calls, no rate limits, and it works offline once dependencies are installed.
- `vector-search.ipynb` includes earlier experimentation with Google's Generative AI embeddings before settling on the local `sentence-transformers` approach for cost and portability reasons.
- The notebooks are kept in the repo intentionally, to show the full pipeline from raw data to deployed app rather than just the final result.

## License

MIT
