from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import uvicorn
import requests
import faiss
from sklearn.preprocessing import normalize
import logging
import tensorflow_hub as hub

app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Allow frontend to access backend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Universal Sentence Encoder model
model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

# Load dataset
DATASET_URL = "https://datasets-server.huggingface.co/rows?dataset=tarsssss%2Ftranslation-bj-en&config=default&split=train&offset=0&length=100"

def load_dataset():
    try:
        response = requests.get(DATASET_URL)
        response.raise_for_status()
        dataset = response.json()["rows"]
        return dataset
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise HTTPException(status_code=500, detail="Failed to load dataset")

# Load dataset and preprocess embeddings
dataset = load_dataset()
dataset_texts = [entry["row"]["text"] for entry in dataset]
dataset_bj_texts = [entry["row"]["bj_translation"] for entry in dataset]
dataset_embeddings = np.array([entry["row"]["embedding"] for entry in dataset], dtype=np.float32)
dataset_embeddings = normalize(dataset_embeddings, norm='l2', axis=1)

# Initialize FAISS index
d = dataset_embeddings.shape[1]
index = faiss.IndexFlatL2(d)
index.add(dataset_embeddings)

class CombinedSearchRequest(BaseModel):
    query: str
    search_language: str  # "en" for English, "bj" for BJ

@app.post("/combined-search")
def combined_search(request: CombinedSearchRequest):
    try:
        query = request.query.lower()  # Convert query to lowercase for case-insensitive search
        search_language = request.search_language  # Get the search language
        results = []

        # 1. Text Search: Find phrases containing the query word
        text_results = []
        for entry in dataset:
            text = entry["row"]["text"].lower() if search_language == "en" else entry["row"]["bj_translation"].lower()
            if query in text:
                text_results.append({
                    "english": entry["row"]["text"],
                    "bj_translation": entry["row"]["bj_translation"],
                    "type": "text_match"  # Indicate that this is a text match
                })

        # 2. Semantic Search: Find semantically similar phrases
        query_embedding = model([query]).numpy()  # Generate query embedding
        query_embedding = normalize(query_embedding, norm='l2', axis=1)
        distances, indices = index.search(query_embedding, 10)  # Top 10 results

        semantic_results = []
        for i in range(len(indices[0])):
            entry = dataset[indices[0][i]]["row"]
            cosine_similarity = 1 - distances[0][i] / 2  # Convert L2 distance to cosine similarity
            semantic_results.append({
                "english": entry["text"],
                "bj_translation": entry["bj_translation"],
                "similarity": cosine_similarity,
                "type": "semantic_match"  # Indicate that this is a semantic match
            })

        # Combine results, prioritizing text matches
        results = text_results + semantic_results

        # Remove duplicates (if a phrase appears in both text and semantic results)
        unique_results = []
        seen_texts = set()
        for result in results:
            if result["english"] not in seen_texts:
                seen_texts.add(result["english"])
                unique_results.append(result)

        return {"results": unique_results}
    except Exception as e:
        logger.error(f"Error during combined search: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)