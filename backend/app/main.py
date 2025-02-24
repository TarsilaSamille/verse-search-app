import os
import logging
import requests
import numpy as np
import faiss
import tensorflow_hub as hub
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.preprocessing import normalize
from typing import List, Dict
import uvicorn
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set TensorFlow Hub cache directory
os.environ["TFHUB_CACHE_DIR"] = "/tmp/tfhub_cache"

# Load the TensorFlow model
def load_model():
    try:
        model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
        logger.info("Model loaded successfully.")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise HTTPException(status_code=500, detail="Failed to load model")

# Fetch dataset in batches
def fetch_batch(offset=0, length=10):
    DATASET_URL = "https://datasets-server.huggingface.co/rows?dataset=tarsssss%2Ftranslation-bj-en&config=default&split=train"
    try:
        response = requests.get(f"{DATASET_URL}&offset={offset}&length={length}")
        response.raise_for_status()
        return response.json().get("rows", [])
    except Exception as e:
        logger.error(f"Error fetching batch: {e}")
        return []

# Load model on startup
model = load_model()

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "port": os.environ.get("PORT", "Not Set")}

# Search request model
class SearchRequest(BaseModel):
    query: str
    search_language: str  # "en" or "bj"

# Search endpoint
@app.post("/search")
async def search(request: SearchRequest) -> Dict:
    try:
        query = request.query.strip().lower()
        search_lang = request.search_language.lower()

        query_embedding = model([query]).numpy()
        query_embedding = normalize(query_embedding, norm="l2", axis=1)

        results, offset, batch_size = [], 0, 100  # Processa 100 por vez

        while True:
            batch = fetch_batch(offset, batch_size)
            if not batch:
                break  # Se não houver mais dados, para

            batch_texts = [entry["row"]["text"] + " " + entry["row"]["bj_translation"] for entry in batch]
            batch_embeddings = model(batch_texts).numpy()
            batch_embeddings = normalize(batch_embeddings, norm="l2", axis=1)

            similarities = np.dot(batch_embeddings, query_embedding.T).flatten()

            for idx, score in enumerate(similarities):
                if score > 0.5:  # Apenas resultados relevantes
                    results.append({
                        "english": batch[idx]["row"]["text"],
                        "bj_translation": batch[idx]["row"]["bj_translation"],
                        "similarity": float(score),
                        "type": "semantic_match",
                        "id": batch[idx]["row"].get("id", hash(batch[idx]["row"]["text"]))
                    })

            offset += batch_size  # Avança para o próximo lote

        results = sorted(results, key=lambda x: x["similarity"], reverse=True)[:20]

        return {"results": results}
    except Exception as e:
        logger.error(f"Search error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Search error")

# Run the application
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, access_log=False)
