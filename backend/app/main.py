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
import tensorflow as tf

# Load environment variables from .env file
load_dotenv()
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


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

# Global variables for model, dataset, and index
model = None
dataset = None
index = None

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "port": os.environ.get("PORT", "Not Set")}

# Load the TensorFlow model
async def load_model():
    global model
    try:
        model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
        logger.info("Model loaded successfully.")
        logger.info(f"Available signatures: {list(model.signatures.keys())}")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise HTTPException(status_code=500, detail="Failed to load model")

# Load the dataset in smaller batches
async def load_dataset() -> List[Dict]:
    global dataset
    DATASET_URL = "https://datasets-server.huggingface.co/rows?dataset=tarsssss%2Ftranslation-bj-en&config=default&split=train"
    BATCH_SIZE = 100
    MAX_DATASET_SIZE = 10000  # Limit dataset size
    dataset, offset = [], 0
    try:
        while True:
            response = requests.get(f"{DATASET_URL}&offset={offset}&length={BATCH_SIZE}")
            response.raise_for_status()
            batch = response.json().get("rows", [])
            if not batch or len(dataset) >= MAX_DATASET_SIZE:
                break
            dataset.extend(batch)
            offset += BATCH_SIZE
        logger.info(f"Dataset loaded with {len(dataset)} entries.")
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise HTTPException(status_code=500, detail="Failed to load dataset")

# Create FAISS index with memory-efficient settings
def create_faiss_index(embeddings: np.ndarray) -> faiss.IndexIVFFlat:
    d = embeddings.shape[1]
    quantizer = faiss.IndexFlatL2(d)
    index = faiss.IndexIVFFlat(quantizer, d, 100)  # 100 clusters
    index.train(embeddings)
    index.add(embeddings)
    return index

# Initialize model, dataset, and index on startup
@app.on_event("startup")
async def startup_event():
    await load_model()
    await load_dataset()
    global index
    if dataset and model:
        dataset_texts = [entry["row"]["text"] + " " + entry["row"]["bj_translation"] for entry in dataset]
        
        # Process embeddings in smaller batches to avoid memory issues
        batch_size = 100
        dataset_embeddings = []
        for i in range(0, len(dataset_texts), batch_size):
            batch_texts = dataset_texts[i:i + batch_size]
            # Use the correct signature for the model
            embed_fn = model.signatures["serving_default"]
            embeddings = embed_fn(tf.constant(batch_texts))["outputs"].numpy()
            embeddings = normalize(embeddings, norm='l2', axis=1)
            dataset_embeddings.append(embeddings)
        
        dataset_embeddings = np.vstack(dataset_embeddings)
        index = create_faiss_index(dataset_embeddings)
        logger.info("FAISS index created successfully.")

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
        
        text_matches = [
            {
                "english": entry["row"]["text"],
                "bj_translation": entry["row"]["bj_translation"],
                "similarity": 1.0,
                "type": "text_match",
                "id": entry["row"].get("id", hash(entry["row"]["text"]))
            }
            for entry in dataset
            if query in (entry["row"]["text"].lower() if search_lang == "en" else entry["row"]["bj_translation"].lower())
        ]

        # Use the correct signature for the model
        query_embedding = model.signatures["default"](inputs=tf.constant([query]))["outputs"].numpy()
        query_embedding = normalize(query_embedding, norm='l2', axis=1)
        similarities, indices = index.search(query_embedding, 10)
        
        semantic_matches = [
            {
                "english": dataset[idx]["row"]["text"],
                "bj_translation": dataset[idx]["row"]["bj_translation"],
                "similarity": float(score),
                "type": "semantic_match",
                "id": dataset[idx]["row"].get("id", hash(dataset[idx]["row"]["text"]))
            }
            for idx, score in zip(indices[0], similarities[0]) if idx >= 0
        ]

        final_results = sorted(text_matches + semantic_matches, key=lambda x: x["similarity"], reverse=True)
        seen_ids, unique_results = set(), []
        for result in final_results:
            if result["id"] not in seen_ids:
                seen_ids.add(result["id"])
                unique_results.append(result)

        return {"results": unique_results[:20]}
    except Exception as e:
        logger.error(f"Search error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Search error")

# Run the application
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, access_log=False)

