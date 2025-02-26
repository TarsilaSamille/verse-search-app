import os
import logging
import numpy as np
import faiss
import torch
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.preprocessing import normalize
from typing import List, Dict
import uvicorn
from dotenv import load_dotenv
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from tqdm import tqdm 

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

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for model, dataset, and index
model = None
dataset = []  # Initialize as empty list
index = None

def load_model():
    global model
    try:
        model = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise HTTPException(status_code=500, detail="Failed to load model")

async def load_translation_dataset() -> List[Dict]:
    global dataset
    try:
        logger.info("Loading dataset from Hugging Face...")
        dataset = load_dataset('tarsssss/translation-bj-en', split='train')
        logger.info(f"Dataset loaded with {len(dataset)} entries.")
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise HTTPException(status_code=500, detail="Failed to load dataset")

def create_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatL2:
    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)
    return index

async def initialize_index():
    global index
    if dataset and model:
        # Extract embeddings from the dataset
        embeddings = np.array([model.encode(entry["text"]) for entry in tqdm(dataset, desc="Creating embeddings")], dtype=np.float32)
        embeddings = normalize(embeddings, norm='l2', axis=1)  # Normalize embeddings
        logger.info(f"Embedding dimension: {embeddings.shape[1]}")
        index = create_faiss_index(embeddings)
        logger.info("FAISS index created successfully.")

@app.on_event("startup")
async def startup_event():
    load_model()
    await load_translation_dataset()
    await initialize_index()

# Search request model
class SearchRequest(BaseModel):
    query: str
    search_language: str  # "en" or "bj"

@app.post("/search")
async def search(request: SearchRequest) -> Dict:
    try:
        if not dataset:
            raise HTTPException(status_code=500, detail="Dataset not loaded")

        query = request.query.strip().lower()
        search_lang = request.search_language.lower()
    
        text_matches = [
            {
                "english": entry["text"],
                "bj_translation": entry["bj_translation"],
                "similarity": 1.0,
                "type": "text_match",
                "id": entry.get("id", hash(entry["text"]))
            }
            for entry in dataset
            if query in (entry["text"].lower() if search_lang == "en" else entry["bj_translation"].lower())
        ]

        # Use the model to encode the query
        query_embedding = model.encode([query], convert_to_numpy=True)
        query_embedding = normalize(query_embedding, norm='l2', axis=1)  # Normalize query embedding
        similarities, indices = index.search(query_embedding, 10)
        
        semantic_matches = [
            {
                "english": dataset[int(idx)]["text"],
                "bj_translation": dataset[int(idx)]["bj_translation"],
                "similarity": float(score),
                "type": "semantic_match",
                "id": dataset[int(idx)].get("id", hash(dataset[int(idx)]["text"]))
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

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, access_log=False)