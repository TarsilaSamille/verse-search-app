from fastapi import FastAPI
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
from dotenv import load_dotenv
import tensorflow as tf

app = FastAPI()

@app.get("/")
async def root():
    return {"greeting": "Hello, World!", "message": "Welcome to FastAPI!"}

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check endpoint
@app.get("/health")
def health_check():
    return {"status": "healthy", "port": os.environ.get("PORT")}

# Load Universal Sentence Encoder model
try:
    model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
    logger.info("Universal Sentence Encoder model loaded successfully.")
except Exception as e:
    logger.error(f"Error loading Universal Sentence Encoder model: {e}")
    raise HTTPException(status_code=500, detail="Failed to load the model")

# Dataset configuration
DATASET_BASE_URL = "https://datasets-server.huggingface.co/rows?dataset=tarsssss%2Ftranslation-bj-en&config=default&split=train"
BATCH_SIZE = 100  

def load_dataset() -> List[Dict]:
    """Load dataset with pagination to handle server limits"""
    dataset = []
    offset = 0
    
    try:
        while True:
            url = f"{DATASET_BASE_URL}&offset={offset}&length={BATCH_SIZE}"
            response = requests.get(url)
            response.raise_for_status()
            
            batch = response.json().get("rows", [])
            if not batch:
                break
                
            dataset.extend(batch)
            offset += BATCH_SIZE
            
            # Stop if we get fewer results than requested
            if len(batch) < BATCH_SIZE:
                break
                
        logger.info(f"Loaded {len(dataset)} entries from dataset")
        return dataset
        
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise HTTPException(status_code=500, detail="Failed to load dataset")

def create_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """Create and populate FAISS index with normalized embeddings"""
    embeddings = normalize(embeddings, norm='l2', axis=1)
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)  # Use Inner Product for cosine similarity
    index.add(embeddings)
    return index

# Initialize dataset and index
try:
    dataset = load_dataset()
    dataset_texts = [entry["row"]["text"] + " " + entry["row"]["bj_translation"] for entry in dataset]
    dataset_embeddings = model(dataset_texts).numpy()
    index = create_faiss_index(dataset_embeddings)
    logger.info("Dataset and FAISS index initialized successfully.")
except Exception as e:
    logger.error(f"Error initializing dataset or FAISS index: {e}")
    raise HTTPException(status_code=500, detail="Failed to initialize dataset or index")

class CombinedSearchRequest(BaseModel):
    query: str
    search_language: str  # "en" for English, "bj" for BJ

@app.post("/combined-search")
def combined_search(request: CombinedSearchRequest) -> Dict:
    try:
        query = request.query.strip().lower()
        search_language = request.search_language.lower()
        results = []

        # 1. Text-based search
        text_matches = []
        for entry in dataset:
            target_text = (
                entry["row"]["text"].lower() 
                if search_language == "en" 
                else entry["row"]["bj_translation"].lower()
            )
            
            if query in target_text:
                text_matches.append({
                    "english": entry["row"]["text"],
                    "bj_translation": entry["row"]["bj_translation"],
                    "similarity": 1.0,  # Exact match gets max score
                    "type": "text_match",
                    "id": entry["row"].get("id", hash(entry["row"]["text"]))  # Unique identifier
                })

        # 2. Semantic search
        query_embedding = model([query]).numpy()
        query_embedding = normalize(query_embedding, norm='l2', axis=1)
        
        # Search with FAISS
        similarities, indices = index.search(query_embedding, 10)
        
        semantic_matches = []
        for idx, score in zip(indices[0], similarities[0]):
            if idx < 0:  # FAISS returns -1 for invalid indices
                continue
                
            entry = dataset[idx]["row"]
            semantic_matches.append({
                "english": entry["text"],
                "bj_translation": entry["bj_translation"],
                "similarity": float(score),
                "type": "semantic_match",
                "id": entry.get("id", hash(entry["text"]))
            })

        # Combine and deduplicate results
        combined_results = text_matches + semantic_matches
        seen_ids = set()
        final_results = []
        
        for result in sorted(combined_results, key=lambda x: x["similarity"], reverse=True):
            if result["id"] not in seen_ids:
                seen_ids.add(result["id"])
                final_results.append(result)

        return {"results": final_results[:20]}  # Return top 20 results

    except Exception as e:
        logger.error(f"Search error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Search failed")
    

if __name__ == "__main__":
    import hypercorn.asyncio
    from hypercorn.config import Config

    config = Config()
    config.bind = ["0.0.0.0:" + os.environ.get("PORT", "8000")]
    hypercorn.asyncio.run_single(config, app)