import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import uvicorn
import requests
import logging
from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict

app = FastAPI()

# Health check endpoint
@app.get("/health")
def health_check():
    return {"status": "healthy", "port": os.environ.get("PORT")}

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://verse-search-app.vercel.app", "https://verse-search-app-3.vercel.app"],
    allow_credentials=True,
    allow_methods=["OPTIONS", "POST", "GET"],
    allow_headers=["*"],
)

# Middleware to ensure CORS headers are present
@app.middleware("http")
async def add_cors_headers(request, call_next):
    response = await call_next(request)
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Credentials"] = "true"
    return response

# Load Hugging Face Transformer Model
try:
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModel.from_pretrained("distilbert-base-uncased")
    logger.info("DistilBERT model loaded successfully.")
except Exception as e:
    logger.error(f"Error loading Hugging Face model: {e}")
    raise HTTPException(status_code=500, detail="Failed to load the model")

# Dataset configuration
DATASET_BASE_URL = "https://datasets-server.huggingface.co/rows?dataset=tarsssss%2Ftranslation-bj-en&config=default&split=train"
BATCH_SIZE = 100  # Max allowed by server

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
            
            if len(batch) < BATCH_SIZE:
                break
                
        logger.info(f"Loaded {len(dataset)} entries from dataset")
        print(f"Dataset length: {len(dataset)}")
        return dataset
        
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise HTTPException(status_code=500, detail="Failed to load dataset")

def generate_embeddings(texts: List[str]) -> np.ndarray:
    """Generate embeddings using the Hugging Face transformer model"""
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1).numpy()
    return embeddings

# Initialize dataset
dataset = load_dataset()

# Generate embeddings for dataset
dataset_texts = [entry["row"]["text"] + " " + entry["row"]["bj_translation"] for entry in dataset]
dataset_embeddings = generate_embeddings(dataset_texts)

# Normalize and create a nearest neighbor search index
dataset_embeddings = normalize(dataset_embeddings, norm='l2', axis=1)
nn_model = NearestNeighbors(n_neighbors=10, algorithm='auto')
nn_model.fit(dataset_embeddings)

class CombinedSearchRequest(BaseModel):
    query: str
    search_language: str  # "en" for English, "bj" for BJ

@app.post("/combined-search")
def combined_search(request: CombinedSearchRequest) -> Dict:
    try:
        query = request.query.strip().lower()
        search_language = request.search_language.lower()

        # 1. Text-based search
        text_matches = []
        for entry in dataset:
            target_text = (
                entry["row"]["text"].lower() if search_language == "en" 
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
        query_embedding = generate_embeddings([query])
        query_embedding = normalize(query_embedding, norm='l2', axis=1)
        
        # Use NearestNeighbors for searching
        distances, indices = nn_model.kneighbors(query_embedding, n_neighbors=10)
        
        semantic_matches = []
        for idx, distance in zip(indices[0], distances[0]):
            entry = dataset[idx]["row"]
            semantic_matches.append({
                "english": entry["text"],
                "bj_translation": entry["bj_translation"],
                "similarity": float(1.0 / (1.0 + distance)),  # Convert distance to similarity
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
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "combined-search:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        access_log=False
    )
