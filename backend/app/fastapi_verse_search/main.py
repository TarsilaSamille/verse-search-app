import os
import logging
import numpy as np
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict
import uvicorn
from dotenv import load_dotenv
from datasets import load_dataset
from annoy import AnnoyIndex

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

model = None
dataset = []  # Initialize as empty list
index = None
annoy_index = None
dim = 384  # Dimension of embeddings for MiniLM model

def load_model():
    global model
    try:
        model = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise HTTPException(status_code=500, detail="Failed to load model")

async def load_translation_dataset() -> List[Dict]:
    global dataset, annoy_index
    try:
        logger.info("Loading dataset from Hugging Face...")
        dataset = load_dataset('tarsssss/translation-bj-en', split='train')
        
        # Initialize Annoy index
        annoy_index = AnnoyIndex(dim, 'angular')
        for i, entry in enumerate(dataset):
            embedding = np.array(entry['embedding'])
            annoy_index.add_item(i, embedding)
        
        annoy_index.build(10)  # 10 trees for fast lookup
        logger.info(f"Dataset loaded with {len(dataset)} entries.")
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise HTTPException(status_code=500, detail="Failed to load dataset")

@app.on_event("startup")
async def startup_event():
    load_model()
    await load_translation_dataset()

# Search request model
class SearchRequest(BaseModel):
    query: str
    search_language: str  # "en" or "bj"

def search_q(query, k=5):
    query_embedding = model.encode(query, convert_to_numpy=True)
    indices = annoy_index.get_nns_by_vector(query_embedding, k, include_distances=True)
    results = [
        {"idx": idx, "score": 1 - dist, **dataset[idx]}
        for idx, dist in zip(*indices)
        if idx < len(dataset)
    ]
    return results

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

        samples = search_q(query, 10)
        semantic_matches = [
            {
                "english": s["text"],
                "bj_translation": s["bj_translation"],
                "similarity": float(s["score"]),
                "type": "semantic_match",
                "id": s["idx"]
            }
            for s in samples if s["idx"] >= 0
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
