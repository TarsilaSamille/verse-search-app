import os
import logging
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.preprocessing import normalize
from typing import List, Dict
import uvicorn
from dotenv import load_dotenv
from datasets import load_dataset

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
dataset = None
index = None

def load_model():
    global model
    try:
        model = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise HTTPException(status_code=500, detail="Failed to load model")

async def load_translation_dataset():
    global dataset, index
    try:
        logger.info("Loading dataset...")
        dataset = load_dataset('tarsssss/translation-bj-en', split='train')
        
        # Load and normalize embeddings
        embeddings = np.array(dataset['embedding'], dtype=np.float32)
        embeddings = normalize(embeddings, axis=1, norm='l2')
        
        # Configure FAISS parameters
        d = embeddings.shape[1]
        nlist = 100  # Number of clusters
        nprobe = 10  # Number of clusters to search
        
        # Check for existing index
        if os.path.exists("saved_index.faiss"):
            logger.info("Loading saved FAISS index...")
            index = faiss.read_index("saved_index.faiss")
            index.nprobe = nprobe
        else:
            logger.info("Creating new FAISS index...")
            quantizer = faiss.IndexFlatIP(d)
            index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
            
            # Train and add embeddings
            faiss.omp_set_num_threads(1)  # Limit to single thread for stability
            index.train(embeddings)
            index.add(embeddings)
            index.nprobe = nprobe
            faiss.write_index(index, "saved_index.faiss")
        
        logger.info(f"Index ready with {index.ntotal} entries")

    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise HTTPException(status_code=500, detail="Failed to load dataset")

@app.on_event("startup")
async def startup_event():
    load_model()
    await load_translation_dataset()

class SearchRequest(BaseModel):
    query: str
    search_language: str

async def search_q(query: str, k: int = 5):
    try:
        # Encode query asynchronously
        query_embedding = await run_in_threadpool(
            model.encode, query, convert_to_numpy=True
        )
        query_embedding = normalize(np.array([query_embedding]), axis=1, norm='l2')
        
        # Search FAISS index
        scores, indices = index.search(query_embedding.astype(np.float32), k)
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx >= 0:
                entry = dataset[int(idx)]
                results.append({
                    "score": float(score),
                    "english": entry["text"],
                    "bj_translation": entry["bj_translation"],
                    "idx": int(idx)
                })
        return results
    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        return []

@app.post("/search")
async def search(request: SearchRequest) -> Dict:
    try:
        query = request.query.strip().lower()
        if len(query) < 2:
            return {"results": []}

        # Perform semantic search
        semantic_matches = await search_q(query, 10)
        
        # Deduplicate results
        seen_ids = set()
        final_results = []
        for result in semantic_matches:
            if result["idx"] not in seen_ids:
                seen_ids.add(result["idx"])
                final_results.append({
                    "english": result["english"],
                    "bj_translation": result["bj_translation"],
                    "similarity": result["score"],
                    "type": "semantic_match",
                    "id": result["idx"]
                })

        return {"results": final_results[:10]}
    except Exception as e:
        logger.error(f"Search error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Search error")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        workers=1,  # Reduce workers for low-CPU environments
        log_level="info",
        access_log=False
    )