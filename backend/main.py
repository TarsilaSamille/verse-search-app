import json
import faiss
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Permitir requisições de qualquer origem (ajuste conforme necessário)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Pode restringir a origem específica do frontend, ex: ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],  # Permite todos os métodos (GET, POST, OPTIONS, etc.)
    allow_headers=["*"],  # Permite todos os headers
)

# Load JSON data
json_file = "backend/jagoy-english.json"

with open(json_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# Extract texts and translations
english_texts = [entry["translation"]["en"] for entry in data]
bj_translations = {entry["translation"]["en"]: entry["translation"]["bj"] for entry in data}

# Load sentence transformer model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Encode sentences into embeddings
embeddings = np.array(model.encode(english_texts, convert_to_numpy=True), dtype=np.float32)

# Create FAISS index
d = embeddings.shape[1]
index = faiss.IndexFlatL2(d)
index.add(embeddings)

# FastAPI app
class Query(BaseModel):
    text: str
    top_k: int = 5

@app.post("/search/")
def find_similar_verse(query: Query):
    query_embedding = np.array(model.encode([query.text], convert_to_numpy=True), dtype=np.float32)
    distances, indices = index.search(query_embedding, query.top_k)

    results = []
    for d, idx in zip(distances[0], indices[0]):
        en_text = english_texts[idx]
        bj_text = bj_translations.get(en_text, "No translation available")
        results.append({"english": en_text, "bj_translation": bj_text, "similarity": round(1 - d, 4)})

    return {"query": query.text, "results": results}

# Run server: uvicorn backend.main:app --reload
