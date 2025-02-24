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

# Carrega variáveis de ambiente do arquivo .env
load_dotenv()

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Inicializa a aplicação FastAPI
app = FastAPI()

# Adiciona middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Endpoint de verificação de saúde
@app.get("/health")
def health_check():
    return {"status": "healthy", "port": os.environ.get("PORT", "Not Set")}

# Configuração do modelo e cache
os.environ["TFHUB_CACHE_DIR"] = "/tmp/tfhub_cache"

try:
    model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
    logger.info("Modelo Universal Sentence Encoder carregado com sucesso.")
except Exception as e:
    logger.error(f"Erro ao carregar o modelo: {e}")
    raise HTTPException(status_code=500, detail="Falha ao carregar o modelo")

# Configuração do dataset
DATASET_URL = "https://datasets-server.huggingface.co/rows?dataset=tarsssss%2Ftranslation-bj-en&config=default&split=train"
BATCH_SIZE = 100

def load_dataset() -> List[Dict]:
    dataset, offset = [], 0
    try:
        while True:
            response = requests.get(f"{DATASET_URL}&offset={offset}&length={BATCH_SIZE}")
            response.raise_for_status()
            batch = response.json().get("rows", [])
            if not batch:
                break
            dataset.extend(batch)
            offset += BATCH_SIZE
        logger.info(f"Dataset carregado com {len(dataset)} entradas.")
        return dataset
    except Exception as e:
        logger.error(f"Erro ao carregar dataset: {e}")
        raise HTTPException(status_code=500, detail="Falha ao carregar dataset")

# Inicializa dataset e indexação
logger.info("Carregando dataset...")
dataset = load_dataset()
dataset_texts = [entry["row"]["text"] + " " + entry["row"]["bj_translation"] for entry in dataset]
dataset_embeddings = model(dataset_texts).numpy()
dataset_embeddings = normalize(dataset_embeddings, norm='l2', axis=1)

def create_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(embeddings)
    return index

index = create_faiss_index(dataset_embeddings)

class SearchRequest(BaseModel):
    query: str
    search_language: str  # "en" ou "bj"

@app.post("/search")
def search(request: SearchRequest) -> Dict:
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

        query_embedding = model([query]).numpy()
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
        logger.error(f"Erro na busca: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Erro na busca")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, access_log=False)