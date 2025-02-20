import tensorflow_hub as hub
import tensorflow as tf
from datasets import Dataset, DatasetDict, load_dataset
import json

use_model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

# Load your dataset (Make sure the JSON structure matches your needs)
with open("jagoy-english.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Extract texts
texts = [entry["translation"]["en"] for entry in data]

# Generate embeddings
embeddings = use_model(texts)
embeddings = embeddings.numpy().tolist()  # Convert tensors to lists


# Create Hugging Face dataset
dataset = Dataset.from_dict({
    "text": texts,
    "bj_translation": [entry["translation"].get("bj", "No translation") for entry in data],
    "embedding": embeddings
})

# Save locally
dataset.save_to_disk("local_embeddings")

# Upload to Hugging Face
dataset.push_to_hub("tarsssss/translation-bj-en")
