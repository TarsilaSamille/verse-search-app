from sentence_transformers import SentenceTransformer
from datasets import Dataset, DatasetDict, load_dataset
import json
import faiss
import numpy as np

# Load the SentenceTransformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load your dataset (Make sure the JSON structure matches your needs)
with open("jagoy-english.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Extract texts
texts = [entry["translation"]["en"] for entry in data]

# Generate embeddings
embeddings = model.encode(texts, convert_to_numpy=True).tolist()  # Convert tensors to lists

# Create Hugging Face dataset
dataset = Dataset.from_dict({
    "text": texts,
    "bj_translation": [entry["translation"].get("bj", "No translation") for entry in data],
    "embedding": embeddings
})

# Create FAISS index
d = len(embeddings[0])  # Dimensionality of embeddings
index = faiss.IndexFlatL2(d)
index.add(np.array(embeddings))

# Add FAISS index to dataset
dataset.add_faiss_index("embedding", custom_index=index)

# Remove FAISS index before saving
dataset.drop_index("embedding")

# Save locally
dataset.save_to_disk("local_embeddings")

# Reload dataset and add FAISS index again
dataset = Dataset.load_from_disk("local_embeddings")
embeddings_array = np.array(dataset['embedding'])  # Convert to numpy array
index = faiss.IndexFlatL2(embeddings_array.shape[1])  # Correct dimensionality
index.add(embeddings_array)
dataset.add_faiss_index("embedding", custom_index=index)

# Remove FAISS index before uploading
dataset.drop_index("embedding")

# Upload to Hugging Face
dataset.push_to_hub("tarsssss/translation-bj-en")

# Reload dataset and add FAISS index again after upload
dataset = Dataset.load_from_disk("local_embeddings")
embeddings_array = np.array(dataset['embedding'])  # Convert to numpy array
index = faiss.IndexFlatL2(embeddings_array.shape[1])  # Correct dimensionality
index.add(embeddings_array)
dataset.add_faiss_index("embedding", custom_index=index)

# Function to search in the dataset
def search(query, k=5):
    query_embedding = model.encode([query], convert_to_numpy=True)
    
    # Search in the FAISS index
    scores, indices = index.search(np.array(query_embedding), k=k)
    indices = indices[0].tolist()  # Convert numpy array to a regular list of ints
    
    # Ensure indices are within bounds
    valid_indices = [i for i in indices if i < len(dataset)]
    
    # Fetch the corresponding samples from the dataset
    samples = [dataset[i] for i in valid_indices]  # Accessing samples using indices
    return samples

# Example search
query = "example search text"
results = search(query)
query_embedding = model.encode([query], convert_to_numpy=True)
scores, retrieved_examples = dataset.get_nearest_examples('embedding', query_embedding, k=10)


print(retrieved_examples)