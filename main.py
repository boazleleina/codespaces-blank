import os
from fastapi import FastAPI
from dotenv import load_dotenv, find_dotenv
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone

load_dotenv(find_dotenv())

app = FastAPI()
model = None
index = None

@app.on_event("startup")
def startup():
    """Warms up the model in global memory before serving traffic."""
    global model, index
    model = SentenceTransformer("google/embeddinggemma-300m", token=os.getenv("HF_TOKEN"))
    index = Pinecone(api_key=os.getenv("PINECONE_API_KEY")).Index(os.getenv("PINECONE_INDEX_NAME"))

@app.get("/search")
def search(query: str):
    # Step A: Convert search query to a vector
    query_vector = model.encode(query).tolist()
    
    # Step B: Perform Cosine Similarity geometric search in Pinecone
    search_results = index.query(
        vector=query_vector,
        top_k=2,
        include_metadata=True
    )
    
    # Step C: Format the response
    return [
        {
            "score": match["score"], 
            "text": match["metadata"]["text"]
        } 
        for match in search_results["matches"]
    ]