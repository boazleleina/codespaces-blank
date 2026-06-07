import os, glob
from dotenv import load_dotenv, find_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec

# 1. Load Keys securely
load_dotenv(find_dotenv())
HF_TOKEN = os.getenv("HF_TOKEN")

# 2. Initialize Pinecone Database
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))

# 3. Load the Embedding Model (passing the VIP token for gated access)
model = SentenceTransformer("google/embeddinggemma-300m", token=HF_TOKEN)

# 4. Read files and apply Recursive Chunking
splitter = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=20)
vectors_to_upsert = []
vector_id = 0

for path in glob.glob("data/*.txt"):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
        
    for chunk in splitter.split_text(text):
        # Translate text to math
        vector = model.encode(chunk).tolist()
        # Structure the payload: (ID, Vector Coordinates, Original Text)
        vectors_to_upsert.append((f"chunk_{vector_id}", vector, {"text": chunk}))
        vector_id += 1

# 5. Upload to the cloud
index.upsert(vectors=vectors_to_upsert)
print("Data successfully indexed!")