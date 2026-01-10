from dotenv import load_dotenv
import os
import time

from src.helper import (
    load_pdf_documents,
    filter_to_minimal_docs,
    text_split,
    download_hugging_face_embeddings,
)

from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY is not set in .env")

print("STEP 1: Connecting to Pinecone...", flush=True)
pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "medical-chatbot"

# Create index if it does not exist
if not pc.has_index(index_name):
    print("STEP 2: Creating index...", flush=True)
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    print("STEP 3: Index created.", flush=True)
else:
    print("STEP 2: Index already exists.", flush=True)

# Load PDF documents
print("STEP 4: Loading PDF documents...", flush=True)
docs = load_pdf_documents("data/")
print(f"STEP 5: Loaded {len(docs)} pages", flush=True)

# Filter and split documents
docs = filter_to_minimal_docs(docs)
chunks = text_split(docs)
print(f"STEP 6: Split into {len(chunks)} chunks", flush=True)

# Load embeddings
print("STEP 7: Loading embedding model...", flush=True)
start = time.time()
embeddings = download_hugging_face_embeddings()
print(f"STEP 8: Embeddings ready in {time.time() - start:.2f} seconds", flush=True)

# Upload to Pinecone
print("STEP 9: Uploading chunks to Pinecone...", flush=True)
PineconeVectorStore.from_documents(
    documents=chunks,
    index_name=index_name,
    embedding=embeddings,
    pinecone_api_key=PINECONE_API_KEY,
)

print("STEP 10: Upload complete!", flush=True)
