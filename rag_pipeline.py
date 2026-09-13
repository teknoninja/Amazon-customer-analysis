import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
import os
import logging
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROCESSED_DATA_PATH = "data/amazon_help_interactions.csv"
CHROMA_DB_PATH = "data/chroma_db"
COLLECTION_NAME = "amazon_help_rag"

def get_embedding_function():
    # Uses local all-MiniLM-L6-v2 via ONNX (Sentence-Transformers)
    # 100% free, runs locally in milliseconds with zero API rate limits
    return embedding_functions.DefaultEmbeddingFunction()

def build_rag_db(sample_size=3000):
    if not os.path.exists(PROCESSED_DATA_PATH):
        logger.error(f"{PROCESSED_DATA_PATH} not found. Run data_processing.py first.")
        return

    logger.info("Loading processed data...")
    df = pd.read_csv(PROCESSED_DATA_PATH)
    
    # Sample for the RAG knowledge base
    if len(df) > sample_size:
        df_rag = df.sample(n=sample_size, random_state=42).copy()
    else:
        df_rag = df.copy()
        
    df_rag = df_rag.dropna(subset=["customer_text", "brand_text"])
    
    logger.info(f"Building RAG database with {len(df_rag)} interactions using Local Sentence-Transformers...")
    
    ef = get_embedding_function()
    
    # Initialize ChromaDB
    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    
    # Get or create collection
    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"}
    )
    
    documents = df_rag["customer_text"].tolist()
    metadatas = [{"brand_text": row["brand_text"], "intent": str(row["heuristic_intent"])} for _, row in df_rag.iterrows()]
    ids = [str(row["customer_tweet_id"]) for _, row in df_rag.iterrows()]
    
    # Insert in batches
    batch_size = 500
    for i in range(0, len(documents), batch_size):
        logger.info(f"Indexing batch {i} to {min(i + batch_size, len(documents))}...")
        collection.upsert(
            documents=documents[i:i+batch_size],
            metadatas=metadatas[i:i+batch_size],
            ids=ids[i:i+batch_size]
        )
        
    logger.info("RAG database built successfully!")

class RAGRetriever:
    def __init__(self):
        self.ef = get_embedding_function()
        self.client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        self.collection = self.client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=self.ef
        )
        
    def retrieve(self, query: str, k: int = 3):
        results = self.collection.query(
            query_texts=[query],
            n_results=k
        )
        
        retrieved_context = []
        if results["documents"] and results["metadatas"]:
            for i in range(len(results["documents"][0])):
                doc = results["documents"][0][i]
                meta = results["metadatas"][0][i]
                retrieved_context.append({
                    "similar_customer_question": doc,
                    "historical_brand_reply": meta["brand_text"]
                })
        return retrieved_context

if __name__ == "__main__":
    build_rag_db()
