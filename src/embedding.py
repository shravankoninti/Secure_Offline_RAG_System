"""Embedding module for document retrieval."""

import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from typing import List, Tuple

class EmbeddingManager:
    """
    Manages document embeddings and similarity search using FAISS.
    
    Features:
    - GPU-accelerated embedding generation
    - Efficient similarity search
    - Batched processing for large datasets
    """
    
    def __init__(self):
        self.model = SentenceTransformer("multi-qa-mpnet-base-dot-v1")
        self.model.to('cuda')
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatL2(self.dimension)
        
    def encode_documents(self, documents: List[str], batch_size: int = 32) -> np.ndarray:
        """Encode documents in batches."""
        all_embeddings = []
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            embeddings = self.model.encode(batch)
            all_embeddings.append(embeddings)
        return np.vstack(all_embeddings)
    
    def add_to_index(self, embeddings: np.ndarray):
        """Add embeddings to FAISS index."""
        self.index.add(embeddings)
    
    def search(self, query: str, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """Search for similar documents."""
        query_embedding = self.model.encode([query])[0]
        return self.index.search(
            np.array([query_embedding]).astype('float32'), k
        )