"""Main RAG system implementation."""

from typing import Dict, List
import pandas as pd
from tqdm.auto import tqdm
import torch
from src.text_processing import process_training_data
from src.embedding import EmbeddingManager
from src.model import ModelManager

class RAGSystem:
    """
    Ensemble Retrieval-Augmented Generation system.
    
    Features:
    - Enhanced text processing
    - Efficient document retrieval
    - Optimized model inference
    - Batched processing
    """
    
    def __init__(self, model_name: str = "Qwen/Qwen2.5-14B-Instruct"):
        self.embedding_manager = EmbeddingManager()
        self.model_manager = ModelManager(model_name=model_name)
        self.documents = []
        self.stats = {
            'processed_queries': 0,
            'failed_queries': 0
        }
    
    def process_training_data(self, train_df: pd.DataFrame):
        """Process training data and build index."""
        chunks = process_training_data(train_df)
        self.documents.extend(chunks)
        embeddings = self.embedding_manager.encode_documents(chunks)
        self.embedding_manager.add_to_index(embeddings)
    
    def process_test_data_batched(self, test_df: pd.DataFrame, batch_size: int = 8) -> pd.DataFrame:
        """Process test data in batches."""
        results = []
        batches = [test_df[i:i + batch_size] for i in range(0, len(test_df), batch_size)]
        
        for batch_df in tqdm(batches, desc="Processing batches"):
            batch_responses = []
            
            for _, row in batch_df.iterrows():
                try:
                    query = row['Query']
                    distances, indices = self.embedding_manager.search(query)
                    context = " ".join([self.documents[i] for i in indices[0]])
                    
                    response = self.model_manager.generate_response(query, context)
                    self.stats['processed_queries'] += 1
                    
                    batch_responses.append({
                        'trustii_id': row['trustii_id'],
                        'Query': query,
                        'Response': response
                    })
                    
                except Exception as e:
                    print(f"Error processing query: {str(e)}")
                    self.stats['failed_queries'] += 1
                    batch_responses.append({
                        'trustii_id': row['trustii_id'],
                        'Query': query,
                        'Response': "Error processing query"
                    })
            
            results.extend(batch_responses)
            torch.cuda.empty_cache()
        
        return pd.DataFrame(results)