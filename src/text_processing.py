"""Text processing module for documentation."""

from typing import List, Dict, Optional, Tuple
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter

class EnhancedTextSplitter:
    """
    Enhanced text splitting with context preservation for Matter protocol documentation.
    
    Features:
    - Preserves important technical phrases
    - Maintains question-answer context
    - Optimized chunk sizes for technical content
    """
    
    def __init__(self):
        self.chunk_size = 512
        self.chunk_overlap = 50
        self.separators = ["\n\n", "\n", ".", ",", " ", ""]
        self.important_phrases = [
            "Matter protocol",
            "Access Control",
            "Device Attestation",
            "Node Operational",
            "CASE protocol",
            "PASE protocol",
            "Secure Channel",
            "Message Layer",
            "Certificate",
            "Commissioning"
        ]
        self.base_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=self.separators
        )
    
    def preprocess_text(self, text: str) -> str:
        """Preserve important context by tokenizing technical phrases."""
        for idx, phrase in enumerate(self.important_phrases):
            text = text.replace(phrase, f"SPECIAL_TOKEN_{idx}_{phrase.replace(' ', '_')}")
        return text
    
    def postprocess_chunk(self, chunk: str) -> str:
        """Restore technical phrases from tokens."""
        for idx, phrase in enumerate(self.important_phrases):
            chunk = chunk.replace(f"SPECIAL_TOKEN_{idx}_{phrase.replace(' ', '_')}", phrase)
        return chunk
    
    def process_qa_pair(self, question: str, answer: str) -> List[str]:
        """Process Q&A pairs while maintaining context."""
        combined = f"Question: {question}\nAnswer: {answer}"
        preprocessed = self.preprocess_text(combined)
        chunks = self.base_splitter.split_text(preprocessed)
        chunks = [self.postprocess_chunk(chunk) for chunk in chunks]
        
        final_chunks = []
        for chunk in chunks:
            if not chunk.startswith("Question:"):
                question_preview = question[:50] + "..." if len(question) > 50 else question
                chunk = f"Context for: {question_preview}\n{chunk}"
            final_chunks.append(chunk)
        
        return final_chunks
    

def process_training_data(train_df: pd.DataFrame) -> List[str]:
    """Process training data using enhanced chunking."""
    splitter = EnhancedTextSplitter()
    all_chunks = []
    
    for _, row in train_df.iterrows():
        chunks = splitter.process_qa_pair(row['Query'], row['Response'])
        all_chunks.extend(chunks)
    
    return all_chunks