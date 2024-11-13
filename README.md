"""
# Secure Offline RAG System

## Overview
This system implements an Ensemble Retrieval-Augmented Generation (RAG) approach for answering questions about  different things which includes a variety of content types, such as general text, technical documentation, and code snippets extracted from a standard. It uses advanced text processing, embeddings, and language models to provide accurate responses to any technical queries.

## Features
- Enhanced text splitting with context preservation
- Efficient document retrieval using FAISS
- Optimized model inference with 4-bit quantization
- Batched processing for improved performance

## Project Structure
- `application/`: Web application using Streamlit
- `outputs/`: Saves the log files and Leaderboard submission file
- `src/`: Source code modules
- `main.py`: Entry point
- `requirements.txt`: Dependencies

## System Architecture

### 1. Text Processing Module (`text_processing.py`)
- **EnhancedTextSplitter Class**
  - Handles specialized text splitting for Matter protocol documentation
  - Features:
    - Chunk size of 512 tokens with 50 token overlap
    - Preserves important technical phrases
    - Maintains question-answer context
    - Custom separators for technical documentation
  - Key Functions:
    - `preprocess_text()`: Preserves important technical phrases
    - `postprocess_chunk()`: Restores original text format
    - `process_qa_pair()`: Processes Q&A pairs while maintaining context

### 2. Embedding Module (`embedding.py`)
- **EmbeddingManager Class**
  - Manages document embeddings and similarity search
  - Features:
    - Uses SentenceTransformer model "multi-qa-mpnet-base-dot-v1"
    - GPU-accelerated embedding generation
    - FAISS index for efficient similarity search
  - Key Functions:
    - `encode_documents()`: Batch processing of document embeddings
    - `search()`: Retrieves similar documents using FAISS

### 3. Model Module (`model.py`)
- **ModelManager Class**
  - Handles language model operations
  - Features:
    - Uses Qwen2.5-14B-Instruct model
    - 4-bit quantization for efficiency
    - Optimized prompt formatting
  - Key Functions:
    - `generate_response()`: Generates model responses
    - `format_prompt()`: Structures prompts with system instructions

### 4. RAG System (`rag_system.py`)
- **RAGSystem Class**
  - Main system implementation
  - Features:
    - Combines embedding and model components
    - Batch processing of queries
    - Performance statistics tracking
  - Key Functions:
    - `process_training_data()`: Builds knowledge base
    - `process_test_data_batched()`: Handles test queries in batches

### 5. Main Script (`main.py`)
- **Command Line Interface**
  - Configurable parameters:
    - Training data path
    - Test data path
    - Model selection
    - Output directory
    - Batch size
  - Features:
    - Comprehensive logging
    - Error handling
    - Performance statistics

## Installation

### Prerequisites
- Python 3.11.10
- CUDA-capable GPU (recommended)
- 16GB+ RAM (My personal machine has GPU with  24 GB RTX 4090 - Hence all dependencies are with this machine only)

### Setup
1. Clone the repository:
```bash
git clone [repository-url]
cd [repository-name]
```

2. Create and activate a virtual environment:
```bash
python -m venv env
source env/bin/activate  # Linux/Mac
.\env\Scripts\activate   # Windows
```

3. Install dependencies:
* Before we jump on to requirements.txt - We need to run separately
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```
* Now we can go to this requirements.txt file

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage
```bash
python main.py --train path/to/train.csv --test path/to/test.csv
```

### Advanced Options
```bash
python main.py \
    --train path/to/train.csv \
    --test path/to/test.csv \
    --model "Qwen/Qwen2.5-14B-Instruct" \
    --output "outputs" \
    --batch-size 8 \
    --seed 42
```

### Command Line Arguments
- `--train`: Path to training data CSV (required)
- `--test`: Path to test data CSV (required)
- `--model`: Model name/path (default: "Qwen/Qwen2.5-14B-Instruct")
- `--output`: Output directory (default: "outputs")
- `--batch-size`: Processing batch size (default: 8)
- `--seed`: Random seed (default: 42)

## Output Structure
- Results saved as CSV files in the output directory
- Log files with detailed execution information
- Performance statistics including:
  - Processing success rate
  - Execution time
  - Query statistics

## Performance Optimization
- 4-bit quantization for reduced memory usage
- Batch processing for efficient throughput
- GPU acceleration for embeddings
- FAISS indexing for fast similarity search

## Error Handling
- Comprehensive error logging
- Graceful handling of runtime errors
- Input validation
- Resource cleanup


## Resource Requirements
- **Minimum:**
  - 16GB RAM
  - CUDA-capable GPU (My personal machine has GPU with  24 GB RTX 4090 - Hence all dependencies are with this machine only)
  - 50GB disk space
- **Recommended:**
  - 32GB RAM
  - GPU with 16GB+ VRAM (My personal machine has GPU with  24 GB RTX 4090 - Hence all dependencies are with this machine only)
  - 100GB SSD storage

## Limitations and Considerations
- Requires CUDA-capable GPU for optimal performance
- Model size requires significant memory
- Processing speed depends on hardware capabilities
- Requires proper error handling for production use

## Future Improvements
- Multi-GPU support
- Dynamic batch size adjustment
- Additional model support
- Enhanced error recovery
- Performance optimization


## Contributing

Contributions are welcome! Please feel free to submit pull requests.

## Contact Information

### Developer
* `Author`: Shravan Koninti
* `Email` : shravankumar224@gmail.com
* `LinkedIn` : [LinkedIn](https://www.linkedin.com/in/shravankoninti/)

## License

### MIT License
Copyright Notice
Copyright (c) 2024 Shravan Koninti

License Terms
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

* The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.


Disclaimer
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""







