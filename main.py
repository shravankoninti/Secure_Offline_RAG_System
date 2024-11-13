"""
Main entry point for the RAG system.
Includes command-line argument parsing and logging configuration.
"""

import time
import torch
import numpy as np
import pandas as pd
import argparse
import logging
import os
from datetime import datetime
import sys
from pathlib import Path
# Add the src directory to Python path
sys.path.append(str(Path(__file__).parent))
from src.rag_system import RAGSystem

def setup_logging(output_dir: str) -> logging.Logger:
    """
    Configure logging with both file and console handlers.
    
    Args:
        output_dir: Directory to store log files
        
    Returns:
        logging.Logger: Configured logger instance
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Configure logging format
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    
    # Create logger
    logger = logging.getLogger('RAGSystem')
    logger.setLevel(logging.INFO)
    
    # Create formatters and handlers
    formatter = logging.Formatter(log_format, date_format)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    log_file = Path(output_dir) / f'rag_system_{datetime.now():%Y%m%d_%H%M%S}.log'
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger

def validate_paths(train_path: str, test_path: str, output_dir: str) -> bool:
    """
    Validate input and output paths.
    
    Args:
        train_path: Path to training data
        test_path: Path to test data
        output_dir: Output directory path
        
    Returns:
        bool: True if all paths are valid
        
    Raises:
        FileNotFoundError: If input files don't exist
        PermissionError: If output directory can't be created
    """
    # Check input files
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training data not found at: {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test data not found at: {test_path}")
    
    # Create output directory
    try:
        os.makedirs(output_dir, exist_ok=True)
    except PermissionError:
        raise PermissionError(f"Cannot create output directory at: {output_dir}")
    
    return True

def main(
    train_path: str,
    test_path: str,
    model_name: str,
    output_dir: str,
    batch_size: int,
    logger: logging.Logger
) -> None:
    """
    Main execution function with enhanced logging and error handling.
    
    Args:
        train_path: Path to training data CSV
        test_path: Path to test data CSV
        model_name: Name of the model to use
        output_dir: Output directory path
        batch_size: Batch size for processing
        logger: Logger instance
    """
    try:
        start_time = time.time()
        
        # Initialize system
        logger.info("Initializing RAG system...")
        rag_system = RAGSystem(model_name=model_name)
        
        # Process training data
        logger.info("Loading and processing training data...")
        train_df = pd.read_csv(train_path)
        logger.info(f"Loaded {len(train_df)} training examples")
        rag_system.process_training_data(train_df)
        
        # Process test data
        logger.info("Processing test data...")
        test_df = pd.read_csv(test_path)
        logger.info(f"Processing {len(test_df)} test queries with batch size {batch_size}")
        results_df = rag_system.process_test_data_batched(test_df, batch_size=batch_size)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = Path(output_dir) / f'submission_{timestamp}.csv'
        results_df.to_csv(output_file, index=False)
        logger.info(f"Results saved to {output_file}")
        
        # Log statistics
        end_time = time.time()
        execution_time = end_time - start_time
        hours = int(execution_time // 3600)
        minutes = int((execution_time % 3600) // 60)
        seconds = int(execution_time % 60)
        
        logger.info("\nExecution Statistics:")
        logger.info(f"Total queries processed: {rag_system.stats['processed_queries']}")
        logger.info(f"Failed queries: {rag_system.stats['failed_queries']}")
        logger.info(f"Success rate: {(rag_system.stats['processed_queries'] - rag_system.stats['failed_queries']) / rag_system.stats['processed_queries'] * 100:.2f}%")
        logger.info(f"Execution time: {hours:02d}:{minutes:02d}:{seconds:02d}")
        
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Secure Offline RAG System',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        '--train',
        required=True,
        help='Path to training data CSV file'
    )
    parser.add_argument(
        '--test',
        required=True,
        help='Path to test data CSV file'
    )
    
    # Optional arguments
    parser.add_argument(
        '--model',
        default="Qwen/Qwen2.5-14B-Instruct",
        help='Model name or path'
    )
    parser.add_argument(
        '--output',
        default="outputs",
        help='Output directory for results and logs'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=8,
        help='Batch size for processing test data'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    args = parser.parse_args()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    try:
        # Setup logging
        logger = setup_logging(args.output)
        
        # Log configuration
        logger.info("Configuration:")
        for arg, value in vars(args).items():
            logger.info(f"{arg}: {value}")
        
        # Validate paths
        validate_paths(args.train, args.test, args.output)
        
        # Run main function
        main(
            train_path=args.train,
            test_path=args.test,
            model_name=args.model,
            output_dir=args.output,
            batch_size=args.batch_size,
            logger=logger
        )
        
    except KeyboardInterrupt:
        logger.info("\nExecution interrupted by user")
    except Exception as e:
        logger.error(f"Execution failed: {str(e)}", exc_info=True)
        raise