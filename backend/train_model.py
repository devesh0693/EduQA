"""
Script to train and save the QA model
"""
import os
import sys
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add backend to Python path
backend_dir = str(Path(__file__).parent)
if backend_dir not in sys.path:
    sys.path.append(backend_dir)

from ml_models.model_trainer import ModelTrainer
from ml_models.data_preprocessor import DataPreprocessor

def train_and_save_model():
    try:
        logger.info("Starting model training...")
        
        # Initialize components
        preprocessor = DataPreprocessor()
        trainer = ModelTrainer()
        
        # Prepare data
        logger.info("Preparing training data...")
        train_dataset, val_dataset = preprocessor.prepare_training_data()
        
        if not train_dataset or not val_dataset:
            logger.error("Failed to prepare training/validation data")
            return False
            
        logger.info(f"Training samples: {len(train_dataset)}")
        logger.info(f"Validation samples: {len(val_dataset)}")
        
        # Train model
        logger.info("Starting model training...")
        output_dir = Path("ml_models/trained_model")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = trainer.train_model(
            output_dir=output_dir,
            use_wandb=False,
            experiment_name="educational_qa_training"
        )
        
        if results:
            logger.info(f"Model training completed. Results: {results}")
            logger.info(f"Model saved to: {output_dir.absolute()}")
            return True
        else:
            logger.error("Model training failed")
            return False
            
    except Exception as e:
        logger.error(f"Error during model training: {str(e)}", exc_info=True)
        return False

if __name__ == "__main__":
    train_and_save_model()
