"""
Script to train and save the QA model
"""
import os
import sys
import logging
from pathlib import Path

# Add backend to Python path
backend_dir = str(Path(__file__).parent)
if backend_dir not in sys.path:
    sys.path.append(backend_dir)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from ml_models.model_trainer import ModelTrainer
except ImportError as e:
    logger.error(f"Failed to import ModelTrainer: {e}")
    sys.exit(1)

def train_model():
    try:
        logger.info("Initializing model trainer...")
        trainer = ModelTrainer()
        
        logger.info("Preparing training data...")
        train_dataset, eval_dataset = trainer.prepare_training_data()
        
        if not train_dataset or not eval_dataset:
            logger.error("Failed to prepare training/validation data")
            return False
            
        logger.info(f"Training samples: {len(train_dataset)}")
        logger.info(f"Validation samples: {len(eval_dataset)}")
        
        # Define output directory
        output_dir = Path("ml_models/trained_model")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Starting model training...")
        results = trainer.train_model(
            output_dir=str(output_dir),
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
    success = train_model()
    if success:
        logger.info("Model training completed successfully!")
    else:
        logger.error("Model training failed")
