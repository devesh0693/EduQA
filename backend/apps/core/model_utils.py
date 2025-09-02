"""
Model utility functions for handling SentenceTransformer loading issues.
This module provides robust model loading with proper error handling for meta tensor issues.
"""

import logging
import os
import shutil
import torch
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class ModelLoader:
    """Utility class for loading SentenceTransformer models with robust error handling."""
    
    @staticmethod
    def clear_model_cache(model_name='all-mpnet-base-v2'):
        """Clear the model cache to force re-download."""
        try:
            # Clear sentence transformers cache
            cache_dir = os.path.expanduser('~/.cache/torch/sentence_transformers')
            model_cache_dir = os.path.join(cache_dir, f"sentence-transformers_{model_name.replace('/', '_')}")
            
            if os.path.exists(model_cache_dir):
                shutil.rmtree(model_cache_dir)
                logger.info(f"Cleared model cache: {model_cache_dir}")
            
            # Also clear HuggingFace cache
            hf_cache_dir = os.path.expanduser('~/.cache/huggingface')
            if os.path.exists(hf_cache_dir):
                for item in os.listdir(hf_cache_dir):
                    if model_name.replace('/', '--') in item:
                        item_path = os.path.join(hf_cache_dir, item)
                        try:
                            if os.path.isdir(item_path):
                                shutil.rmtree(item_path)
                            else:
                                os.remove(item_path)
                            logger.info(f"Cleared HF cache item: {item_path}")
                        except Exception as e:
                            logger.warning(f"Could not clear cache item {item_path}: {e}")
                            
        except Exception as e:
            logger.warning(f"Error clearing model cache: {e}")
    
    @staticmethod
    def load_sentence_transformer(model_name='all-mpnet-base-v2', device='cpu', max_retries=3):
        """
        Load SentenceTransformer model with robust error handling for meta tensor issues.
        
        Args:
            model_name (str): Name of the model to load
            device (str): Device to load the model on
            max_retries (int): Maximum number of retry attempts
            
        Returns:
            SentenceTransformer: Loaded model
            
        Raises:
            RuntimeError: If model loading fails after all retries
        """
        for attempt in range(max_retries):
            try:
                logger.info(f"Loading {model_name} (attempt {attempt + 1}/{max_retries})")
                
                # Try different loading strategies based on attempt
                if attempt == 0:
                    # Standard loading
                    model = SentenceTransformer(model_name, device=device)
                    
                elif attempt == 1:
                    # Clear cache and try again
                    logger.info("Clearing cache and retrying...")
                    ModelLoader.clear_model_cache(model_name)
                    model = SentenceTransformer(
                        model_name, 
                        device=device,
                        cache_folder=None  # Force re-download
                    )
                    
                else:
                    # Last resort: Load without device specification and move manually
                    logger.info("Trying manual device loading...")
                    model = SentenceTransformer(model_name)
                    
                    # Manually move all components to CPU
                    try:
                        model.to(device)
                    except RuntimeError as move_error:
                        if "meta tensor" in str(move_error):
                            # Handle meta tensor by moving each module individually
                            for name, module in model.named_modules():
                                if hasattr(module, 'weight') and module.weight.is_meta:
                                    # Reset the module to proper tensors
                                    module.to_empty(device=device)
                            model.to(device)
                        else:
                            raise move_error
                
                # Validate the model is working
                model.eval()
                test_embedding = model.encode('test', convert_to_tensor=False)
                
                if len(test_embedding) == 768:  # Expected dimension for all-mpnet-base-v2
                    logger.info(f"Successfully loaded {model_name} on {device}")
                    return model
                else:
                    raise ValueError(f"Unexpected embedding dimension: {len(test_embedding)}")
                    
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                
                if attempt == max_retries - 1:
                    # Last attempt failed
                    logger.error(f"Failed to load {model_name} after {max_retries} attempts")
                    raise RuntimeError(f"Could not load SentenceTransformer model: {str(e)}")
                
                # Clear any partially loaded model
                if 'model' in locals():
                    del model
                    
        raise RuntimeError(f"Exhausted all {max_retries} attempts to load {model_name}")


def get_or_create_model(model_name='all-mpnet-base-v2', device='cpu'):
    """
    Get or create a SentenceTransformer model with caching.
    
    Args:
        model_name (str): Name of the model to load
        device (str): Device to load the model on
        
    Returns:
        SentenceTransformer: Loaded and cached model
    """
    # Use a simple class-level cache
    cache_key = f"{model_name}_{device}"
    
    if not hasattr(get_or_create_model, '_cache'):
        get_or_create_model._cache = {}
    
    if cache_key not in get_or_create_model._cache:
        logger.info(f"Loading new model instance: {cache_key}")
        get_or_create_model._cache[cache_key] = ModelLoader.load_sentence_transformer(
            model_name=model_name,
            device=device
        )
    
    return get_or_create_model._cache[cache_key]
