"""
Configuration file for Educational QA System
Contains all model and dataset configurations
"""

import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / 'data'
DATASETS_DIR = DATA_DIR / 'datasets'
MODELS_DIR = BASE_DIR / 'ml_models'
LOGS_DIR = BASE_DIR / 'logs'

# Create directories if they don't exist
for directory in [DATA_DIR, DATASETS_DIR, MODELS_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Model Paths Configuration
MODEL_PATHS = {
    'bert': {
        'base': 'bert-base-uncased',
        'trained': BASE_DIR / 'trained_model',
        'checkpoint': BASE_DIR / 'test_output'
    },
    'roberta': {
        'base': 'roberta-base',
        'trained': BASE_DIR / 'trained_model'
    }
}

# Performance Targets
PERFORMANCE_TARGETS = {
    'accuracy': 0.85,
    'f1_score': 0.80,
    'inference_time': 0.5,  # seconds
    'memory_usage': 2000,   # MB
    'batch_processing_time': 2.0,  # seconds per batch
    'confidence_threshold': 0.1  # Minimum confidence score for answers
}

# Model Configuration
MODEL_CONFIG = {
    'bert_model_name': 'bert-base-uncased',
    'max_seq_length': 512,
    'doc_stride': 128,
    'max_query_length': 64,
    'max_answer_length': 30,
    'n_best_size': 20,  # Number of predictions to return for each question
    'batch_size': 16,
    'learning_rate': 3e-5,
    'num_epochs': 3,
    'warmup_steps': 1000,
    'weight_decay': 0.01,
    'gradient_accumulation_steps': 1,
    'fp16': False,  # Set to True if you have compatible GPU
    'device': 'cuda' if os.environ.get('CUDA_AVAILABLE') == 'true' else 'cpu'
}

# Dataset Configuration
DATASET_CONFIG = {
    # SQuAD 2.0 datasets
    'squad_train': DATASETS_DIR / 'squad_2.0' / 'train-v2.0.json',
    'squad_dev': DATASETS_DIR / 'squad_2.0' / 'dev-v2.0.json',
    'squad_sample': DATASETS_DIR / 'squad_sample.json',
    
    # Educational datasets
    'educational_qa': DATASETS_DIR / 'educational_qa.json',
    'educational_train': DATASETS_DIR / 'educational_train.json',
    'educational_val': DATASETS_DIR / 'educational_val.json',
    
    # MS MARCO datasets
    'ms_marco_train': DATASETS_DIR / 'ms_marco' / 'train.json',
    'ms_marco_dev': DATASETS_DIR / 'ms_marco' / 'dev.json',
    'ms_marco_summary': DATASETS_DIR / 'ms_marco_summary.json',
    'ms_marco_val': DATASETS_DIR / 'ms_marco_val.json',
    
    # Additional datasets from your structure
    'ms_marco_collection': DATASETS_DIR / 'ms_marco' / 'collection.tsv',
    'ms_marco_qrels_dev': DATASETS_DIR / 'ms_marco' / 'qrels.dev.tsv',
    'ms_marco_qrels_train': DATASETS_DIR / 'ms_marco' / 'qrels.train.tsv',
    'ms_marco_queries_dev': DATASETS_DIR / 'ms_marco' / 'queries.dev.tsv',
    'ms_marco_queries_eval': DATASETS_DIR / 'ms_marco' / 'queries.eval.tsv',
    'ms_marco_queries_train': DATASETS_DIR / 'ms_marco' / 'queries.train.tsv'
}

# Database Configuration
DATABASE_CONFIG = {
    'sqlite_path': BASE_DIR / 'db.sqlite3',
    'backup_interval': 3600,  # seconds
    'max_backup_files': 5
}

# API Configuration
API_CONFIG = {
    'host': '0.0.0.0',
    'port': int(os.environ.get('PORT', 5000)),
    'debug': os.environ.get('DEBUG', 'False').lower() == 'true',
    'cors_origins': ['http://localhost:3000', 'http://127.0.0.1:3000'],
    'rate_limit': '100 per hour',
    'max_query_length': 500,
    'max_results': 50,
    'default_results': 10
}

# Logging Configuration
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        },
        'detailed': {
            'format': '%(asctime)s [%(levelname)s] %(name)s:%(lineno)d: %(message)s'
        }
    },
    'handlers': {
        'default': {
            'level': 'INFO',
            'formatter': 'standard',
            'class': 'logging.StreamHandler',
        },
        'file': {
            'level': 'DEBUG',
            'formatter': 'detailed',
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': LOGS_DIR / 'eduqa.log',
            'maxBytes': 10485760,  # 10MB
            'backupCount': 5
        }
    },
    'loggers': {
        '': {
            'handlers': ['default', 'file'],
            'level': 'DEBUG',
            'propagate': False
        }
    }
}

# Search Configuration
SEARCH_CONFIG = {
    'enable_fuzzy_search': True,
    'fuzzy_threshold': 0.8,
    'enable_semantic_search': False,  # Enable when embeddings are available
    'max_context_length': 1000,
    'snippet_length': 200,
    'highlight_matches': True
}

# Training Configuration
TRAINING_CONFIG = {
    'output_dir': MODELS_DIR / 'trained_models',
    'save_steps': 1000,
    'eval_steps': 1000,
    'logging_steps': 100,
    'save_total_limit': 3,
    'evaluation_strategy': 'steps',
    'load_best_model_at_end': True,
    'metric_for_best_model': 'f1',
    'greater_is_better': True,
    'early_stopping_patience': 3,
    'dataloader_num_workers': 4,
    'prediction_loss_only': False,
    'remove_unused_columns': True
}

# Preprocessing Configuration
PREPROCESSING_CONFIG = {
    'clean_text': True,
    'lowercase': True,
    'remove_punctuation': False,
    'remove_stopwords': False,
    'min_context_length': 50,
    'min_question_length': 5,
    'max_examples_per_dataset': None,  # None for no limit
    'shuffle_datasets': True,
    'validation_split': 0.2
}

# Environment-specific overrides
if os.environ.get('ENVIRONMENT') == 'production':
    API_CONFIG['debug'] = False
    LOGGING_CONFIG['handlers']['default']['level'] = 'WARNING'
elif os.environ.get('ENVIRONMENT') == 'development':
    API_CONFIG['debug'] = True
    MODEL_CONFIG['batch_size'] = 8  # Smaller batch size for development

# Export commonly used paths
def get_dataset_path(dataset_name: str) -> Path:
    """Get path for a specific dataset"""
    return DATASET_CONFIG.get(dataset_name, DATASETS_DIR / f'{dataset_name}.json')

def get_model_path(model_name: str) -> Path:
    """Get path for a specific model"""
    return MODELS_DIR / model_name

def get_log_path(log_name: str = 'eduqa.log') -> Path:
    """Get path for log files"""
    return LOGS_DIR / log_name

# Validate configuration
def validate_config():
    """Validate configuration settings"""
    errors = []
    
    # Check if required directories exist
    for name, path in [('Data', DATA_DIR), ('Models', MODELS_DIR), ('Logs', LOGS_DIR)]:
        if not path.exists():
            errors.append(f"{name} directory does not exist: {path}")
    
    # Check model configuration
    if MODEL_CONFIG['max_seq_length'] > 512:
        errors.append("max_seq_length should not exceed 512 for BERT models")
    
    if MODEL_CONFIG['batch_size'] < 1:
        errors.append("batch_size must be at least 1")
    
    # Check API configuration
    if not (1024 <= API_CONFIG['port'] <= 65535):
        errors.append("port must be between 1024 and 65535")
    
    return errors

if __name__ == "__main__":
    # Validate configuration when run directly
    errors = validate_config()
    if errors:
        print("Configuration errors found:")
        for error in errors:
            print(f"  - {error}")
    else:
        print("Configuration validation passed!")
        
    # Print current configuration
    print(f"\nCurrent configuration:")
    print(f"  Base directory: {BASE_DIR}")
    print(f"  Data directory: {DATA_DIR}")
    print(f"  Models directory: {MODELS_DIR}")
    print(f"  Database path: {DATABASE_CONFIG['sqlite_path']}")
    print(f"  Model: {MODEL_CONFIG['bert_model_name']}")
    print(f"  API port: {API_CONFIG['port']}")
    print(f"  Debug mode: {API_CONFIG['debug']}")