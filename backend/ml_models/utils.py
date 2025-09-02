"""
Utility functions for Educational QA System
"""

import json
from typing import Tuple, List, Dict, Any, Optional, Union
import logging
import time
import os
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import torch
import numpy as np
from functools import wraps
import hashlib

logger = logging.getLogger(__name__)

def setup_logging(log_file: Optional[Path] = None, level: str = "INFO"):
    """Setup logging configuration"""
    log_level = getattr(logging, level.upper())
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Setup console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    
    # Setup file handler if specified
    handlers = [console_handler]
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)
    
    # Configure root logger
    logging.basicConfig(
        level=log_level,
        handlers=handlers,
        force=True
    )

def timer(func):
    """Decorator to measure function execution time"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        logger.info(f"{func.__name__} executed in {end_time - start_time:.4f} seconds")
        return result
    return wrapper

def ensure_dir(path: Union[str, Path]) -> Path:
    """Ensure directory exists"""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path

def save_json(data: Dict[str, Any], file_path: Union[str, Path]):
    """Save data to JSON file"""
    file_path = Path(file_path)
    ensure_dir(file_path.parent)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    
    logger.info(f"Data saved to {file_path}")

def load_json(file_path: Union[str, Path]) -> Dict[str, Any]:
    """Load data from JSON file"""
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    logger.info(f"Data loaded from {file_path}")
    return data

def get_device() -> torch.device:
    """Get the best available device"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
    else:
        device = torch.device('cpu')
        logger.info("Using CPU")
    
    return device

def set_seed(seed: int = 42):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
    # For PyTorch deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    logger.info(f"Random seed set to {seed}")

def format_time(seconds: float) -> str:
    """Format time in human readable format"""
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        seconds = seconds % 60
        return f"{minutes}m {seconds:.2f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = seconds % 60
        return f"{hours}h {minutes}m {seconds:.2f}s"

def get_memory_usage() -> Dict[str, float]:
    """Get current memory usage"""
    memory_info = {}
    
    if torch.cuda.is_available():
        memory_info['gpu_allocated'] = torch.cuda.memory_allocated() / 1024**3  # GB
        memory_info['gpu_cached'] = torch.cuda.memory_reserved() / 1024**3  # GB
    
    # System memory
    try:
        import psutil
        memory_info['cpu_used'] = psutil.virtual_memory().used / 1024**3  # GB
        memory_info['cpu_available'] = psutil.virtual_memory().available / 1024**3  # GB
    except ImportError:
        pass
    
    return memory_info

def clean_text(text: str) -> str:
    """Clean and normalize text"""
    if not text:
        return ""
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Handle special characters
    text = re.sub(r'[^\w\s.,!?;:()\-\'""]', ' ', text)
    
    return text.strip()

def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """Truncate text to specified length"""
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix

def calculate_text_statistics(text: str) -> Dict[str, int]:
    """Calculate basic text statistics"""
    words = text.split()
    sentences = re.split(r'[.!?]+', text)
    
    return {
        'characters': len(text),
        'words': len(words),
        'sentences': len([s for s in sentences if s.strip()]),
        'avg_word_length': np.mean([len(word) for word in words]) if words else 0,
        'avg_sentence_length': len(words) / len(sentences) if sentences else 0
    }

def create_cache_key(*args, **kwargs) -> str:
    """Create a unique cache key from arguments"""
    # Convert arguments to string
    key_parts = []
    
    for arg in args:
        if isinstance(arg, (str, int, float, bool)):
            key_parts.append(str(arg))
        else:
            key_parts.append(str(hash(str(arg))))
    
    for k, v in sorted(kwargs.items()):
        if isinstance(v, (str, int, float, bool)):
            key_parts.append(f"{k}:{v}")
        else:
            key_parts.append(f"{k}:{hash(str(v))}")
    
    # Create hash
    key_string = "|".join(key_parts)
    return hashlib.md5(key_string.encode()).hexdigest()

class SimpleCache:
    """Simple in-memory cache with TTL support"""
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 300):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache = {}
        self.timestamps = {}
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        if key not in self.cache:
            return None
        
        # Check TTL
        if time.time() - self.timestamps[key] > self.default_ttl:
            self.remove(key)
            return None
        
        return self.cache[key]
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set value in cache"""
        # Remove oldest entries if cache is full
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.timestamps.keys(), key=self.timestamps.get)
            self.remove(oldest_key)
        
        self.cache[key] = value
        self.timestamps[key] = time.time()
    
    def remove(self, key: str):
        """Remove key from cache"""
        if key in self.cache:
            del self.cache[key]
            del self.timestamps[key]
    
    def clear(self):
        """Clear all cache"""
        self.cache.clear()
        self.timestamps.clear()
    
    def size(self) -> int:
        """Get cache size"""
        return len(self.cache)

class ProgressTracker:
    """Simple progress tracker"""
    
    def __init__(self, total: int, description: str = "Processing"):
        self.total = total
        self.current = 0
        self.description = description
        self.start_time = time.time()
    
    def update(self, increment: int = 1):
        """Update progress"""
        self.current += increment
        
        if self.current % max(1, self.total // 20) == 0:  # Log every 5%
            percentage = (self.current / self.total) * 100
            elapsed = time.time() - self.start_time
            estimated_total = elapsed * self.total / self.current
            remaining = estimated_total - elapsed
            
            logger.info(
                f"{self.description}: {self.current}/{self.total} "
                f"({percentage:.1f}%) - ETA: {format_time(remaining)}"
            )
    
    def finish(self):
        """Mark as finished"""
        elapsed = time.time() - self.start_time
        logger.info(
            f"{self.description} completed: {self.total}/{self.total} "
            f"in {format_time(elapsed)}"
        )

def validate_model_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and normalize model configuration"""
    required_keys = ['bert_model_name', 'max_seq_length']
    
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")
    
    # Normalize values
    if config['max_seq_length'] > 512:
        logger.warning(f"max_seq_length {config['max_seq_length']} > 512, may cause memory issues")
    
    return config

def check_dataset_format(examples: List[Dict[str, Any]]) -> bool:
    """Check if dataset examples have required format"""
    required_keys = ['question', 'context']
    
    if not examples:
        logger.warning("Empty dataset")
        return False
    
    for i, example in enumerate(examples[:10]):  # Check first 10 examples
        for key in required_keys:
            if key not in example:
                logger.error(f"Example {i} missing required key: {key}")
                return False
        
        if not isinstance(example['question'], str) or not isinstance(example['context'], str):
            logger.error(f"Example {i} has invalid data types")
            return False
    
    logger.info(f"Dataset format validation passed for {len(examples)} examples")
    return True

def estimate_training_time(
    num_examples: int, 
    batch_size: int, 
    num_epochs: int,
    seconds_per_batch: float = 1.0
) -> Dict[str, float]:
    """Estimate training time"""
    batches_per_epoch = num_examples // batch_size
    total_batches = batches_per_epoch * num_epochs
    estimated_seconds = total_batches * seconds_per_batch
    
    return {
        'batches_per_epoch': batches_per_epoch,
        'total_batches': total_batches,
        'estimated_time_seconds': estimated_seconds,
        'estimated_time_formatted': format_time(estimated_seconds)
    }

def compare_model_sizes(model_paths: List[Path]) -> Dict[str, Dict[str, Any]]:
    """Compare sizes of different models"""
    results = {}
    
    for model_path in model_paths:
        if model_path.exists():
            total_size = 0
            file_count = 0
            
            for file_path in model_path.rglob('*'):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
                    file_count += 1
            
            results[model_path.name] = {
                'total_size_mb': total_size / (1024 * 1024),
                'file_count': file_count,
                'path': str(model_path)
            }
    
    return results

def create_model_info(model_path: Path, config: Dict[str, Any]) -> Dict[str, Any]:
    """Create model information dictionary"""
    model_info = {
        'model_path': str(model_path),
        'creation_time': time.strftime('%Y-%m-%d %H:%M:%S'),
        'config': config,
        'pytorch_version': torch.__version__,
        'device': str(get_device())
    }
    
    # Add size information if model exists
    if model_path.exists():
        size_info = compare_model_sizes([model_path])
        model_info.update(size_info.get(model_path.name, {}))
    
    return model_info

# Context managers
class temporary_seed:
    """Context manager for temporary random seed"""
    
    def __init__(self, seed: int):
        self.seed = seed
        self.original_state = None
    
    def __enter__(self):
        self.original_state = torch.get_rng_state()
        set_seed(self.seed)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        torch.set_rng_state(self.original_state)

class model_inference_mode:
    """Context manager for model inference"""
    
    def __init__(self, model):
        self.model = model
        self.training_mode = None
    
    def __enter__(self):
        self.training_mode = self.model.training
        self.model.eval()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.model.train(self.training_mode)

# Performance monitoring
class PerformanceMonitor:
    """Monitor model performance metrics"""
    
    def __init__(self):
        self.metrics = {
            'response_times': [],
            'memory_usage': [],
            'accuracy_scores': [],
            'confidence_scores': []
        }
        self.start_time = time.time()
    
    def log_response_time(self, response_time: float):
        """Log response time"""
        self.metrics['response_times'].append(response_time)
    
    def log_memory_usage(self):
        """Log current memory usage"""
        memory_info = get_memory_usage()
        self.metrics['memory_usage'].append(memory_info)
    
    def log_accuracy(self, accuracy: float):
        """Log accuracy score"""
        self.metrics['accuracy_scores'].append(accuracy)
    
    def log_confidence(self, confidence: float):
        """Log confidence score"""
        self.metrics['confidence_scores'].append(confidence)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        summary = {
            'total_runtime': time.time() - self.start_time,
            'total_requests': len(self.metrics['response_times'])
        }
        
        if self.metrics['response_times']:
            summary.update({
                'avg_response_time': np.mean(self.metrics['response_times']),
                'median_response_time': np.median(self.metrics['response_times']),
                'max_response_time': np.max(self.metrics['response_times']),
                'min_response_time': np.min(self.metrics['response_times']),
                'requests_per_second': len(self.metrics['response_times']) / summary['total_runtime']
            })
        
        if self.metrics['accuracy_scores']:
            summary.update({
                'avg_accuracy': np.mean(self.metrics['accuracy_scores']),
                'accuracy_std': np.std(self.metrics['accuracy_scores'])
            })
        
        if self.metrics['confidence_scores']:
            summary.update({
                'avg_confidence': np.mean(self.metrics['confidence_scores']),
                'confidence_std': np.std(self.metrics['confidence_scores'])
            })
        
        return summary
    
    def reset(self):
        """Reset all metrics"""
        self.metrics = {
            'response_times': [],
            'memory_usage': [],
            'accuracy_scores': [],
            'confidence_scores': []
        }
        self.start_time = time.time()

def check_system_requirements() -> Dict[str, Any]:
    """Check if system meets requirements for the QA system"""
    requirements = {
        'python_version': {
            'required': '3.8+',
            'current': f"{os.sys.version_info.major}.{os.sys.version_info.minor}",
            'meets_requirement': os.sys.version_info >= (3, 8)
        },
        'pytorch': {
            'available': True,
            'version': torch.__version__,
            'cuda_available': torch.cuda.is_available()
        }
    }
    
    # Check available memory
    try:
        import psutil
        memory = psutil.virtual_memory()
        requirements['memory'] = {
            'total_gb': memory.total / (1024**3),
            'available_gb': memory.available / (1024**3),
            'meets_requirement': memory.available > 4 * (1024**3)  # 4GB minimum
        }
    except ImportError:
        requirements['memory'] = {'status': 'psutil not available'}
    
    # Check GPU
    if torch.cuda.is_available():
        requirements['gpu'] = {
            'available': True,
            'name': torch.cuda.get_device_name(),
            'memory_gb': torch.cuda.get_device_properties(0).total_memory / (1024**3)
        }
    else:
        requirements['gpu'] = {'available': False}
    
    return requirements

def print_system_info():
    """Print system information"""
    requirements = check_system_requirements()
    
    print("=== System Information ===")
    print(f"Python Version: {requirements['python_version']['current']} ({'✅' if requirements['python_version']['meets_requirement'] else '❌'})")
    print(f"PyTorch Version: {requirements['pytorch']['version']}")
    print(f"CUDA Available: {'✅' if requirements['pytorch']['cuda_available'] else '❌'}")
    
    if 'memory' in requirements and 'total_gb' in requirements['memory']:
        memory = requirements['memory']
        print(f"System Memory: {memory['total_gb']:.1f}GB total, {memory['available_gb']:.1f}GB available ({'✅' if memory['meets_requirement'] else '❌'})")
    
    if requirements['gpu']['available']:
        gpu = requirements['gpu']
        print(f"GPU: {gpu['name']} ({gpu['memory_gb']:.1f}GB)")
    else:
        print("GPU: Not available")
    
    print("=" * 30)

# Data validation utilities
def validate_qa_example(example: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validate a single QA example"""
    errors = []
    
    # Required fields
    required_fields = ['question', 'context']
    for field in required_fields:
        if field not in example:
            errors.append(f"Missing required field: {field}")
        elif not isinstance(example[field], str):
            errors.append(f"Field {field} must be string")
        elif not example[field].strip():
            errors.append(f"Field {field} cannot be empty")
    
    # Optional fields validation
    if 'answer_text' in example:
        if not isinstance(example['answer_text'], str):
            errors.append("answer_text must be string")
        
        # Check if answer exists in context
        if example['answer_text'] and example['answer_text'] not in example.get('context', ''):
            errors.append("answer_text not found in context")
    
    if 'start_char' in example:
        if not isinstance(example['start_char'], int):
            errors.append("start_char must be integer")
        elif example['start_char'] < 0 and not example.get('is_impossible', False):
            errors.append("start_char must be non-negative for answerable questions")
    
    return len(errors) == 0, errors

def batch_validate_dataset(examples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Validate entire dataset"""
    total_examples = len(examples)
    valid_examples = 0
    all_errors = []
    
    for i, example in enumerate(examples):
        is_valid, errors = validate_qa_example(example)
        if is_valid:
            valid_examples += 1
        else:
            all_errors.extend([f"Example {i}: {error}" for error in errors])
    
    return {
        'total_examples': total_examples,
        'valid_examples': valid_examples,
        'invalid_examples': total_examples - valid_examples,
        'validity_rate': valid_examples / total_examples if total_examples > 0 else 0,
        'errors': all_errors[:100]  # Limit to first 100 errors
    }

# File utilities
def safe_filename(filename: str) -> str:
    """Create a safe filename by removing invalid characters"""
    # Remove invalid characters
    safe_name = re.sub(r'[<>:"/\\|?*]', '_', filename)
    
    # Remove multiple underscores
    safe_name = re.sub(r'_+', '_', safe_name)
    
    # Trim and ensure not empty
    safe_name = safe_name.strip('_')
    if not safe_name:
        safe_name = 'unnamed'
    
    return safe_name

def backup_file(file_path: Union[str, Path], backup_dir: Optional[Path] = None) -> Path:
    """Create a backup of a file"""
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    if backup_dir is None:
        backup_dir = file_path.parent / 'backups'
    
    ensure_dir(backup_dir)
    
    # Create backup filename with timestamp
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    backup_name = f"{file_path.stem}_{timestamp}{file_path.suffix}"
    backup_path = backup_dir / backup_name
    
    # Copy file
    import shutil
    shutil.copy2(file_path, backup_path)
    
    logger.info(f"Backup created: {backup_path}")
    return backup_path

# Configuration utilities
def merge_configs(base_config: Dict, user_config: Dict) -> Dict:
    """Merge user configuration with base configuration"""
    merged = base_config.copy()
    
    for key, value in user_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    
    return merged

def validate_config(config: Dict[str, Any], schema: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validate configuration against schema"""
    errors = []
    
    # Check required keys
    for key, requirements in schema.items():
        if requirements.get('required', False) and key not in config:
            errors.append(f"Missing required configuration: {key}")
            continue
        
        if key in config:
            value = config[key]
            expected_type = requirements.get('type')
            
            # Type validation
            if expected_type and not isinstance(value, expected_type):
                errors.append(f"Configuration {key} must be of type {expected_type.__name__}")
            
            # Range validation
            if 'min_value' in requirements and value < requirements['min_value']:
                errors.append(f"Configuration {key} must be >= {requirements['min_value']}")
            
            if 'max_value' in requirements and value > requirements['max_value']:
                errors.append(f"Configuration {key} must be <= {requirements['max_value']}")
            
            # Choices validation
            if 'choices' in requirements and value not in requirements['choices']:
                errors.append(f"Configuration {key} must be one of {requirements['choices']}")
    
    return len(errors) == 0, errors

# Debugging utilities
def debug_model_inputs(inputs: Dict[str, torch.Tensor], tokenizer=None):
    """Debug model inputs by printing shapes and sample data"""
    print("=== Model Input Debug ===")
    
    for key, tensor in inputs.items():
        print(f"{key}: shape={tensor.shape}, dtype={tensor.dtype}")
        
        if key == 'input_ids' and tokenizer:
            # Decode first example
            if len(tensor.shape) > 1:
                decoded = tokenizer.decode(tensor[0], skip_special_tokens=False)
                print(f"  Decoded (first example): {decoded[:200]}...")
            else:
                decoded = tokenizer.decode(tensor, skip_special_tokens=False)
                print(f"  Decoded: {decoded[:200]}...")
        
        # Show first few values
        flat_tensor = tensor.flatten()
        sample_values = flat_tensor[:10].tolist()
        print(f"  Sample values: {sample_values}")
    
    print("=" * 30)

def profile_function(func):
    """Decorator to profile function execution"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        import cProfile
        import pstats
        from io import StringIO
        
        profiler = cProfile.Profile()
        profiler.enable()
        
        try:
            result = func(*args, **kwargs)
        finally:
            profiler.disable()
        
        # Print profile results
        string_io = StringIO()
        stats = pstats.Stats(profiler, stream=string_io)
        stats.sort_stats('cumulative')
        stats.print_stats(20)  # Top 20 functions
        
        print(f"\n=== Profile for {func.__name__} ===")
        print(string_io.getvalue())
        print("=" * 50)
        
        return result
    
    return wrapper

if __name__ == "__main__":
    # Test utility functions
    print("Testing Educational QA System Utilities...")
    
    # Test system requirements
    print_system_info()
    
    # Test text utilities
    sample_text = "This is a sample text for testing   various utility functions."
    print(f"\nOriginal text: '{sample_text}'")
    print(f"Cleaned text: '{clean_text(sample_text)}'")
    print(f"Text statistics: {calculate_text_statistics(sample_text)}")
    
    # Test cache
    cache = SimpleCache(max_size=3)
    cache.set("key1", "value1")
    cache.set("key2", "value2")
    print(f"\nCache test: {cache.get('key1')}")
    print(f"Cache size: {cache.size()}")
    
    # Test progress tracker
    tracker = ProgressTracker(100, "Testing")
    for i in range(0, 101, 25):
        tracker.update(25)
    tracker.finish()
    
    print("\nUtility functions test completed!")