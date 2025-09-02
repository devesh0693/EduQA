"""
Backend integration for Educational QA System
Connects the data preprocessor with the Flask/Django backend
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from flask import Flask, request, jsonify
from flask_cors import CORS

# Add the current directory to Python path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

# Import your data preprocessor
try:
    from data_preprocessor import DataPreprocessor
except ImportError as e:
    logging.error(f"Could not import DataPreprocessor: {e}")
    DataPreprocessor = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QABackend:
    def __init__(self, db_path: str = 'db.sqlite3'):
        """Initialize the QA backend"""
        self.db_path = db_path
        self.preprocessor = None
        
        if DataPreprocessor:
            try:
                self.preprocessor = DataPreprocessor(db_path=db_path)
                logger.info("DataPreprocessor initialized successfully")
            except Exception as e:
                logger.error(f"Error initializing DataPreprocessor: {e}")
        else:
            logger.error("DataPreprocessor not available")
    
    def initialize_datasets(self):
        """Initialize and load all datasets"""
        if not self.preprocessor:
            return {"error": "Preprocessor not available"}
        
        try:
            logger.info("Starting dataset initialization...")
            datasets = self.preprocessor.prepare_datasets()
            stats = self.preprocessor.get_dataset_stats()
            
            return {
                "success": True,
                "message": "Datasets initialized successfully",
                "stats": stats,
                "datasets_loaded": list(datasets.keys())
            }
        except Exception as e:
            logger.error(f"Error initializing datasets: {e}")
            return {"error": str(e)}
    
    def search_qa(self, query: str, limit: int = 10) -> Dict[str, Any]:
        """Search for QA pairs"""
        if not self.preprocessor:
            return {"error": "Preprocessor not available"}
        
        try:
            results = self.preprocessor.search_qa_pairs(query, limit)
            return {
                "success": True,
                "query": query,
                "results": results,
                "count": len(results)
            }
        except Exception as e:
            logger.error(f"Error searching QA pairs: {e}")
            return {"error": str(e)}
    
    def get_stats(self) -> Dict[str, Any]:
        """Get dataset statistics"""
        if not self.preprocessor:
            return {"error": "Preprocessor not available"}
        
        try:
            stats = self.preprocessor.get_dataset_stats()
            return {
                "success": True,
                "stats": stats
            }
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {"error": str(e)}
    
    def process_single_dataset(self, dataset_name: str, file_path: str) -> Dict[str, Any]:
        """Process a single dataset file"""
        if not self.preprocessor:
            return {"error": "Preprocessor not available"}
        
        try:
            path = Path(file_path)
            if not path.exists():
                return {"error": f"File not found: {file_path}"}
            
            # Determine dataset type and load accordingly
            if 'squad' in dataset_name.lower():
                examples = self.preprocessor.load_squad_dataset(path)
            elif 'marco' in dataset_name.lower():
                examples = self.preprocessor.load_ms_marco_dataset(path)
            else:
                examples = self.preprocessor.load_educational_dataset(path)
            
            # Save to database
            self.preprocessor.save_to_database(examples, dataset_name)
            
            return {
                "success": True,
                "message": f"Processed {len(examples)} examples from {dataset_name}",
                "count": len(examples)
            }
        except Exception as e:
            logger.error(f"Error processing dataset {dataset_name}: {e}")
            return {"error": str(e)}

# Create Flask app for API endpoints
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend integration

# Initialize backend
qa_backend = QABackend()

@app.route('/api/initialize', methods=['POST'])
def initialize_datasets():
    """Initialize all datasets"""
    result = qa_backend.initialize_datasets()
    return jsonify(result)

@app.route('/api/search', methods=['GET', 'POST'])
def search_qa():
    """Search QA pairs"""
    if request.method == 'POST':
        data = request.get_json()
        query = data.get('query', '')
        limit = data.get('limit', 10)
    else:
        query = request.args.get('query', '')
        limit = int(request.args.get('limit', 10))
    
    if not query:
        return jsonify({"error": "Query parameter is required"}), 400
    
    result = qa_backend.search_qa(query, limit)
    return jsonify(result)

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get dataset statistics"""
    result = qa_backend.get_stats()
    return jsonify(result)

@app.route('/api/process_dataset', methods=['POST'])
def process_dataset():
    """Process a single dataset"""
    data = request.get_json()
    dataset_name = data.get('dataset_name', '')
    file_path = data.get('file_path', '')
    
    if not dataset_name or not file_path:
        return jsonify({"error": "dataset_name and file_path are required"}), 400
    
    result = qa_backend.process_single_dataset(dataset_name, file_path)
    return jsonify(result)

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "preprocessor_available": qa_backend.preprocessor is not None,
        "database_path": qa_backend.db_path
    })

@app.route('/')
def index():
    """Root endpoint"""
    return jsonify({
        "message": "Educational QA System Backend",
        "endpoints": [
            "/api/initialize - POST - Initialize datasets",
            "/api/search - GET/POST - Search QA pairs",
            "/api/stats - GET - Get dataset statistics",
            "/api/process_dataset - POST - Process single dataset",
            "/api/health - GET - Health check"
        ]
    })

def setup_datasets():
    """Setup function to initialize datasets on startup"""
    logger.info("Setting up datasets...")
    
    # Check if datasets exist
    data_dir = Path('data/datasets')
    required_files = [
        'squad_2.0/train-v2.0.json',
        'squad_2.0/dev-v2.0.json',
        'educational_qa.json',
        'ms_marco_summary.json'
    ]
    
    missing_files = []
    for file_path in required_files:
        full_path = data_dir / file_path
        if not full_path.exists():
            missing_files.append(str(full_path))
    
    if missing_files:
        logger.warning(f"Missing dataset files: {missing_files}")
        logger.info("You can still run the server, but some datasets won't be available")
    
    # Initialize datasets if preprocessor is available
    if qa_backend.preprocessor:
        try:
            result = qa_backend.initialize_datasets()
            if result.get('success'):
                logger.info("Datasets initialized successfully")
                stats = result.get('stats', {})
                for source, count in stats.items():
                    logger.info(f"  {source}: {count} examples")
            else:
                logger.error(f"Dataset initialization failed: {result.get('error')}")
        except Exception as e:
            logger.error(f"Error during dataset setup: {e}")
    else:
        logger.error("Cannot initialize datasets - preprocessor not available")

def create_sample_data():
    """Create sample data for testing if no datasets are found"""
    sample_data = [
        {
            "id": "sample_1",
            "question": "What is machine learning?",
            "context": "Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from data without being explicitly programmed. It uses statistical techniques to give computer systems the ability to learn and improve from experience.",
            "answer_text": "a subset of artificial intelligence that focuses on algorithms that can learn from data",
            "start_char": 21,
            "end_char": 105,
            "is_impossible": False,
            "title": "Introduction to Machine Learning"
        },
        {
            "id": "sample_2", 
            "question": "What are neural networks?",
            "context": "Neural networks are computing systems inspired by biological neural networks. They consist of layers of interconnected nodes that process information. Each connection has a weight that adjusts during learning to improve accuracy.",
            "answer_text": "computing systems inspired by biological neural networks",
            "start_char": 21,
            "end_char": 77,
            "is_impossible": False,
            "title": "Neural Networks Basics"
        },
        {
            "id": "sample_3",
            "question": "What is deep learning?",
            "context": "Deep learning is a subset of machine learning that uses neural networks with multiple layers to model and understand complex patterns in data. It has revolutionized fields like computer vision and natural language processing.",
            "answer_text": "a subset of machine learning that uses neural networks with multiple layers",
            "start_char": 18,
            "end_char": 94,
            "is_impossible": False,
            "title": "Deep Learning Applications"
        }
    ]
    
    if qa_backend.preprocessor:
        try:
            qa_backend.preprocessor.save_to_database(sample_data, "sample_data")
            logger.info(f"Created {len(sample_data)} sample QA pairs")
        except Exception as e:
            logger.error(f"Error creating sample data: {e}")

if __name__ == '__main__':
    # Setup datasets on startup
    setup_datasets()
    
    # Create sample data if no real data exists
    stats = qa_backend.get_stats()
    if stats.get('stats', {}).get('total', 0) == 0:
        logger.info("No existing data found, creating sample data...")
        create_sample_data()
    
    # Run the Flask app
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('DEBUG', 'False').lower() == 'true'
    
    logger.info(f"Starting Educational QA Backend on port {port}")
    app.run(host='0.0.0.0', port=port, debug=debug)