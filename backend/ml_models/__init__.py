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

# Add the parent directory to Python path to find ml_models
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.append(str(parent_dir))

# Try different import paths
DataPreprocessor = None
try:
    # First try importing from ml_models directory
    from ml_models.data_preprocessor import DataPreprocessor
    print("Successfully imported DataPreprocessor from ml_models")
except ImportError:
    try:
        # Try importing from current directory
        from data_preprocessor import DataPreprocessor
        print("Successfully imported DataPreprocessor from current directory")
    except ImportError:
        try:
            # Try importing from parent directory
            sys.path.append(str(parent_dir / 'ml_models'))
            from data_preprocessor import DataPreprocessor
            print("Successfully imported DataPreprocessor from parent/ml_models")
        except ImportError as e:
            print(f"Could not import DataPreprocessor from any location: {e}")
            print(f"Current directory: {current_dir}")
            print(f"Python path: {sys.path}")
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
                # Create a mock preprocessor for basic functionality
                self.create_mock_preprocessor()
        else:
            logger.error("DataPreprocessor not available, creating mock preprocessor")
            self.create_mock_preprocessor()
    
    def create_mock_preprocessor(self):
        """Create a mock preprocessor with basic functionality"""
        class MockPreprocessor:
            def __init__(self, db_path):
                self.db_path = db_path
                self.sample_data = [
                    {
                        'id': 'sample_1',
                        'question': 'What is machine learning?',
                        'context': 'Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from data without being explicitly programmed.',
                        'answer': 'a subset of artificial intelligence that focuses on algorithms that can learn from data',
                        'source': 'sample_data',
                        'title': 'Machine Learning Basics',
                        'relevance': 0.95
                    },
                    {
                        'id': 'sample_2',
                        'question': 'What are neural networks?',
                        'context': 'Neural networks are computing systems inspired by biological neural networks. They consist of layers of interconnected nodes that process information.',
                        'answer': 'computing systems inspired by biological neural networks',
                        'source': 'sample_data', 
                        'title': 'Neural Networks',
                        'relevance': 0.90
                    }
                ]
            
            def search_qa_pairs(self, query, limit=10):
                # Simple text search in sample data
                results = []
                query_lower = query.lower()
                for item in self.sample_data:
                    if (query_lower in item['question'].lower() or 
                        query_lower in item['context'].lower() or
                        query_lower in item['answer'].lower()):
                        results.append(item)
                return results[:limit]
            
            def get_dataset_stats(self):
                return {'sample_data': len(self.sample_data), 'total': len(self.sample_data)}
            
            def prepare_datasets(self):
                return {'sample_data': self.sample_data}
        
        self.preprocessor = MockPreprocessor(self.db_path)
        logger.info("Mock preprocessor created with sample data")
    
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
                "datasets_loaded": list(datasets.keys()) if isinstance(datasets, dict) else ["sample_data"]
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
        
        # For mock preprocessor, return success message
        if hasattr(self.preprocessor, 'sample_data'):
            return {
                "success": True,
                "message": f"Mock processing completed for {dataset_name}",
                "count": len(self.preprocessor.sample_data)
            }
        
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
        query = data.get('query', '') if data else ''
        limit = data.get('limit', 10) if data else 10
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
    if not data:
        return jsonify({"error": "JSON data required"}), 400
        
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
        "preprocessor_type": "real" if DataPreprocessor and not hasattr(qa_backend.preprocessor, 'sample_data') else "mock",
        "database_path": qa_backend.db_path
    })

@app.route('/')
def index():
    """Root endpoint"""
    return jsonify({
        "message": "Educational QA System Backend",
        "status": "running",
        "preprocessor_available": qa_backend.preprocessor is not None,
        "endpoints": [
            "POST /api/initialize - Initialize datasets",
            "GET/POST /api/search - Search QA pairs",
            "GET /api/stats - Get dataset statistics", 
            "POST /api/process_dataset - Process single dataset",
            "GET /api/health - Health check"
        ]
    })

def setup_datasets():
    """Setup function to initialize datasets on startup"""
    logger.info("Setting up datasets...")
    
    # Check if datasets exist
    data_dir = Path('data/datasets')
    if not data_dir.exists():
        logger.warning(f"Data directory not found: {data_dir}")
        return
    
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

if __name__ == '__main__':
    print(f"Starting from directory: {Path.cwd()}")
    print(f"Python path includes: {sys.path[:3]}...")
    
    # Setup datasets on startup
    setup_datasets()
    
    # Run the Flask app
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('DEBUG', 'True').lower() == 'true'
    
    logger.info(f"Starting Educational QA Backend on port {port}")
    print(f"Server running at http://localhost:{port}")
    print("Available endpoints:")
    print("  GET  / - API information")
    print("  GET  /api/health - Health check")
    print("  POST /api/initialize - Initialize datasets")
    print("  GET  /api/search?query=... - Search QA pairs")
    print("  GET  /api/stats - Dataset statistics")
    print("\nPress Ctrl+C to stop the server")
    
    app.run(host='0.0.0.0', port=port, debug=debug)