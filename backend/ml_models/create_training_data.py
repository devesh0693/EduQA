"""
Script to create proper training data from your existing datasets
This will fix the data issues and create clean training samples
"""

import json
import logging
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def validate_json_file(file_path):
    """Validate and inspect JSON file structure"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            # Read first few lines to inspect structure
            first_lines = []
            for i, line in enumerate(f):
                first_lines.append(line.strip())
                if i >= 5:  # Read first 6 lines
                    break
        
        logger.info(f"First few lines of {file_path}:")
        for i, line in enumerate(first_lines):
            logger.info(f"  Line {i+1}: {line[:100]}...")
        
        # Try to parse the entire file
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            
        # Check if it's multiple JSON objects (JSONL format)
        if content.count('}\n{') > 0 or content.count('}\r\n{') > 0:
            logger.info("Detected JSONL format (multiple JSON objects)")
            return "jsonl"
        
        # Try parsing as single JSON
        try:
            data = json.loads(content)
            logger.info("Successfully parsed as single JSON object")
            return "json"
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {e}")
            return "invalid"
            
    except Exception as e:
        logger.error(f"Error validating JSON file: {e}")
        return "error"

def load_squad_data_robust(file_path):
    """Robustly load SQuAD data with multiple format support"""
    logger.info(f"Attempting to load SQuAD data from {file_path}")
    
    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        return None
    
    # First validate the file
    file_type = validate_json_file(file_path)
    
    if file_type == "error" or file_type == "invalid":
        logger.error("Cannot parse JSON file")
        return None
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            if file_type == "jsonl":
                # Handle JSONL format
                logger.info("Loading as JSONL format...")
                all_data = []
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line:
                        try:
                            data = json.loads(line)
                            all_data.append(data)
                        except json.JSONDecodeError as e:
                            logger.warning(f"Skipping invalid JSON on line {line_num}: {e}")
                
                # Try to find data structure
                if all_data and isinstance(all_data[0], dict):
                    if 'data' in all_data[0]:
                        return all_data[0]  # Standard SQuAD format
                    else:
                        # Create SQuAD-like structure
                        return {'data': all_data}
                        
            else:
                # Handle standard JSON format
                logger.info("Loading as standard JSON format...")
                content = f.read().strip()
                
                # Try to fix common JSON issues
                if content.endswith(','):
                    content = content[:-1]
                
                # Remove any trailing content after the main JSON
                try:
                    # Find the end of the first complete JSON object
                    decoder = json.JSONDecoder()
                    data, idx = decoder.raw_decode(content)
                    if idx < len(content):
                        remaining = content[idx:].strip()
                        if remaining:
                            logger.warning(f"Found extra content after JSON: {remaining[:100]}...")
                    return data
                except (json.JSONDecodeError, ValueError) as e:
                    logger.error(f"Failed to parse JSON: {e}")
                    # Try alternative parsing
                    try:
                        data = json.loads(content)
                        return data
                    except Exception as e2:
                        logger.error(f"Alternative parsing also failed: {e2}")
                        return None
        
    except Exception as e:
        logger.error(f"Error loading SQuAD data: {e}")
        return None

def create_clean_squad_sample():
    """Create a clean SQuAD sample file from the main dataset"""
    logger.info("Creating clean SQuAD sample...")
    
    # Try multiple possible paths
    possible_paths = [
        Path("data/datasets/squad_2.0/train-v2.0.json"),
        Path("data/datasets/squad/train-v2.0.json"),
        Path("data/squad_2.0/train-v2.0.json"),
        Path("train-v2.0.json"),
        Path("data/train-v2.0.json")
    ]
    
    squad_data = None
    squad_train_path = None
    
    for path in possible_paths:
        if path.exists():
            logger.info(f"Found SQuAD file at: {path}")
            squad_train_path = path
            squad_data = load_squad_data_robust(path)
            if squad_data:
                break
        else:
            logger.debug(f"Path not found: {path}")
    
    if not squad_data:
        logger.warning("No valid SQuAD training file found. Creating sample data instead...")
        return create_minimal_squad_sample()
    
    try:
        squad_sample_path = Path("data/datasets/squad_sample.json")
        
        # Extract examples
        examples = []
        
        # Handle different data structures
        if 'data' in squad_data:
            articles = squad_data['data']
        elif isinstance(squad_data, list):
            articles = squad_data
        else:
            logger.error("Unrecognized SQuAD data structure")
            return create_minimal_squad_sample()
        
        for article in articles:
            if 'paragraphs' in article:
                paragraphs = article['paragraphs']
            elif 'context' in article:
                # Direct paragraph format
                paragraphs = [article]
            else:
                continue
                
            for paragraph in paragraphs:
                context = paragraph.get('context', '')
                qas = paragraph.get('qas', [])
                
                for qa in qas:
                    # Only include answerable questions
                    if not qa.get('is_impossible', False) and qa.get('answers'):
                        answer = qa['answers'][0]
                        example = {
                            'id': qa.get('id', f'squad_{len(examples)}'),
                            'question': qa.get('question', ''),
                            'context': context,
                            'answer_text': answer.get('text', ''),
                            'answer_start': answer.get('answer_start', 0),
                            'is_impossible': False
                        }
                        
                        # Validate example
                        if (example['question'].strip() and 
                            example['context'].strip() and 
                            example['answer_text'].strip()):
                            examples.append(example)
                        
                        # Limit to 1000 examples for sample
                        if len(examples) >= 1000:
                            break
                if len(examples) >= 1000:
                    break
            if len(examples) >= 1000:
                break
        
        if not examples:
            logger.warning("No valid examples extracted from SQuAD data")
            return create_minimal_squad_sample()
        
        # Save clean sample
        squad_sample_path.parent.mkdir(parents=True, exist_ok=True)
        with open(squad_sample_path, 'w', encoding='utf-8') as f:
            json.dump(examples, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Created clean SQuAD sample with {len(examples)} examples at {squad_sample_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error creating SQuAD sample: {e}")
        return create_minimal_squad_sample()

def create_minimal_squad_sample():
    """Create a minimal SQuAD sample when the main dataset is not available"""
    logger.info("Creating minimal SQuAD sample...")
    
    try:
        minimal_examples = [
            {
                'id': 'sample_001',
                'question': 'What is the capital of France?',
                'context': 'France is a country in Western Europe. The capital and largest city of France is Paris. Paris is located in the north-central part of France along the Seine River.',
                'answer_text': 'Paris',
                'answer_start': 78,
                'is_impossible': False
            },
            {
                'id': 'sample_002',
                'question': 'Where is Paris located?',
                'context': 'France is a country in Western Europe. The capital and largest city of France is Paris. Paris is located in the north-central part of France along the Seine River.',
                'answer_text': 'in the north-central part of France along the Seine River',
                'answer_start': 106,
                'is_impossible': False
            },
            {
                'id': 'sample_003',
                'question': 'What is photosynthesis?',
                'context': 'Photosynthesis is the process by which green plants and some other organisms use sunlight to synthesize nutrients from carbon dioxide and water. This process converts light energy into chemical energy stored in glucose.',
                'answer_text': 'the process by which green plants and some other organisms use sunlight to synthesize nutrients from carbon dioxide and water',
                'answer_start': 17,
                'is_impossible': False
            },
            {
                'id': 'sample_004',
                'question': 'Who wrote Romeo and Juliet?',
                'context': 'William Shakespeare was an English playwright and poet. He is widely regarded as the greatest writer in the English language. Shakespeare wrote many famous plays including Romeo and Juliet, Hamlet, and Macbeth.',
                'answer_text': 'William Shakespeare',
                'answer_start': 0,
                'is_impossible': False
            },
            {
                'id': 'sample_005',
                'question': 'What causes earthquakes?',
                'context': 'Earthquakes are caused by the sudden release of energy stored in rocks. This usually happens when rocks break along fault lines due to tectonic plate movement. The released energy creates seismic waves that shake the ground.',
                'answer_text': 'the sudden release of energy stored in rocks',
                'answer_start': 24,
                'is_impossible': False
            }
        ]
        
        squad_sample_path = Path("data/datasets/squad_sample.json")
        squad_sample_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(squad_sample_path, 'w', encoding='utf-8') as f:
            json.dump(minimal_examples, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Created minimal SQuAD sample with {len(minimal_examples)} examples")
        return True
        
    except Exception as e:
        logger.error(f"Error creating minimal SQuAD sample: {e}")
        return False

def create_educational_samples():
    """Create better educational examples"""
    logger.info("Creating educational samples...")
    
    educational_examples = [
        {
            'id': 'edu_001',
            'question': 'What is photosynthesis?',
            'context': 'Photosynthesis is the process by which plants and other organisms convert light energy into chemical energy that can later be released to fuel the organism activities. This chemical energy is stored in carbohydrate molecules, such as sugars, which are synthesized from carbon dioxide and water. In most cases, oxygen is also released as a waste product. Most plants, most algae, and cyanobacteria perform photosynthesis.',
            'answer_text': 'the process by which plants and other organisms convert light energy into chemical energy',
            'answer_start': 17,
            'is_impossible': False
        },
        {
            'id': 'edu_002', 
            'question': 'What is the chemical formula for water?',
            'context': 'Water is an inorganic compound with the chemical formula H2O. It is a transparent, tasteless, odorless, and nearly colorless chemical substance. Water is essential for all known forms of life and is the most abundant substance on Earth surface.',
            'answer_text': 'H2O',
            'answer_start': 58,
            'is_impossible': False
        },
        {
            'id': 'edu_003',
            'question': 'Who developed the theory of relativity?',
            'context': 'Albert Einstein was a German-born theoretical physicist who developed the theory of relativity, one of the two pillars of modern physics alongside quantum mechanics. His work is also known for its influence on the philosophy of science.',
            'answer_text': 'Albert Einstein',
            'answer_start': 0,
            'is_impossible': False
        },
        {
            'id': 'edu_004',
            'question': 'What is the capital of France?',
            'context': 'France is a country located in Western Europe. Its capital city is Paris, which is also the largest city in the country. Paris is known for its art, fashion, gastronomy, and culture.',
            'answer_text': 'Paris',
            'answer_start': 70,
            'is_impossible': False
        },
        {
            'id': 'edu_005',
            'question': 'What causes earthquakes?',
            'context': 'Earthquakes are usually caused when rock underground suddenly breaks along a fault. This sudden release of energy causes the seismic waves that make the ground shake. When two blocks of rock or two plates are rubbing against each other, they stick a little.',
            'answer_text': 'when rock underground suddenly breaks along a fault',
            'answer_start': 34,
            'is_impossible': False
        },
        {
            'id': 'edu_006',
            'question': 'What is DNA?',
            'context': 'DNA, or deoxyribonucleic acid, is a molecule that contains the genetic instructions for the development and function of all living organisms. DNA is composed of two strands that wind around each other to form a double helix structure.',
            'answer_text': 'a molecule that contains the genetic instructions for the development and function of all living organisms',
            'answer_start': 38,
            'is_impossible': False
        }
    ]
    
    # Split into train and validation
    train_examples = educational_examples[:4]
    val_examples = educational_examples[4:]
    
    # Save files
    edu_train_path = Path("data/datasets/educational_train.json")
    edu_val_path = Path("data/datasets/educational_val.json")
    
    edu_train_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(edu_train_path, 'w', encoding='utf-8') as f:
        json.dump(train_examples, f, indent=2, ensure_ascii=False)
    
    with open(edu_val_path, 'w', encoding='utf-8') as f:
        json.dump(val_examples, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Created {len(train_examples)} educational training examples")
    logger.info(f"Created {len(val_examples)} educational validation examples")
    
    return True

def test_data_loading():
    """Test that the created data can be loaded properly"""
    logger.info("Testing data loading...")
    
    try:
        # Try to import the data preprocessor
        try:
            from ml_models.data_preprocessor import DataPreprocessor
            preprocessor = DataPreprocessor()
            datasets = preprocessor.prepare_datasets()
        except ImportError:
            logger.warning("Cannot import DataPreprocessor, testing file loading directly")
            return test_file_loading_directly()
        
        logger.info("Dataset loading results:")
        total_examples = 0
        
        for name, data in datasets.items():
            if data:
                logger.info(f"  {name}: {len(data)} examples")
                total_examples += len(data)
                # Show first example
                if data:
                    ex = data[0]
                    logger.info(f"    Sample Q: '{ex.get('question', 'N/A')[:50]}...'")
                    logger.info(f"    Sample A: '{ex.get('answer_text', 'N/A')}'")
            else:
                logger.info(f"  {name}: No data")
        
        if total_examples > 0:
            logger.info(f"Total examples available: {total_examples}")
            return True
        else:
            logger.warning("No examples loaded")
            return False
        
    except Exception as e:
        logger.error(f"Error testing data loading: {e}")
        return False

def test_file_loading_directly():
    """Test loading files directly without DataPreprocessor"""
    logger.info("Testing direct file loading...")
    
    files_to_test = [
        "data/datasets/squad_sample.json",
        "data/datasets/educational_train.json", 
        "data/datasets/educational_val.json"
    ]
    
    success_count = 0
    total_examples = 0
    
    for file_path in files_to_test:
        path = Path(file_path)
        if path.exists():
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                logger.info(f"  ✓ {file_path}: {len(data)} examples")
                success_count += 1
                total_examples += len(data)
            except Exception as e:
                logger.error(f"  ✗ {file_path}: Error loading - {e}")
        else:
            logger.warning(f"  ? {file_path}: File not found")
    
    logger.info(f"Successfully loaded {success_count}/{len(files_to_test)} files")
    logger.info(f"Total examples: {total_examples}")
    
    return success_count > 0

def main():
    """Main function to create all training data"""
    logger.info("=" * 50)
    logger.info("CREATING CLEAN TRAINING DATA")
    logger.info("=" * 50)
    
    success = True
    
    # Step 1: Create clean SQuAD sample (with fallback)
    logger.info("\nStep 1: Creating SQuAD sample data...")
    if not create_clean_squad_sample():
        logger.error("Failed to create SQuAD sample")
        success = False
    
    # Step 2: Create educational samples
    logger.info("\nStep 2: Creating educational samples...")
    if not create_educational_samples():
        logger.error("Failed to create educational samples")
        success = False
    
    # Step 3: Test data loading
    if success:
        logger.info("\nStep 3: Testing data loading...")
        if not test_data_loading():
            logger.warning("Data loading test had issues, but continuing...")
    
    # Final report
    logger.info("\n" + "=" * 50)
    if success:
        logger.info("✓ DATA CREATION COMPLETED!")
        logger.info("=" * 50)
        logger.info("Created files:")
        logger.info("  - data/datasets/squad_sample.json")
        logger.info("  - data/datasets/educational_train.json")
        logger.info("  - data/datasets/educational_val.json")
        logger.info("\nYou can now run training:")
        logger.info("  python backend/ml_models/test_msmarco.py")
        logger.info("  python backend/ml_models/model_trainer.py")
    else:
        logger.error("✗ DATA CREATION HAD ISSUES")
        logger.error("=" * 50)
        logger.error("Some steps failed, but partial data may be available")
    
    return success

if __name__ == "__main__":
    main()