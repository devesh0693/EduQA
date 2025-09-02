"""
Enhanced MS MARCO Data Processor
Processes MS MARCO dataset files and creates training-ready samples
"""

import json
import logging
import pandas as pd
from pathlib import Path
import sys
from collections import defaultdict
import random

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MSMARCOProcessor:
    def __init__(self, data_dir="data/datasets/ms_marco"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path("data/datasets")
        
    def load_queries(self, file_path):
        """Load queries from TSV file"""
        logger.info(f"Loading queries from {file_path}")
        queries = {}
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        parts = line.strip().split('\t')
                        if len(parts) >= 2:
                            qid = int(parts[0])
                            query = parts[1]
                            queries[qid] = query
                        else:
                            logger.warning(f"Invalid line format at {line_num}: {line[:50]}...")
                    except ValueError as e:
                        logger.warning(f"Error parsing line {line_num}: {e}")
                        continue
                        
            logger.info(f"Loaded {len(queries)} queries")
            return queries
            
        except Exception as e:
            logger.error(f"Error loading queries: {e}")
            return {}
    
    def load_collection(self, file_path, max_passages=50000):
        """Load passage collection from TSV file"""
        logger.info(f"Loading collection from {file_path} (max {max_passages} passages)")
        collection = {}
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        parts = line.strip().split('\t')
                        if len(parts) >= 2:
                            pid = int(parts[0])
                            passage = parts[1]
                            collection[pid] = passage
                            
                            if len(collection) >= max_passages:
                                logger.info(f"Reached max passages limit: {max_passages}")
                                break
                        else:
                            logger.warning(f"Invalid line format at {line_num}: {line[:50]}...")
                    except ValueError as e:
                        logger.warning(f"Error parsing line {line_num}: {e}")
                        continue
                    except Exception as e:
                        logger.warning(f"Unexpected error at line {line_num}: {e}")
                        continue
                        
            logger.info(f"Loaded {len(collection)} passages")
            return collection
            
        except Exception as e:
            logger.error(f"Error loading collection: {e}")
            return {}
    
    def load_qrels(self, file_path):
        """Load relevance judgments from TSV file"""
        logger.info(f"Loading qrels from {file_path}")
        qrels = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        parts = line.strip().split('\t')
                        if len(parts) >= 4:
                            qid = int(parts[0])
                            # parts[1] is usually '0' (unused)
                            pid = int(parts[2])
                            relevance = int(parts[3])
                            qrels.append({
                                'qid': qid,
                                'pid': pid, 
                                'relevance': relevance
                            })
                        else:
                            logger.warning(f"Invalid qrels line at {line_num}: {line[:50]}...")
                    except ValueError as e:
                        logger.warning(f"Error parsing qrels line {line_num}: {e}")
                        continue
                        
            logger.info(f"Loaded {len(qrels)} relevance judgments")
            return qrels
            
        except Exception as e:
            logger.error(f"Error loading qrels: {e}")
            return []
    
    def create_qa_examples(self, queries, collection, qrels, max_examples=5000):
        """Create QA examples from MS MARCO data"""
        logger.info(f"Creating QA examples (max {max_examples})")
        
        examples = []
        
        # Group qrels by query
        qrels_by_query = defaultdict(list)
        for qrel in qrels:
            qrels_by_query[qrel['qid']].append(qrel)
        
        processed_queries = 0
        for qid, query_qrels in qrels_by_query.items():
            if processed_queries >= max_examples:
                break
                
            if qid not in queries:
                continue
            
            question = queries[qid]
            
            # Process each relevant passage for this query
            for qrel in query_qrels:
                if processed_queries >= max_examples:
                    break
                    
                pid = qrel['pid']
                relevance = qrel['relevance']
                
                if pid not in collection:
                    continue
                
                context = collection[pid]
                
                # Create example
                example = {
                    'id': f'msmarco_{qid}_{pid}',
                    'question': question,
                    'context': context,
                    'relevance': relevance
                }
                
                # For MS MARCO, we don't have exact answer spans
                # We'll mark highly relevant passages as "answerable" with empty answer
                if relevance >= 1:
                    example.update({
                        'answer_text': '',
                        'answer_start': -1,
                        'is_impossible': True  # No exact answer span available
                    })
                else:
                    example.update({
                        'answer_text': '',
                        'answer_start': -1,
                        'is_impossible': True
                    })
                
                examples.append(example)
                processed_queries += 1
        
        logger.info(f"Created {len(examples)} QA examples")
        return examples
    
    def create_synthetic_answerable_examples(self, examples, num_synthetic=1000):
        """Create synthetic answerable examples from MS MARCO data"""
        logger.info(f"Creating {num_synthetic} synthetic answerable examples")
        
        synthetic_examples = []
        
        # Filter for relevant examples
        relevant_examples = [ex for ex in examples if ex.get('relevance', 0) >= 1]
        
        if not relevant_examples:
            logger.warning("No relevant examples found for synthetic generation")
            return []
        
        for i in range(min(num_synthetic, len(relevant_examples))):
            original = relevant_examples[i % len(relevant_examples)]
            
            # Create a synthetic answerable example
            # Extract potential answer from context based on question keywords
            question = original['question'].lower()
            context = original['context']
            
            # Simple heuristic: find sentences that might contain answers
            sentences = context.split('. ')
            
            best_sentence = ""
            best_score = 0
            
            for sentence in sentences:
                sentence_lower = sentence.lower()
                score = sum(1 for word in question.split() if word in sentence_lower and len(word) > 3)
                if score > best_score:
                    best_score = score
                    best_sentence = sentence.strip()
            
            if best_sentence:
                # Find the position of the best sentence in the context
                answer_start = context.find(best_sentence)
                if answer_start >= 0:
                    synthetic_example = {
                        'id': f'msmarco_synthetic_{i}',
                        'question': original['question'],
                        'context': context,
                        'answer_text': best_sentence,
                        'answer_start': answer_start,
                        'is_impossible': False,
                        'source': 'ms_marco_synthetic'
                    }
                    synthetic_examples.append(synthetic_example)
        
        logger.info(f"Created {len(synthetic_examples)} synthetic answerable examples")
        return synthetic_examples
    
    def process_msmarco_files(self):
        """Process all MS MARCO files and create training samples"""
        logger.info("Processing MS MARCO files...")
        
        # Define file paths
        files = {
            'queries_train': self.data_dir / 'queries.train.tsv',
            'queries_dev': self.data_dir / 'queries.dev.tsv', 
            'queries_eval': self.data_dir / 'queries.eval.tsv',
            'collection': self.data_dir / 'collection.tsv',
            'qrels_train': self.data_dir / 'qrels.train.tsv',
            'qrels_dev': self.data_dir / 'qrels.dev.tsv'
        }
        
        # Check which files exist
        existing_files = {}
        for name, path in files.items():
            if path.exists():
                existing_files[name] = path
                logger.info(f"Found: {name} -> {path}")
            else:
                logger.warning(f"Missing: {name} -> {path}")
        
        if not existing_files:
            logger.error("No MS MARCO files found!")
            return False
        
        # Load data progressively
        all_examples = []
        
        # Load collection first (needed for all examples)
        if 'collection' in existing_files:
            collection = self.load_collection(existing_files['collection'], max_passages=10000)
        else:
            logger.error("Collection file is required but not found")
            return False
        
        # Process training data if available
        if 'queries_train' in existing_files and 'qrels_train' in existing_files:
            logger.info("Processing training data...")
            queries_train = self.load_queries(existing_files['queries_train'])
            qrels_train = self.load_qrels(existing_files['qrels_train'])
            
            train_examples = self.create_qa_examples(
                queries_train, collection, qrels_train, max_examples=2000
            )
            all_examples.extend(train_examples)
            
            # Create synthetic answerable examples
            synthetic_examples = self.create_synthetic_answerable_examples(train_examples, 500)
            all_examples.extend(synthetic_examples)
        
        # Process dev/eval data if available
        for data_type in ['dev', 'eval']:
            queries_key = f'queries_{data_type}'
            qrels_key = f'qrels_{data_type}'
            
            if queries_key in existing_files:
                logger.info(f"Processing {data_type} data...")
                queries = self.load_queries(existing_files[queries_key])
                
                if qrels_key in existing_files:
                    qrels = self.load_qrels(existing_files[qrels_key])
                    examples = self.create_qa_examples(
                        queries, collection, qrels, max_examples=500
                    )
                    all_examples.extend(examples)
                else:
                    # Create examples without relevance judgments
                    logger.info(f"No qrels for {data_type}, creating examples without relevance")
                    for qid, question in list(queries.items())[:200]:  # Limit to 200
                        # Use random passages for this question
                        random_pids = random.sample(list(collection.keys()), min(3, len(collection)))
                        for pid in random_pids:
                            example = {
                                'id': f'msmarco_{data_type}_{qid}_{pid}',
                                'question': question,
                                'context': collection[pid],
                                'answer_text': '',
                                'answer_start': -1,
                                'is_impossible': True,
                                'source': f'ms_marco_{data_type}'
                            }
                            all_examples.append(example)
        
        if not all_examples:
            logger.error("No examples were created!")
            return False
        
        # Split into train/validation
        random.shuffle(all_examples)
        split_idx = int(len(all_examples) * 0.8)
        train_examples = all_examples[:split_idx]
        val_examples = all_examples[split_idx:]
        
        # Save the processed data
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save training data
        train_path = self.output_dir / 'ms_marco_train.json'
        with open(train_path, 'w', encoding='utf-8') as f:
            json.dump(train_examples, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(train_examples)} training examples to {train_path}")
        
        # Save validation data
        val_path = self.output_dir / 'ms_marco_val.json'
        with open(val_path, 'w', encoding='utf-8') as f:
            json.dump(val_examples, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(val_examples)} validation examples to {val_path}")
        
        # Create summary
        summary = {
            'total_examples': len(all_examples),
            'train_examples': len(train_examples),
            'val_examples': len(val_examples),
            'answerable_examples': len([ex for ex in all_examples if not ex.get('is_impossible', True)]),
            'synthetic_examples': len([ex for ex in all_examples if ex.get('source') == 'ms_marco_synthetic']),
            'files_processed': list(existing_files.keys())
        }
        
        summary_path = self.output_dir / 'ms_marco_summary.json'
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        
        logger.info("MS MARCO processing summary:")
        for key, value in summary.items():
            logger.info(f"  {key}: {value}")
        
        return True

def main():
    """Main function to process MS MARCO data"""
    logger.info("=" * 60)
    logger.info("MS MARCO DATA PROCESSOR")
    logger.info("=" * 60)
    
    processor = MSMARCOProcessor()
    
    success = processor.process_msmarco_files()
    
    logger.info("\n" + "=" * 60)
    if success:
        logger.info("✓ MS MARCO PROCESSING COMPLETED!")
        logger.info("Created files:")
        logger.info("  - data/datasets/ms_marco_train.json")
        logger.info("  - data/datasets/ms_marco_val.json")
        logger.info("  - data/datasets/ms_marco_summary.json")
        logger.info("\nNext steps:")
        logger.info("  1. Run: python backend/ml_models/test_msmarco.py")
        logger.info("  2. Or run: python backend/ml_models/model_trainer.py")
    else:
        logger.error("✗ MS MARCO PROCESSING FAILED")
        logger.error("Check the logs above for details")
    
    logger.info("=" * 60)
    
    return success

if __name__ == "__main__":
    main()