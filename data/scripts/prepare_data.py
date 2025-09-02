# data/scripts/prepare_data.py
"""
Data preparation script for Educational QA System
Processes MS MARCO and SQuAD data for training
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
import pickle
import gzip

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataPreparator:
    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        self.datasets_dir = self.data_dir / "datasets"
        self.processed_dir = self.data_dir / "processed"
        self.processed_dir.mkdir(exist_ok=True)
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    def load_squad_data(self, filepath):
        """Load and process SQuAD format data"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        examples = []
        for article in data['data']:
            title = article['title']
            for paragraph in article['paragraphs']:
                context = paragraph['context']
                for qa in paragraph['qas']:
                    example = {
                        'id': qa['id'],
                        'title': title,
                        'question': qa['question'],
                        'context': context,
                        'answers': qa.get('answers', []),
                        'is_impossible': qa.get('is_impossible', False)
                    }
                    examples.append(example)
        return examples

    def load_ms_marco_data(self):
        """Load and process MS MARCO data"""
        queries_file = self.datasets_dir / "ms_marco" / "queries.train.tsv"
        qrels_file = self.datasets_dir / "ms_marco" / "qrels.train.tsv"
        
        if not queries_file.exists() or not qrels_file.exists():
            logger.warning("MS MARCO files not found, skipping MS MARCO processing")
            return []
        
        # Load queries
        queries_df = pd.read_csv(queries_file, sep='\t', header=None, names=['qid', 'query'])
        qrels_df = pd.read_csv(qrels_file, sep='\t', header=None, names=['qid', 'Q0', 'docid', 'relevance'])
        
        # For simplicity, we'll create a sample of MS MARCO data
        # In practice, you'd need the full corpus for complete processing
        examples = []
        sample_size = min(1000, len(queries_df))  # Process first 1000 queries
        
        for idx, row in queries_df.head(sample_size).iterrows():
            qid = row['qid']
            query = row['query']
            
            # Find relevant documents (this is simplified)
            relevant_docs = qrels_df[qrels_df['qid'] == qid]
            
            if not relevant_docs.empty:
                example = {
                    'id': f"msmarco_{qid}",
                    'title': "MS MARCO",
                    'question': query,
                    'context': f"This is a sample context for query: {query}",  # Placeholder
                    'answers': [{'text': 'Sample answer', 'answer_start': 0}],
                    'is_impossible': False
                }
                examples.append(example)
        
        logger.info(f"Processed {len(examples)} MS MARCO examples")
        return examples

    def preprocess_examples(self, examples, max_length=512):
        """Preprocess examples for BERT training"""
        processed_examples = []
        
        for example in examples:
            # Tokenize question and context
            encoding = self.tokenizer.encode_plus(
                example['question'],
                example['context'],
                max_length=max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            # Find answer positions (simplified)
            start_positions = []
            end_positions = []
            
            if example['answers'] and not example['is_impossible']:
                for answer in example['answers']:
                    answer_text = answer['text']
                    answer_start = answer['answer_start']
                    
                    # Convert character positions to token positions
                    # This is a simplified version - you'd want more robust alignment
                    token_start = len(self.tokenizer.tokenize(example['context'][:answer_start]))
                    token_end = token_start + len(self.tokenizer.tokenize(answer_text))
                    
                    start_positions.append(token_start)
                    end_positions.append(token_end)
            else:
                # Impossible question
                start_positions.append(0)
                end_positions.append(0)
            
            processed_example = {
                'id': example['id'],
                'input_ids': encoding['input_ids'].squeeze().tolist(),
                'attention_mask': encoding['attention_mask'].squeeze().tolist(),
                'token_type_ids': encoding['token_type_ids'].squeeze().tolist(),
                'start_positions': start_positions[0] if start_positions else 0,
                'end_positions': end_positions[0] if end_positions else 0,
                'is_impossible': example['is_impossible'],
                'question': example['question'],
                'context': example['context'],
                'answers': example['answers']
            }
            
            processed_examples.append(processed_example)
        
        return processed_examples

    def create_training_data(self):
        """Create training and validation datasets"""
        logger.info("Creating training data...")
        
        all_examples = []
        
        # Load SQuAD data
        squad_files = [
            self.datasets_dir / "squad_2.0" / "train-v2.0.json",
            self.datasets_dir / "squad_2.0" / "dev-v2.0.json",
            self.datasets_dir / "squad_sample.json",
            self.datasets_dir / "educational_qa.json"
        ]
        
        for squad_file in squad_files:
            if squad_file.exists():
                logger.info(f"Loading {squad_file.name}...")
                examples = self.load_squad_data(squad_file)
                all_examples.extend(examples)
                logger.info(f"Loaded {len(examples)} examples from {squad_file.name}")
        
        # Load MS MARCO data
        ms_marco_examples = self.load_ms_marco_data()
        all_examples.extend(ms_marco_examples)
        
        logger.info(f"Total examples loaded: {len(all_examples)}")
        
        # Split data
        train_examples, val_examples = train_test_split(
            all_examples, 
            test_size=0.1, 
            random_state=42
        )
        
        logger.info(f"Train examples: {len(train_examples)}")
        logger.info(f"Validation examples: {len(val_examples)}")
        
        # Preprocess examples
        logger.info("Preprocessing training data...")
        train_processed = self.preprocess_examples(train_examples)
        
        logger.info("Preprocessing validation data...")
        val_processed = self.preprocess_examples(val_examples)
        
        # Save processed data
        train_file = self.processed_dir / "train.json"
        val_file = self.processed_dir / "validation.json"
        
        with open(train_file, 'w') as f:
            json.dump(train_processed, f, indent=2)
        
        with open(val_file, 'w') as f:
            json.dump(val_processed, f, indent=2)
        
        logger.info(f"Saved training data to: {train_file}")
        logger.info(f"Saved validation data to: {val_file}")
        
        return train_processed, val_processed

    def create_embeddings_placeholder(self):
        """Create placeholder embedding files"""
        logger.info("Creating placeholder embedding files...")
        
        # Create dummy embeddings for demonstration
        dummy_doc_embeddings = {
            'document_1': np.random.randn(768).tolist(),  # BERT embedding size
            'document_2': np.random.randn(768).tolist(),
            'document_3': np.random.randn(768).tolist()
        }
        
        dummy_sentence_embeddings = {
            'sentence_1': np.random.randn(768).tolist(),
            'sentence_2': np.random.randn(768).tolist(),
            'sentence_3': np.random.randn(768).tolist()
        }
        
        # Save embeddings
        doc_emb_file = self.data_dir / "embeddings" / "document_embeddings.pkl"
        sent_emb_file = self.data_dir / "embeddings" / "sentence_embeddings.pkl"
        
        with open(doc_emb_file, 'wb') as f:
            pickle.dump(dummy_doc_embeddings, f)
        
        with open(sent_emb_file, 'wb') as f:
            pickle.dump(dummy_sentence_embeddings, f)
        
        logger.info(f"Created placeholder embeddings: {doc_emb_file}")
        logger.info(f"Created placeholder embeddings: {sent_emb_file}")

    def generate_statistics(self):
        """Generate dataset statistics"""
        train_file = self.processed_dir / "train.json"
        val_file = self.processed_dir / "validation.json"
        
        if not train_file.exists():
            logger.warning("Training data not found")
            return
        
        with open(train_file, 'r') as f:
            train_data = json.load(f)
        
        with open(val_file, 'r') as f:
            val_data = json.load(f)
        
        stats = {
            'total_train_examples': len(train_data),
            'total_val_examples': len(val_data),
            'impossible_train': sum(1 for ex in train_data if ex['is_impossible']),
            'impossible_val': sum(1 for ex in val_data if ex['is_impossible']),
            'avg_question_length': np.mean([len(ex['question'].split()) for ex in train_data]),
            'avg_context_length': np.mean([len(ex['context'].split()) for ex in train_data])
        }
        
        stats_file = self.processed_dir / "dataset_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info("Dataset Statistics:")
        for key, value in stats.items():
            logger.info(f"  {key}: {value}")
        
        return stats

    def prepare_all(self):
        """Run complete data preparation pipeline"""
        logger.info("Starting complete data preparation...")
        
        try:
            # Create training data
            self.create_training_data()
            
            # Create placeholder embeddings
            self.create_embeddings_placeholder()
            
            # Generate statistics
            self.generate_statistics()
            
            logger.info("✅ Data preparation completed successfully!")
            
        except Exception as e:
            logger.error(f"Error during data preparation: {e}")
            raise


if __name__ == "__main__":
    preparator = DataPreparator()
    preparator.prepare_all()