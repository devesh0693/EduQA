"""
Data preprocessing module for Educational QA System
Handles SQuAD 2.0, MS MARCO, and custom educational datasets
"""

import json
import logging
import re
import os
import sqlite3
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
from transformers import AutoTokenizer
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self, model_name: str = 'bert-base-uncased', db_path: str = 'db.sqlite3'):
        """Initialize the data preprocessor with database connection"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        except Exception as e:
            logger.error(f"Error loading tokenizer: {e}")
            # Fallback to basic tokenization if needed
            self.tokenizer = None
            
        # Configuration
        self.max_seq_length = 512
        self.doc_stride = 128
        self.max_query_length = 64
        self.max_answer_length = 30
        
        # Database connection
        self.db_path = db_path
        self.init_database()
        
        # Dataset paths
        self.data_dir = Path('data/datasets')
        self.dataset_paths = {
            'squad_train': self.data_dir / 'squad_2.0' / 'train-v2.0.json',
            'squad_dev': self.data_dir / 'squad_2.0' / 'dev-v2.0.json',
            'squad_sample': self.data_dir / 'squad_sample.json',
            'educational_qa': self.data_dir / 'educational_qa.json',
            'educational_train': self.data_dir / 'educational_train.json',
            'educational_val': self.data_dir / 'educational_val.json',
            'ms_marco_train': self.data_dir / 'ms_marco' / 'train.json',
            'ms_marco_dev': self.data_dir / 'ms_marco' / 'dev.json',
            'ms_marco_summary': self.data_dir / 'ms_marco_summary.json'
        }
    
    def init_database(self):
        """Initialize database tables for storing processed data"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create table for QA pairs
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS qa_pairs (
                    id TEXT PRIMARY KEY,
                    question TEXT NOT NULL,
                    context TEXT NOT NULL,
                    answer_text TEXT,
                    start_position INTEGER,
                    end_position INTEGER,
                    is_impossible BOOLEAN DEFAULT 0,
                    dataset_source TEXT,
                    title TEXT,
                    relevance_score REAL DEFAULT 1.0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create index for faster searches
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_question ON qa_pairs(question)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_context ON qa_pairs(context)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_source ON qa_pairs(dataset_source)')
            
            conn.commit()
            conn.close()
            logger.info("Database initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text for better processing"""
        if not text:
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Handle common educational text patterns
        text = re.sub(r'\n+', ' ', text)  # Replace newlines with spaces
        text = re.sub(r'\t+', ' ', text)  # Replace tabs with spaces
        
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s.,!?;:()\-\'\""]', ' ', text)
        
        return text.strip()
    
    def load_squad_dataset(self, file_path: Path) -> List[Dict]:
        """Load and preprocess SQuAD dataset in various formats including JSONL"""
        logger.info(f"Loading SQuAD dataset from {file_path}")
        
        if not file_path.exists():
            logger.warning(f"File not found: {file_path}")
            return []
        
        examples = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                # Try JSONL format first
                content = f.read()
                f.seek(0)
                
                # Check if it's JSONL (multiple JSON objects)
                lines = content.strip().split('\n')
                if len(lines) > 1 and all(line.strip().startswith('{') for line in lines[:5]):
                    # JSONL format
                    for line_num, line in enumerate(lines):
                        if not line.strip():
                            continue
                        try:
                            item = json.loads(line.strip())
                            context = self.clean_text(item.get('context', ''))
                            question = self.clean_text(item.get('question', ''))
                            qas_id = item.get('id', f'sample_{line_num}')
                            
                            answers = item.get('answers', {})
                            answer_texts = answers.get('text', [])
                            answer_starts = answers.get('answer_start', [])
                            
                            if not answer_texts:  # Handle unanswerable questions
                                examples.append({
                                    'id': qas_id,
                                    'title': item.get('title', ''),
                                    'question': question,
                                    'context': context,
                                    'answer_text': '',
                                    'start_char': -1,
                                    'end_char': -1,
                                    'is_impossible': True
                                })
                            else:
                                for i, (text, start) in enumerate(zip(answer_texts, answer_starts)):
                                    answer_text = self.clean_text(text)
                                    start_char = int(start)
                                    end_char = start_char + len(answer_text)
                                    
                                    examples.append({
                                        'id': f"{qas_id}_{i}" if i > 0 else qas_id,
                                        'title': item.get('title', ''),
                                        'question': question,
                                        'context': context,
                                        'answer_text': answer_text,
                                        'start_char': start_char,
                                        'end_char': end_char,
                                        'is_impossible': False
                                    })
                        except json.JSONDecodeError:
                            logger.warning(f"Skipping invalid JSON line {line_num + 1}")
                            continue
                else:
                    # Standard JSON format
                    data = json.loads(content)
                    
                    if isinstance(data, dict) and 'data' in data:  # Full SQuAD format
                        for article in data['data']:
                            for paragraph in article['paragraphs']:
                                context = self.clean_text(paragraph['context'])
                                for qa in paragraph['qas']:
                                    question = self.clean_text(qa['question'])
                                    qas_id = qa['id']
                                    is_impossible = qa.get('is_impossible', False)
                                    
                                    if is_impossible or not qa.get('answers'):
                                        examples.append({
                                            'id': qas_id,
                                            'title': article.get('title', ''),
                                            'question': question,
                                            'context': context,
                                            'answer_text': '',
                                            'start_char': -1,
                                            'end_char': -1,
                                            'is_impossible': True
                                        })
                                    else:
                                        for i, answer in enumerate(qa['answers']):
                                            answer_text = self.clean_text(answer['text'])
                                            start_char = answer['answer_start']
                                            end_char = start_char + len(answer_text)
                                            
                                            examples.append({
                                                'id': f"{qas_id}_{i}" if i > 0 else qas_id,
                                                'title': article.get('title', ''),
                                                'question': question,
                                                'context': context,
                                                'answer_text': answer_text,
                                                'start_char': start_char,
                                                'end_char': end_char,
                                                'is_impossible': False
                                            })
                    
                    elif isinstance(data, list):  # Simplified format
                        for i, item in enumerate(data):
                            context = self.clean_text(item.get('context', ''))
                            question = self.clean_text(item.get('question', ''))
                            qas_id = item.get('id', f'sample_{i}')
                            
                            answers = item.get('answers', {})
                            answer_texts = answers.get('text', []) if isinstance(answers, dict) else []
                            answer_starts = answers.get('answer_start', []) if isinstance(answers, dict) else []
                            
                            if not answer_texts:
                                examples.append({
                                    'id': qas_id,
                                    'title': item.get('title', ''),
                                    'question': question,
                                    'context': context,
                                    'answer_text': '',
                                    'start_char': -1,
                                    'end_char': -1,
                                    'is_impossible': True
                                })
                            else:
                                for j, (answer_text, start_char) in enumerate(zip(answer_texts, answer_starts)):
                                    answer_text = self.clean_text(answer_text)
                                    start_char = int(start_char)
                                    end_char = start_char + len(answer_text)
                                    
                                    examples.append({
                                        'id': f"{qas_id}_{j}" if j > 0 else qas_id,
                                        'title': item.get('title', ''),
                                        'question': question,
                                        'context': context,
                                        'answer_text': answer_text,
                                        'start_char': start_char,
                                        'end_char': end_char,
                                        'is_impossible': False
                                    })
                                    
        except Exception as e:
            logger.error(f"Error loading SQuAD dataset: {e}")
            return []
        
        logger.info(f"Loaded {len(examples)} examples from SQuAD dataset")
        return examples
    
    def load_ms_marco_dataset(self, file_path: Path) -> List[Dict]:
        """Load and preprocess MS MARCO dataset"""
        logger.info(f"Loading MS MARCO dataset from {file_path}")
        
        if not file_path.exists():
            logger.warning(f"File not found: {file_path}")
            return []
        
        examples = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f):
                    if not line.strip():
                        continue
                    try:
                        data = json.loads(line.strip())
                        
                        query = self.clean_text(data.get('query', ''))
                        query_id = data.get('query_id', f'msmarco_{line_num}')
                        
                        passages = data.get('passages', [])
                        for passage in passages:
                            if passage.get('is_selected', 0) == 1:
                                context = self.clean_text(passage.get('passage_text', ''))
                                
                                examples.append({
                                    'id': query_id,
                                    'title': '',
                                    'question': query,
                                    'context': context,
                                    'answer_text': '',  # MS MARCO format
                                    'start_char': -1,
                                    'end_char': -1,
                                    'is_impossible': False
                                })
                    except json.JSONDecodeError:
                        logger.warning(f"Skipping invalid JSON line {line_num + 1}")
                        continue
        except Exception as e:
            logger.error(f"Error loading MS MARCO dataset: {e}")
            return []
        
        logger.info(f"Loaded {len(examples)} examples from MS MARCO dataset")
        return examples

    def load_educational_dataset(self, file_path: Path) -> List[Dict]:
        """Load and preprocess custom educational dataset"""
        logger.info(f"Loading educational dataset from {file_path}")
        
        if not file_path.exists():
            logger.warning(f"File not found: {file_path}")
            return []
        
        # Try MS MARCO format first (JSONL)
        try:
            return self.load_ms_marco_dataset(file_path)
        except:
            pass
            
        # Try SQuAD format
        try:
            return self.load_squad_dataset(file_path)
        except Exception as e:
            logger.error(f"Error loading educational dataset: {e}")
            return []
    
    def save_to_database(self, examples: List[Dict], dataset_source: str):
        """Save processed examples to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for example in examples:
                cursor.execute('''
                    INSERT OR REPLACE INTO qa_pairs 
                    (id, question, context, answer_text, start_position, end_position, 
                     is_impossible, dataset_source, title, relevance_score)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    example['id'],
                    example['question'],
                    example['context'],
                    example['answer_text'],
                    example['start_char'],
                    example['end_char'],
                    example['is_impossible'],
                    dataset_source,
                    example.get('title', ''),
                    1.0  # Default relevance score
                ))
            
            conn.commit()
            conn.close()
            logger.info(f"Saved {len(examples)} examples to database from {dataset_source}")
        except Exception as e:
            logger.error(f"Error saving to database: {e}")
    
    def tokenize_examples(self, examples: List[Dict], is_training: bool = True) -> Dict:
        """Tokenize examples for BERT model"""
        if not self.tokenizer:
            logger.error("Tokenizer not available")
            return {}
            
        logger.info(f"Tokenizing {len(examples)} examples...")
        
        tokenized_examples = {
            'input_ids': [],
            'attention_mask': [],
            'token_type_ids': [],
        }
        
        if is_training:
            tokenized_examples.update({
                'start_positions': [],
                'end_positions': [],
            })
        
        for example in examples:
            try:
                # Tokenize question and context
                tokenized = self.tokenizer(
                    example['question'],
                    example['context'],
                    truncation='only_second',
                    max_length=self.max_seq_length,
                    stride=self.doc_stride,
                    return_overflowing_tokens=True,
                    return_offsets_mapping=True,
                    padding='max_length'
                )
                
                # Handle overflowing tokens (for long contexts)
                for i in range(len(tokenized['input_ids'])):
                    input_ids = tokenized['input_ids'][i]
                    attention_mask = tokenized['attention_mask'][i]
                    token_type_ids = tokenized['token_type_ids'][i]
                    
                    tokenized_examples['input_ids'].append(input_ids)
                    tokenized_examples['attention_mask'].append(attention_mask)
                    tokenized_examples['token_type_ids'].append(token_type_ids)
                    
                    if is_training:
                        # Find answer positions in tokenized sequence
                        if example['is_impossible'] or example['start_char'] == -1:
                            start_pos = 0  # CLS token position for impossible answers
                            end_pos = 0
                        else:
                            # Map character positions to token positions
                            offset_mapping = tokenized['offset_mapping'][i]
                            start_pos = 0
                            end_pos = 0
                            
                            for idx, (start_off, end_off) in enumerate(offset_mapping):
                                if start_off <= example['start_char'] < end_off:
                                    start_pos = idx
                                if start_off < example['end_char'] <= end_off:
                                    end_pos = idx
                                    break
                        
                        tokenized_examples['start_positions'].append(start_pos)
                        tokenized_examples['end_positions'].append(end_pos)
            except Exception as e:
                logger.error(f"Error tokenizing example {example['id']}: {e}")
                continue
        
        logger.info(f"Tokenization complete. Created {len(tokenized_examples['input_ids'])} tokenized examples")
        return tokenized_examples
    
    def create_dataset_splits(self, examples: List[Dict], train_ratio: float = 0.8) -> Tuple[List[Dict], List[Dict]]:
        """Split dataset into train and validation sets"""
        split_idx = int(len(examples) * train_ratio)
        train_examples = examples[:split_idx]
        val_examples = examples[split_idx:]
        
        logger.info(f"Created train split: {len(train_examples)} examples")
        logger.info(f"Created validation split: {len(val_examples)} examples")
        
        return train_examples, val_examples
    
    def prepare_datasets(self) -> Dict[str, List[Dict]]:
        """Prepare all datasets for training and save to database"""
        datasets = {}
        
        # Load SQuAD 2.0
        for name, path in [('squad_train', self.dataset_paths['squad_train']),
                          ('squad_dev', self.dataset_paths['squad_dev']),
                          ('squad_sample', self.dataset_paths['squad_sample'])]:
            if path.exists():
                examples = self.load_squad_dataset(path)
                datasets[name] = examples
                self.save_to_database(examples, name)
        
        # Load MS MARCO
        for name, path in [('ms_marco_train', self.dataset_paths['ms_marco_train']),
                          ('ms_marco_dev', self.dataset_paths['ms_marco_dev'])]:
            if path.exists():
                examples = self.load_ms_marco_dataset(path)
                datasets[name] = examples
                self.save_to_database(examples, name)
        
        # Load educational datasets
        for name in ['educational_qa', 'educational_train', 'educational_val']:
            path = self.dataset_paths[name]
            if path.exists():
                examples = self.load_educational_dataset(path)
                datasets[name] = examples
                self.save_to_database(examples, name)
        
        # If educational_qa exists but train/val don't, create splits
        if 'educational_qa' in datasets and 'educational_train' not in datasets:
            edu_train, edu_val = self.create_dataset_splits(datasets['educational_qa'])
            datasets['educational_train'] = edu_train
            datasets['educational_val'] = edu_val
            self.save_to_database(edu_train, 'educational_train')
            self.save_to_database(edu_val, 'educational_val')
        
        return datasets
    
    def search_qa_pairs(self, query: str, limit: int = 10) -> List[Dict]:
        """Search QA pairs from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Simple text search (can be enhanced with embeddings later)
            cursor.execute('''
                SELECT id, question, context, answer_text, dataset_source, title, relevance_score
                FROM qa_pairs 
                WHERE question LIKE ? OR context LIKE ?
                ORDER BY relevance_score DESC
                LIMIT ?
            ''', (f'%{query}%', f'%{query}%', limit))
            
            results = []
            for row in cursor.fetchall():
                results.append({
                    'id': row[0],
                    'question': row[1],
                    'context': row[2],
                    'answer': row[3],
                    'source': row[4],
                    'title': row[5],
                    'relevance': row[6]
                })
            
            conn.close()
            return results
        except Exception as e:
            logger.error(f"Error searching QA pairs: {e}")
            return []
    
    def get_dataset_stats(self) -> Dict[str, Any]:
        """Get statistics about loaded datasets"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT dataset_source, COUNT(*) as count
                FROM qa_pairs
                GROUP BY dataset_source
            ''')
            
            stats = {}
            total = 0
            for row in cursor.fetchall():
                source, count = row
                stats[source] = count
                total += count
            
            stats['total'] = total
            conn.close()
            
            return stats
        except Exception as e:
            logger.error(f"Error getting dataset stats: {e}")
            return {}
    
    def save_processed_dataset(self, examples: List[Dict], output_path: Path):
        """Save processed dataset to file"""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(examples, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Saved {len(examples)} processed examples to {output_path}")
        except Exception as e:
            logger.error(f"Error saving dataset to {output_path}: {e}")

def main():
    """Main function to process all datasets"""
    preprocessor = DataPreprocessor()
    
    # Prepare and load all datasets
    datasets = preprocessor.prepare_datasets()
    
    # Print statistics
    stats = preprocessor.get_dataset_stats()
    print("\nDataset Statistics:")
    for source, count in stats.items():
        print(f"{source}: {count} examples")
    
    # Test search functionality
    print("\nTesting search functionality...")
    results = preprocessor.search_qa_pairs("machine learning", limit=3)
    for i, result in enumerate(results, 1):
        print(f"{i}. {result['question'][:100]}...")
        print(f"   Answer: {result['answer'][:100]}...")
        print(f"   Source: {result['source']}")
        print()

if __name__ == "__main__":
    main()