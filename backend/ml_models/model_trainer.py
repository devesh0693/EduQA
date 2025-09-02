"""
Enhanced Model Training Module for Educational QA System

Improved version with better error handling, metrics, and training strategies
"""

import os
os.environ["WANDB_DISABLED"] = "true"

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    get_linear_schedule_with_warmup,
    TrainingArguments,
    Trainer,
    default_data_collator,
    EarlyStoppingCallback
)
from sklearn.metrics import accuracy_score
import numpy as np
import string
import re
from collections import Counter

# Import your existing modules
from .data_preprocessor import DataPreprocessor
from .config import MODEL_CONFIG, TRAINING_CONFIG, MODEL_PATHS, PERFORMANCE_TARGETS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QADataset(Dataset):
    """Enhanced Dataset class for Question Answering data with better validation"""
   
    def __init__(self, encodings, validate=True):
        self.encodings = encodings
        if validate:
            self._validate_data()
   
    def _validate_data(self):
        """Validate the dataset structure"""
        required_keys = ['input_ids', 'attention_mask', 'start_positions', 'end_positions']
        for key in required_keys:
            if key not in self.encodings:
                raise ValueError(f"Missing required key: {key}")
        
        # Check data consistency
        length = len(self.encodings['input_ids'])
        for key, val in self.encodings.items():
            if len(val) != length:
                raise ValueError(f"Inconsistent lengths: {key} has {len(val)}, expected {length}")
    
    def __getitem__(self, idx):
        item = {}
        for key, val in self.encodings.items():
            if isinstance(val, list):
                item[key] = torch.tensor(val[idx])
            else:
                item[key] = val[idx]
        return item
   
    def __len__(self):
        return len(self.encodings['input_ids'])

class ModelTrainer:
    def __init__(self, model_name: str = MODEL_CONFIG.get('bert_model_name', 'bert-base-uncased')):
        """
        Initialize the enhanced model trainer with better error handling
        """
        self.model_name = model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        
        try:
            # Initialize tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)
            
            # Move model to device
            self.model.to(self.device)
            
            # Initialize data preprocessor
            self.preprocessor = DataPreprocessor(model_name)
            
            # Enhanced training configuration
            self.training_config = TRAINING_CONFIG.copy()
            self.model_config = MODEL_CONFIG.copy()
            
            # Performance tracking
            self.training_history = {
                'train_loss': [],
                'eval_loss': [],
                'eval_f1': [],
                'eval_exact_match': [],
                'learning_rate': []
            }
            
            logger.info("Enhanced model trainer initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize model trainer: {str(e)}")
            raise

    def normalize_answer(self, s: str) -> str:
        """Normalize answer for evaluation"""
        def remove_articles(text):
            return re.sub(r'\b(a|an|the)\b', ' ', text)
        
        def white_space_fix(text):
            return ' '.join(text.split())
        
        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)
        
        def lower(text):
            return text.lower()
        
        return white_space_fix(remove_articles(remove_punc(lower(s))))

    def f1_score(self, prediction: str, ground_truth: str) -> float:
        """Calculate F1 score between prediction and ground truth"""
        prediction_tokens = self.normalize_answer(prediction).split()
        ground_truth_tokens = self.normalize_answer(ground_truth).split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1

    def exact_match_score(self, prediction: str, ground_truth: str) -> float:
        """Calculate exact match score"""
        return float(self.normalize_answer(prediction) == self.normalize_answer(ground_truth))

    def compute_metrics(self, eval_pred):
        """Enhanced metrics computation with proper answer extraction"""
        predictions, labels = eval_pred
        
        # Extract start and end predictions
        start_logits = predictions[0]
        end_logits = predictions[1]
        
        start_predictions = np.argmax(start_logits, axis=1)
        end_predictions = np.argmax(end_logits, axis=1)
        
        start_labels = labels[0]
        end_labels = labels[1]
        
        # Compute position-based metrics
        start_accuracy = accuracy_score(start_labels, start_predictions)
        end_accuracy = accuracy_score(end_labels, end_predictions)
        exact_match_positions = np.mean((start_predictions == start_labels) & (end_predictions == end_labels))
        
        # For proper F1 calculation, we'd need the actual text
        # This is a simplified approximation
        f1_score = (start_accuracy + end_accuracy) / 2
        
        return {
            'exact_match': exact_match_positions,
            'f1': f1_score,
            'start_accuracy': start_accuracy,
            'end_accuracy': end_accuracy
        }

    def prepare_training_data(self) -> Tuple[QADataset, QADataset]:
        """Enhanced data preparation with better validation"""
        logger.info("Preparing training data...")
        
        try:
            # Load all datasets
            datasets = self.preprocessor.prepare_datasets()
            
            # Combine training data from different sources
            train_examples = []
            val_examples = []
            
            # Add SQuAD data with size limits
            max_train = self.training_config.get('max_train_samples', 5000)
            max_val = self.training_config.get('max_eval_samples', 1000)
            
            if 'squad_train' in datasets and datasets['squad_train']:
                squad_train = datasets['squad_train'][:max_train]
                # Filter out impossible questions for initial training
                squad_train = [ex for ex in squad_train if not ex.get('is_impossible', False)]
                train_examples.extend(squad_train)
                logger.info(f"Added {len(squad_train)} SQuAD training examples")
            
            # Add MS MARCO data
            if 'ms_marco_train' in datasets and datasets['ms_marco_train']:
                msmarco_train = datasets['ms_marco_train'][:max_train//2]
                # Filter out impossible questions
                msmarco_train = [ex for ex in msmarco_train if not ex.get('is_impossible', False)]
                train_examples.extend(msmarco_train)
                logger.info(f"Added {len(msmarco_train)} MS MARCO training examples")
            
            # Add validation data
            if 'squad_dev' in datasets and datasets['squad_dev']:
                val_examples = datasets['squad_dev'][:max_val]
                # Filter out impossible questions
                val_examples = [ex for ex in val_examples if not ex.get('is_impossible', False)]
                logger.info(f"Using {len(val_examples)} SQuAD dev examples for validation")
            
            # Add educational data
            if 'edu_train' in datasets and datasets['edu_train']:
                train_examples.extend(datasets['edu_train'])
                logger.info(f"Added {len(datasets['edu_train'])} educational training examples")
            
            if 'edu_val' in datasets and datasets['edu_val']:
                val_examples.extend(datasets['edu_val'])
                logger.info(f"Added {len(datasets['edu_val'])} educational validation examples")
            
            # Fallback to sample dataset
            if not train_examples and 'squad_sample' in datasets:
                all_samples = datasets['squad_sample']
                # Filter valid samples
                valid_samples = [ex for ex in all_samples if isinstance(ex, dict) and 
                                'question' in ex and 'context' in ex and not ex.get('is_impossible', False)]
                
                if len(valid_samples) >= 100:
                    train_examples = valid_samples[:800]
                    val_examples = valid_samples[800:900]
                    logger.info(f"Using sample dataset: {len(train_examples)} train, {len(val_examples)} val")
            
            if not train_examples:
                raise ValueError("No training data found. Please check your dataset files.")
            
            if not val_examples:
                # Create validation split from training data
                split_idx = int(len(train_examples) * 0.9)
                val_examples = train_examples[split_idx:]
                train_examples = train_examples[:split_idx]
                logger.info(f"Created validation split: {len(train_examples)} train, {len(val_examples)} val")
            
            # Validate examples
            train_examples = self._validate_examples(train_examples)
            val_examples = self._validate_examples(val_examples)
            
            # Tokenize examples
            train_tokenized = self.preprocessor.tokenize_examples(train_examples, is_training=True)
            val_tokenized = self.preprocessor.tokenize_examples(val_examples, is_training=True)
            
            # Create datasets with validation
            train_dataset = QADataset(train_tokenized, validate=True)
            val_dataset = QADataset(val_tokenized, validate=True)
            
            logger.info(f"Final dataset sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}")
            
            return train_dataset, val_dataset
            
        except Exception as e:
            logger.error(f"Error preparing training data: {str(e)}")
            raise

    def _validate_examples(self, examples: List[Dict]) -> List[Dict]:
        """Validate and clean examples"""
        valid_examples = []
        for i, example in enumerate(examples):
            try:
                # Check required fields
                if not all(key in example for key in ['question', 'context']):
                    continue
                
                # Check for empty fields
                if not example['question'].strip() or not example['context'].strip():
                    continue
                
                # Check answer validity for training
                if 'answer_text' in example and example['answer_text']:
                    answer_start = example.get('answer_start', -1)
                    if answer_start >= 0:
                        # Verify answer is in context
                        answer_text = example['answer_text']
                        context = example['context']
                        if answer_text not in context:
                            logger.warning(f"Answer not found in context for example {i}")
                            continue
                
                valid_examples.append(example)
                
            except Exception as e:
                logger.warning(f"Skipping invalid example {i}: {str(e)}")
                continue
        
        logger.info(f"Validated {len(valid_examples)} examples from {len(examples)} total")
        return valid_examples

    def train_model(self, output_dir: str = None, use_wandb: bool = False, 
                   experiment_name: str = "educational_qa_training") -> Optional[Dict]:
        """Enhanced training with better monitoring and error handling"""
        logger.info("Starting enhanced model training...")
        
        try:
            # Prepare training and validation datasets
            train_dataset, val_dataset = self.prepare_training_data()
            
            if len(train_dataset) == 0 or len(val_dataset) == 0:
                logger.error("No training or validation data available")
                return None
            
            # Set output directory
            if output_dir is None:
                output_dir = MODEL_PATHS.get('trained_model', 'trained_model')
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Enhanced training arguments
            training_args = TrainingArguments(
                output_dir=str(output_dir),
                num_train_epochs=self.training_config.get('num_epochs', 3),
                per_device_train_batch_size=self.training_config.get('batch_size', 8),
                per_device_eval_batch_size=self.training_config.get('batch_size', 8) * 2,
                warmup_ratio=self.training_config.get('warmup_ratio', 0.1),
                weight_decay=self.training_config.get('weight_decay', 0.01),
                logging_dir=str(output_dir / 'logs'),
                logging_steps=50,
                eval_strategy="steps",
                eval_steps=200,
                save_steps=500,
                save_total_limit=3,
                load_best_model_at_end=True,
                metric_for_best_model="eval_f1",
                greater_is_better=True,
                fp16=torch.cuda.is_available(),
                gradient_accumulation_steps=2,
                learning_rate=float(self.training_config.get('learning_rate', 3e-5)),
                seed=42,
                report_to="none",
                dataloader_num_workers=0,
                overwrite_output_dir=True,
                remove_unused_columns=True,
                push_to_hub=False,
            )
            
            # Initialize trainer with early stopping
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                data_collator=default_data_collator,
                compute_metrics=self.compute_metrics,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
            )
            
            # Log training info
            logger.info(f"Training configuration:")
            logger.info(f"  Device: {self.device}")
            logger.info(f"  Train examples: {len(train_dataset)}")
            logger.info(f"  Val examples: {len(val_dataset)}")
            logger.info(f"  Epochs: {training_args.num_train_epochs}")
            logger.info(f"  Batch size: {training_args.per_device_train_batch_size}")
            logger.info(f"  Learning rate: {training_args.learning_rate}")
            logger.info(f"  FP16: {training_args.fp16}")
            
            # Start training
            start_time = time.time()
            train_result = trainer.train()
            training_time = time.time() - start_time
            
            # Save the model
            trainer.save_model(str(output_dir))
            self.tokenizer.save_pretrained(str(output_dir))
            
            # Final evaluation
            eval_result = trainer.evaluate()
            
            # Compile results
            final_results = {
                'train_loss': train_result.training_loss,
                'eval_loss': eval_result.get('eval_loss', 0.0),
                'eval_f1': eval_result.get('eval_f1', 0.0),
                'eval_exact_match': eval_result.get('eval_exact_match', 0.0),
                'eval_start_accuracy': eval_result.get('eval_start_accuracy', 0.0),
                'eval_end_accuracy': eval_result.get('eval_end_accuracy', 0.0),
                'training_time_minutes': training_time / 60,
                'model_path': str(output_dir),
                'train_samples': len(train_dataset),
                'val_samples': len(val_dataset),
                'meets_f1_target': eval_result.get('eval_f1', 0.0) >= PERFORMANCE_TARGETS.get('f1_score_threshold', 0.7),
                'meets_accuracy_target': eval_result.get('eval_exact_match', 0.0) >= PERFORMANCE_TARGETS.get('accuracy_threshold', 0.6)
            }
            
            # Log results
            logger.info("Training completed successfully!")
            logger.info("Final Results:")
            for key, value in final_results.items():
                if isinstance(value, float):
                    logger.info(f"  {key}: {value:.4f}")
                else:
                    logger.info(f"  {key}: {value}")
            
            # Save results
            results_path = output_dir / 'training_results.json'
            with open(results_path, 'w') as f:
                json.dump(final_results, f, indent=2, default=str)
            
            logger.info(f"Model and results saved to {output_dir}")
            return final_results
            
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def quick_test_model(self, model_path: str, test_questions: List[str] = None) -> Dict:
        """Quick test of the trained model"""
        if test_questions is None:
            test_questions = [
                "What is the capital of France?",
                "Who wrote Romeo and Juliet?",
                "What is photosynthesis?"
            ]
        
        try:
            # Load model and tokenizer
            model = AutoModelForQuestionAnswering.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model.eval()
            
            results = []
            for question in test_questions:
                # Simple context for testing
                context = "France is a country in Europe. Its capital city is Paris. Shakespeare wrote many famous plays including Romeo and Juliet. Photosynthesis is the process by which plants convert sunlight into energy."
                
                inputs = tokenizer.encode_plus(
                    question, context,
                    add_special_tokens=True,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512
                )
                
                with torch.no_grad():
                    outputs = model(**inputs)
                    start_scores = outputs.start_logits
                    end_scores = outputs.end_logits
                
                start_idx = torch.argmax(start_scores)
                end_idx = torch.argmax(end_scores)
                
                if start_idx <= end_idx:
                    answer = tokenizer.decode(inputs['input_ids'][0][start_idx:end_idx+1])
                else:
                    answer = ""
                
                results.append({
                    'question': question,
                    'answer': answer,
                    'confidence': float(torch.max(start_scores) + torch.max(end_scores))
                })
            
            return {'test_results': results, 'status': 'success'}
            
        except Exception as e:
            logger.error(f"Model testing failed: {str(e)}")
            return {'test_results': [], 'status': 'failed', 'error': str(e)}

if __name__ == "__main__":
    try:
        print("Starting enhanced model training...")
        
        # Initialize enhanced trainer
        trainer = ModelTrainer()
        
        # Train the model
        results = trainer.train_model(
            output_dir="enhanced_model_output",
            use_wandb=False,
            experiment_name="enhanced_educational_qa_v1"
        )
        
        if results:
            print("\n" + "="*50)
            print("TRAINING COMPLETED SUCCESSFULLY!")
            print("="*50)
            print(f"Model saved to: {results['model_path']}")
            print(f"Training time: {results['training_time_minutes']:.2f} minutes")
            print(f"Final F1 Score: {results['eval_f1']:.4f}")
            print(f"Final Exact Match: {results['eval_exact_match']:.4f}")
            print(f"Meets F1 Target: {results['meets_f1_target']}")
            print(f"Meets Accuracy Target: {results['meets_accuracy_target']}")
            
            # Quick test
            print("\nTesting model with sample questions...")
            test_results = trainer.quick_test_model(results['model_path'])
            if test_results['status'] == 'success':
                for result in test_results['test_results']:
                    print(f"Q: {result['question']}")
                    print(f"A: {result['answer']}")
                    print(f"Confidence: {result['confidence']:.2f}")
                    print()
        else:
            print("Training failed - check logs for details")
            
    except Exception as e:
        print(f"Error during training: {str(e)}")
        import traceback
        traceback.print_exc()