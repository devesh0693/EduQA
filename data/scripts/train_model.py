# data/scripts/train_model.py
"""
Simple training script for educational QA model
"""

import json
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForQuestionAnswering,
    TrainingArguments,
    Trainer
)
from torch.utils.data import Dataset
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QADataset(Dataset):
    def __init__(self, data_path):
        with open(data_path, 'r') as f:
            self.data = json.load(f)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            'input_ids': torch.tensor(item['input_ids']),
            'attention_mask': torch.tensor(item['attention_mask']),
            'start_positions': torch.tensor(item['start_positions']),
            'end_positions': torch.tensor(item['end_positions'])
        }

def train_model():
    # Load model and tokenizer
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    
    # Load datasets
    train_dataset = QADataset("data/processed/train.json")
    val_dataset = QADataset("data/processed/validation.json")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./ml_models/custom_models",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="./logs",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
    )
    
    # Train
    logger.info("Starting training...")
    trainer.train()
    
    # Save model
    trainer.save_model("./ml_models/custom_models/final_model")
    logger.info("Training completed!")

if __name__ == "__main__":
    train_model()