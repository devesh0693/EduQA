# data/scripts/verify_data.py
import json
from pathlib import Path

def verify_data():
    data_dir = Path("data")
    
    # Check processed files
    train_file = data_dir / "processed" / "train.json"
    val_file = data_dir / "processed" / "validation.json"
    
    if train_file.exists() and val_file.exists():
        with open(train_file) as f:
            train_data = json.load(f)
        with open(val_file) as f:
            val_data = json.load(f)
        
        print(f"✅ Training examples: {len(train_data)}")
        print(f"✅ Validation examples: {len(val_data)}")
        
        # Check first example
        if train_data:
            first_example = train_data[0]
            print(f"✅ Example question: {first_example['question'][:100]}...")
            print(f"✅ Example context: {first_example['context'][:100]}...")
    
    # Check dataset files
    datasets_dir = data_dir / "datasets"
    required_files = [
        "educational_qa.json",
        "squad_sample.json",
        "squad_2.0/train-v2.0.json",
        "squad_2.0/dev-v2.0.json"
    ]
    
    for file_path in required_files:
        full_path = datasets_dir / file_path
        if full_path.exists():
            print(f"✅ Found: {file_path}")
        else:
            print(f"❌ Missing: {file_path}")

if __name__ == "__main__":
    verify_data()