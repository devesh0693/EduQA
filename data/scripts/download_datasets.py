"""
Script to download and setup datasets for the Educational QA System
- SQuAD 2.0 (from Hugging Face)
- MS MARCO v1.1 (from Hugging Face, or local TSVs if provided)
- Sample Educational QA dataset
- Mini SQuAD sample
"""

import os
import json
import logging
from pathlib import Path
from datasets import load_dataset

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class DatasetDownloader:
    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        self.datasets_dir = self.data_dir / "datasets"
        self.squad_dir = self.datasets_dir / "squad_2.0"
        self.ms_marco_dir = self.datasets_dir / "ms_marco"

        # Ensure directories exist
        self.datasets_dir.mkdir(parents=True, exist_ok=True)
        self.squad_dir.mkdir(parents=True, exist_ok=True)
        self.ms_marco_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------
    # SQuAD 2.0
    # ------------------------------
    def download_squad(self):
        """Download SQuAD 2.0 dataset from Hugging Face"""
        logger.info("Downloading SQuAD 2.0 dataset from Hugging Face...")
        dataset = load_dataset("squad_v2")

        for split, filename in [("train", "train-v2.0.json"), ("validation", "dev-v2.0.json")]:
            filepath = self.squad_dir / filename
            if filepath.exists():
                logger.info(f"Already exists: {filepath}")
                continue

            dataset[split].to_json(filepath, orient="records", lines=False, indent=2)
            logger.info(f"✅ Saved {split} → {filepath}")

    # ------------------------------
    # MS MARCO
    # ------------------------------
    def download_ms_marco(self):
        """Download MS MARCO v1.1 dataset from Hugging Face"""
        logger.info("Downloading MS MARCO v1.1 dataset from Hugging Face...")
        dataset = load_dataset("ms_marco", "v1.1")

        for split, filename in [
            ("train", "train.json"),
            ("validation", "dev.json"),
            ("test", "test.json"),
        ]:
            filepath = self.ms_marco_dir / filename
            if filepath.exists():
                logger.info(f"Already exists: {filepath}")
                continue

            # Convert to pandas DataFrame first, then save
            df = dataset[split].to_pandas()
            df.to_json(filepath, orient="records", indent=2, force_ascii=False)
            logger.info(f"✅ Saved {split} → {filepath}")

        # Check if manual TSV files are available
        tsv_files = ["collection.tsv", "queries.train.tsv", "queries.dev.tsv", "qrels.train.tsv", "qrels.dev.tsv"]
        for fname in tsv_files:
            fpath = self.ms_marco_dir / fname
            if fpath.exists():
                logger.info(f"📂 Found local MS MARCO TSV file: {fname}")
            else:
                logger.warning(f"⚠️ Missing local MS MARCO TSV file: {fname}")

    # ------------------------------
    # Educational QA sample
    # ------------------------------
    def create_sample_educational_data(self):
        """Create sample educational QA data"""
        logger.info("Creating sample educational dataset...")

        sample_data = {
            "data": [
                {
                    "title": "Introduction to Machine Learning",
                    "paragraphs": [
                        {
                            "context": "Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed.",
                            "qas": [
                                {
                                    "id": "ml_001",
                                    "question": "What is machine learning?",
                                    "answers": [
                                        {
                                            "text": "a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed",
                                            "answer_start": 19,
                                        }
                                    ],
                                }
                            ],
                        }
                    ],
                }
            ]
        }

        filepath = self.datasets_dir / "educational_qa.json"
        with open(filepath, "w") as f:
            json.dump(sample_data, f, indent=2)

        logger.info(f"✅ Created: {filepath}")

    # ------------------------------
    # Mini SQuAD sample
    # ------------------------------
    def create_squad_sample(self):
        """Create a smaller sample from SQuAD for testing"""
        squad_file = self.squad_dir / "dev-v2.0.json"
        if not squad_file.exists():
            logger.warning("⚠️ SQuAD dev file not found, skipping sample creation")
            return

        # Load JSONL (each line = one example)
        with open(squad_file, "r", encoding="utf-8") as f:
            lines = [json.loads(line) for line in f]

        # Take first 5 examples
        sample_data = lines[:5]

        sample_file = self.datasets_dir / "squad_sample.json"
        with open(sample_file, "w", encoding="utf-8") as f:
            json.dump(sample_data, f, indent=2, ensure_ascii=False)

        logger.info(f"✅ Created: {sample_file}")


    # ------------------------------
    # Setup all
    # ------------------------------
    def setup_all(self):
        logger.info("Starting dataset setup...")

        self.download_squad()
        self.download_ms_marco()
        self.create_sample_educational_data()
        self.create_squad_sample()

        logger.info("🎉 All datasets downloaded and setup successfully!")


if __name__ == "__main__":
    downloader = DatasetDownloader()
    downloader.setup_all()
