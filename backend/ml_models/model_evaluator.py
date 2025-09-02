"""
Model evaluation module for Educational QA System
Comprehensive evaluation and benchmarking
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from question_answering import EducationalQASystem
from data_preprocessor import DataPreprocessor
from config import PERFORMANCE_TARGETS, DATASET_CONFIG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelEvaluator:
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize model evaluator
        
        Args:
            model_path: Path to the model to evaluate
        """
        self.qa_system = EducationalQASystem(model_path) if model_path else None
        self.preprocessor = DataPreprocessor()
        self.evaluation_results = {}
        
        logger.info("Model evaluator initialized")
    
    def comprehensive_evaluation(
        self, 
        model_path: str, 
        test_datasets: Dict[str, List[Dict]],
        output_dir: Path
    ) -> Dict[str, Any]:
        """
        Perform comprehensive evaluation on multiple datasets
        
        Args:
            model_path: Path to the model
            test_datasets: Dictionary of test datasets
            output_dir: Directory to save evaluation results
            
        Returns:
            Comprehensive evaluation results
        """
        logger.info("Starting comprehensive model evaluation...")
        
        # Initialize QA system with the model
        self.qa_system = EducationalQASystem(model_path)
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        overall_results = {
            'model_path': model_path,
            'evaluation_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'datasets': {},
            'overall_metrics': {},
            'performance_analysis': {}
        }
        
        all_predictions = []
        all_ground_truths = []
        all_response_times = []
        all_confidences = []
        
        # Evaluate on each dataset
        for dataset_name, examples in test_datasets.items():
            logger.info(f"Evaluating on {dataset_name} ({len(examples)} examples)...")
            
            dataset_results = self.evaluate_on_dataset(examples, dataset_name)
            overall_results['datasets'][dataset_name] = dataset_results
            
            # Collect data for overall analysis
            all_predictions.extend(dataset_results['predictions'])
            all_ground_truths.extend(dataset_results['ground_truths'])
            all_response_times.extend(dataset_results['response_times'])
            all_confidences.extend(dataset_results['confidences'])
        
        # Calculate overall metrics
        overall_results['overall_metrics'] = self.calculate_overall_metrics(
            all_predictions, all_ground_truths, all_response_times, all_confidences
        )
        
        # Performance analysis
        overall_results['performance_analysis'] = self.analyze_performance(
            all_predictions, all_ground_truths, all_confidences, all_response_times
        )
        
        # Generate visualizations
        self.generate_evaluation_plots(overall_results, output_dir)
        
        # Save results
        results_file = output_dir / 'evaluation_results.json'
        with open(results_file, 'w') as f:
            json.dump(overall_results, f, indent=2, default=str)
        
        # Generate report
        self.generate_evaluation_report(overall_results, output_dir)
        
        logger.info("Comprehensive evaluation completed")
        return overall_results
    
    def evaluate_on_dataset(self, examples: List[Dict], dataset_name: str) -> Dict[str, Any]:
        """Evaluate model on a single dataset"""
        predictions = []
        ground_truths = []
        response_times = []
        confidences = []
        question_types = []
        
        correct_answers = 0
        total_f1 = 0.0
        no_answer_predictions = 0
        
        for i, example in enumerate(examples):
            if i % 100 == 0:
                logger.info(f"  Processing example {i}/{len(examples)}")
            
            # Get prediction
            result = self.qa_system.answer_question(
                example['question'],
                example['context'],
                return_confidence=True
            )
            
            # Extract ground truth
            ground_truth = example.get('answer_text', '') if not example.get('is_impossible', False) else ''
            
            # Store results
            predictions.append(result['answer'])
            ground_truths.append(ground_truth)
            response_times.append(result['response_time'])
            confidences.append(result['confidence'])
            
            # Classify question type
            q_type = self.classify_question_type(example['question'])
            question_types.append(q_type)
            
            # Calculate metrics
            if ground_truth:  # Has answer
                metrics = self.qa_system.evaluate_answer_quality(result['answer'], ground_truth)
                if metrics['exact_match'] == 1.0:
                    correct_answers += 1
                total_f1 += metrics['f1_score']
            else:  # No answer expected
                if not result['answer'] or result['confidence'] < 0.5:
                    correct_answers += 1
                    total_f1 += 1.0
            
            if not result['answer']:
                no_answer_predictions += 1
        
        # Calculate dataset metrics
        accuracy = correct_answers / len(examples)
        avg_f1 = total_f1 / len(examples)
        avg_response_time = np.mean(response_times)
        avg_confidence = np.mean(confidences)
        
        return {
            'dataset_name': dataset_name,
            'total_examples': len(examples),
            'accuracy': accuracy,
            'f1_score': avg_f1,
            'avg_response_time': avg_response_time,
            'avg_confidence': avg_confidence,
            'no_answer_predictions': no_answer_predictions,
            'predictions': predictions,
            'ground_truths': ground_truths,
            'response_times': response_times,
            'confidences': confidences,
            'question_types': question_types,
            'meets_accuracy_target': accuracy >= PERFORMANCE_TARGETS['accuracy_threshold'],
            'meets_f1_target': avg_f1 >= PERFORMANCE_TARGETS['f1_score_threshold'],
            'meets_speed_target': avg_response_time <= PERFORMANCE_TARGETS['response_time_threshold']
        }
    
    def calculate_overall_metrics(
        self, 
        predictions: List[str], 
        ground_truths: List[str],
        response_times: List[float],
        confidences: List[float]
    ) -> Dict[str, float]:
        """Calculate overall performance metrics"""
        
        # Exact match and F1 calculation
        correct = 0
        total_f1 = 0.0
        
        for pred, truth in zip(predictions, ground_truths):
            if not truth:  # No answer case
                if not pred:
                    correct += 1
                    total_f1 += 1.0
            else:
                # Normalize for comparison
                pred_norm = self.normalize_answer(pred)
                truth_norm = self.normalize_answer(truth)
                
                if pred_norm == truth_norm:
                    correct += 1
                
                # F1 score
                total_f1 += self.compute_f1(pred_norm, truth_norm)
        
        return {
            'overall_accuracy': correct / len(predictions),
            'overall_f1': total_f1 / len(predictions),
            'avg_response_time': np.mean(response_times),
            'median_response_time': np.median(response_times),
            'avg_confidence': np.mean(confidences),
            'std_confidence': np.std(confidences),
            'fast_responses_pct': np.mean([t <= PERFORMANCE_TARGETS['response_time_threshold'] for t in response_times]) * 100,
            'high_confidence_pct': np.mean([c >= PERFORMANCE_TARGETS['confidence_threshold'] for c in confidences]) * 100
        }
    
    def analyze_performance(
        self,
        predictions: List[str],
        ground_truths: List[str], 
        confidences: List[float],
        response_times: List[float]
    ) -> Dict[str, Any]:
        """Analyze model performance patterns"""
        
        # Confidence vs Accuracy analysis
        confidence_bins = np.linspace(0, 1, 11)
        binned_accuracy = []
        
        for i in range(len(confidence_bins) - 1):
            low, high = confidence_bins[i], confidence_bins[i + 1]
            mask = (np.array(confidences) >= low) & (np.array(confidences) < high)
            
            if np.any(mask):
                subset_preds = [p for p, m in zip(predictions, mask) if m]
                subset_truths = [t for t, m in zip(ground_truths, mask) if m]
                
                if subset_preds:
                    correct = sum(1 for p, t in zip(subset_preds, subset_truths) 
                                if self.normalize_answer(p) == self.normalize_answer(t))
                    binned_accuracy.append(correct / len(subset_preds))
                else:
                    binned_accuracy.append(0.0)
            else:
                binned_accuracy.append(0.0)
        
        # Response time analysis
        time_percentiles = {
            'p50': np.percentile(response_times, 50),
            'p90': np.percentile(response_times, 90),
            'p95': np.percentile(response_times, 95),
            'p99': np.percentile(response_times, 99)
        }
        
        # Error analysis
        errors = []
        for pred, truth, conf in zip(predictions, ground_truths, confidences):
            if truth and self.normalize_answer(pred) != self.normalize_answer(truth):
                errors.append({
                    'prediction': pred,
                    'ground_truth': truth,
                    'confidence': conf,
                    'error_type': self.classify_error_type(pred, truth)
                })
        
        return {
            'confidence_accuracy_correlation': np.corrcoef(confidences, [
                1 if self.normalize_answer(p) == self.normalize_answer(t) else 0
                for p, t in zip(predictions, ground_truths)
            ])[0, 1],
            'confidence_bins': confidence_bins.tolist(),
            'binned_accuracy': binned_accuracy,
            'response_time_percentiles': time_percentiles,
            'error_count': len(errors),
            'error_rate': len(errors) / len(predictions),
            'common_error_types': self.analyze_error_types(errors)
        }
    
    def classify_question_type(self, question: str) -> str:
        """Classify the type of question"""
        question_lower = question.lower()
        
        if question_lower.startswith(('what', 'which')):
            return 'factual'
        elif question_lower.startswith(('how', 'why')):
            return 'explanation'
        elif question_lower.startswith(('when', 'where')):
            return 'temporal_spatial'
        elif question_lower.startswith(('who')):
            return 'person'
        elif '?' in question and any(word in question_lower for word in ['is', 'are', 'was', 'were', 'do', 'does', 'did']):
            return 'yes_no'
        else:
            return 'other'
    
    def classify_error_type(self, prediction: str, ground_truth: str) -> str:
        """Classify the type of error made"""
        if not prediction:
            return 'no_answer_given'
        elif not ground_truth:
            return 'answer_when_none_expected'
        elif len(prediction) < len(ground_truth) * 0.5:
            return 'incomplete_answer'
        elif len(prediction) > len(ground_truth) * 2:
            return 'over_detailed_answer'
        else:
            return 'incorrect_content'
    
    def analyze_error_types(self, errors: List[Dict]) -> Dict[str, int]:
        """Analyze common error types"""
        error_types = {}
        for error in errors:
            error_type = error['error_type']
            error_types[error_type] = error_types.get(error_type, 0) + 1
        
        return dict(sorted(error_types.items(), key=lambda x: x[1], reverse=True))
    
    def normalize_answer(self, answer: str) -> str:
        """Normalize answer for evaluation"""
        import re
        import string
        
        if not answer:
            return ""
        
        # Convert to lowercase
        answer = answer.lower()
        
        # Remove punctuation
        answer = ''.join(char for char in answer if char not in string.punctuation)
        
        # Remove extra whitespace
        answer = re.sub(r'\s+', ' ', answer).strip()
        
        return answer
    
    def compute_f1(self, prediction: str, ground_truth: str) -> float:
        """Compute F1 score between prediction and ground truth"""
        if not prediction and not ground_truth:
            return 1.0
        if not prediction or not ground_truth:
            return 0.0
        
        pred_tokens = set(prediction.split())
        true_tokens = set(ground_truth.split())
        
        if not pred_tokens or not true_tokens:
            return 0.0
        
        common_tokens = pred_tokens & true_tokens
        
        if not common_tokens:
            return 0.0
        
        precision = len(common_tokens) / len(pred_tokens)
        recall = len(common_tokens) / len(true_tokens)
        
        f1 = 2 * (precision * recall) / (precision + recall)
        return f1
    
    def generate_evaluation_plots(self, results: Dict[str, Any], output_dir: Path):
        """Generate evaluation visualization plots"""
        logger.info("Generating evaluation plots...")
        
        # Set style
        plt.style.use('seaborn-v0_8')
        
        # 1. Dataset comparison plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Performance Analysis', fontsize=16)
        
        # Extract dataset metrics
        datasets = list(results['datasets'].keys())
        accuracies = [results['datasets'][d]['accuracy'] for d in datasets]
        f1_scores = [results['datasets'][d]['f1_score'] for d in datasets]
        response_times = [results['datasets'][d]['avg_response_time'] for d in datasets]
        
        # Accuracy comparison
        axes[0, 0].bar(datasets, accuracies)
        axes[0, 0].axhline(y=PERFORMANCE_TARGETS['accuracy_threshold'], color='r', linestyle='--', label='Target')
        axes[0, 0].set_title('Accuracy by Dataset')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # F1 Score comparison
        axes[0, 1].bar(datasets, f1_scores)
        axes[0, 1].axhline(y=PERFORMANCE_TARGETS['f1_score_threshold'], color='r', linestyle='--', label='Target')
        axes[0, 1].set_title('F1 Score by Dataset')
        axes[0, 1].set_ylabel('F1 Score')
        axes[0, 1].legend()
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Response time comparison
        axes[1, 0].bar(datasets, response_times)
        axes[1, 0].axhline(y=PERFORMANCE_TARGETS['response_time_threshold'], color='r', linestyle='--', label='Target')
        axes[1, 0].set_title('Response Time by Dataset')
        axes[1, 0].set_ylabel('Response Time (seconds)')
        axes[1, 0].legend()
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Overall metrics pie chart
        overall_metrics = results['overall_metrics']
        metrics_data = [
            overall_metrics['overall_accuracy'],
            overall_metrics['overall_f1'],
            overall_metrics['fast_responses_pct'] / 100,
            overall_metrics['high_confidence_pct'] / 100
        ]
        metrics_labels = ['Accuracy', 'F1 Score', 'Fast Responses', 'High Confidence']
        
        axes[1, 1].bar(metrics_labels, metrics_data)
        axes[1, 1].set_title('Overall Performance Metrics')
        axes[1, 1].set_ylabel('Score/Percentage')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'performance_overview.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Confidence vs Accuracy plot
        if 'performance_analysis' in results:
            perf_analysis = results['performance_analysis']
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Confidence vs Accuracy
            confidence_bins = perf_analysis['confidence_bins']
            binned_accuracy = perf_analysis['binned_accuracy']
            bin_centers = [(confidence_bins[i] + confidence_bins[i+1]) / 2 for i in range(len(confidence_bins)-1)]
            
            ax1.plot(bin_centers, binned_accuracy, marker='o')
            ax1.set_xlabel('Confidence')
            ax1.set_ylabel('Accuracy')
            ax1.set_title('Confidence vs Accuracy')
            ax1.grid(True)
            
            # Response time distribution
            all_response_times = []
            for dataset_name in datasets:
                all_response_times.extend(results['datasets'][dataset_name]['response_times'])
            
            ax2.hist(all_response_times, bins=50, alpha=0.7, edgecolor='black')
            ax2.axvline(x=PERFORMANCE_TARGETS['response_time_threshold'], color='r', linestyle='--', label='Target')
            ax2.set_xlabel('Response Time (seconds)')
            ax2.set_ylabel('Frequency')
            ax2.set_title('Response Time Distribution')
            ax2.legend()
            
            plt.tight_layout()
            plt.savefig(output_dir / 'detailed_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        logger.info("Evaluation plots saved")
    
    def generate_evaluation_report(self, results: Dict[str, Any], output_dir: Path):
        """Generate a comprehensive evaluation report"""
        logger.info("Generating evaluation report...")
        
        report_path = output_dir / 'evaluation_report.md'
        
        with open(report_path, 'w') as f:
            f.write("# Educational QA System - Model Evaluation Report\n\n")
            f.write(f"**Model Path:** {results['model_path']}\n")
            f.write(f"**Evaluation Date:** {results['evaluation_timestamp']}\n\n")
            
            # Overall Performance
            overall = results['overall_metrics']
            f.write("## Overall Performance\n\n")
            f.write(f"- **Overall Accuracy:** {overall['overall_accuracy']:.4f}\n")
            f.write(f"- **Overall F1 Score:** {overall['overall_f1']:.4f}\n")
            f.write(f"- **Average Response Time:** {overall['avg_response_time']:.4f} seconds\n")
            f.write(f"- **Median Response Time:** {overall['median_response_time']:.4f} seconds\n")
            f.write(f"- **Average Confidence:** {overall['avg_confidence']:.4f}\n")
            f.write(f"- **Fast Responses (≤{PERFORMANCE_TARGETS['response_time_threshold']}s):** {overall['fast_responses_pct']:.1f}%\n")
            f.write(f"- **High Confidence (≥{PERFORMANCE_TARGETS['confidence_threshold']}):** {overall['high_confidence_pct']:.1f}%\n\n")
            
            # Performance Targets
            f.write("## Performance Target Achievement\n\n")
            f.write(f"- **Accuracy Target (≥{PERFORMANCE_TARGETS['accuracy_threshold']}):** {'✅ ACHIEVED' if overall['overall_accuracy'] >= PERFORMANCE_TARGETS['accuracy_threshold'] else '❌ NOT ACHIEVED'}\n")
            f.write(f"- **F1 Score Target (≥{PERFORMANCE_TARGETS['f1_score_threshold']}):** {'✅ ACHIEVED' if overall['overall_f1'] >= PERFORMANCE_TARGETS['f1_score_threshold'] else '❌ NOT ACHIEVED'}\n")
            f.write(f"- **Response Time Target (≤{PERFORMANCE_TARGETS['response_time_threshold']}s):** {'✅ ACHIEVED' if overall['avg_response_time'] <= PERFORMANCE_TARGETS['response_time_threshold'] else '❌ NOT ACHIEVED'}\n\n")
            
            # Dataset-wise Performance
            f.write("## Dataset-wise Performance\n\n")
            for dataset_name, dataset_results in results['datasets'].items():
                f.write(f"### {dataset_name}\n")
                f.write(f"- **Examples:** {dataset_results['total_examples']}\n")
                f.write(f"- **Accuracy:** {dataset_results['accuracy']:.4f}\n")
                f.write(f"- **F1 Score:** {dataset_results['f1_score']:.4f}\n")
                f.write(f"- **Avg Response Time:** {dataset_results['avg_response_time']:.4f}s\n")
                f.write(f"- **Avg Confidence:** {dataset_results['avg_confidence']:.4f}\n")
                f.write(f"- **No Answer Predictions:** {dataset_results['no_answer_predictions']}\n")
                
                # Target achievement for this dataset
                targets_met = [
                    dataset_results['meets_accuracy_target'],
                    dataset_results['meets_f1_target'],
                    dataset_results['meets_speed_target']
                ]
                f.write(f"- **Targets Met:** {sum(targets_met)}/3\n\n")
            
            # Error Analysis
            if 'performance_analysis' in results:
                perf_analysis = results['performance_analysis']
                f.write("## Error Analysis\n\n")
                f.write(f"- **Total Errors:** {perf_analysis['error_count']}\n")
                f.write(f"- **Error Rate:** {perf_analysis['error_rate']:.4f}\n")
                f.write(f"- **Confidence-Accuracy Correlation:** {perf_analysis['confidence_accuracy_correlation']:.4f}\n\n")
                
                if 'common_error_types' in perf_analysis:
                    f.write("### Common Error Types\n")
                    for error_type, count in perf_analysis['common_error_types'].items():
                        f.write(f"- **{error_type.replace('_', ' ').title()}:** {count}\n")
                    f.write("\n")
            
            # Response Time Analysis
            if 'response_time_percentiles' in results['performance_analysis']:
                percentiles = results['performance_analysis']['response_time_percentiles']
                f.write("## Response Time Analysis\n\n")
                f.write(f"- **50th percentile:** {percentiles['p50']:.4f}s\n")
                f.write(f"- **90th percentile:** {percentiles['p90']:.4f}s\n")
                f.write(f"- **95th percentile:** {percentiles['p95']:.4f}s\n")
                f.write(f"- **99th percentile:** {percentiles['p99']:.4f}s\n\n")
            
            # Recommendations
            f.write("## Recommendations\n\n")
            
            if overall['overall_accuracy'] < PERFORMANCE_TARGETS['accuracy_threshold']:
                f.write("- **Accuracy Improvement:** Consider fine-tuning on more domain-specific data or increasing model capacity\n")
            
            if overall['overall_f1'] < PERFORMANCE_TARGETS['f1_score_threshold']:
                f.write("- **F1 Score Improvement:** Focus on reducing partial matches and improving answer span detection\n")
            
            if overall['avg_response_time'] > PERFORMANCE_TARGETS['response_time_threshold']:
                f.write("- **Speed Optimization:** Consider model quantization, caching, or using a smaller model variant\n")
            
            if overall['avg_confidence'] < 0.7:
                f.write("- **Confidence Calibration:** Consider confidence calibration techniques or threshold tuning\n")
            
            f.write("\n---\n*Report generated automatically by Educational QA System Evaluator*")
        
        logger.info(f"Evaluation report saved to {report_path}")
    
    def benchmark_against_baselines(self, test_examples: List[Dict]) -> Dict[str, Any]:
        """Benchmark model against simple baselines"""
        logger.info("Running benchmark against baselines...")
        
        results = {}
        
        # Baseline 1: Random answer selection
        random_predictions = []
        for example in test_examples:
            context_words = example['context'].split()
            if len(context_words) > 5:
                start_idx = np.random.randint(0, len(context_words) - 5)
                end_idx = start_idx + np.random.randint(1, 6)
                random_answer = ' '.join(context_words[start_idx:end_idx])
            else:
                random_answer = context_words[0] if context_words else ""
            random_predictions.append(random_answer)
        
        # Baseline 2: First sentence as answer
        first_sentence_predictions = []
        for example in test_examples:
            sentences = example['context'].split('.')
            first_sentence_predictions.append(sentences[0].strip() if sentences else "")
        
        # Evaluate baselines
        ground_truths = [ex.get('answer_text', '') for ex in test_examples]
        
        # Random baseline
        random_metrics = self.calculate_baseline_metrics(random_predictions, ground_truths)
        results['random_baseline'] = random_metrics
        
        # First sentence baseline
        first_sent_metrics = self.calculate_baseline_metrics(first_sentence_predictions, ground_truths)
        results['first_sentence_baseline'] = first_sent_metrics
        
        # Our model results (if available)
        if self.qa_system:
            model_predictions = []
            model_confidences = []
            model_times = []
            
            for example in test_examples:
                result = self.qa_system.answer_question(example['question'], example['context'])
                model_predictions.append(result['answer'])
                model_confidences.append(result['confidence'])
                model_times.append(result['response_time'])
            
            model_metrics = self.calculate_baseline_metrics(model_predictions, ground_truths)
            model_metrics['avg_confidence'] = np.mean(model_confidences)
            model_metrics['avg_response_time'] = np.mean(model_times)
            results['our_model'] = model_metrics
        
        return results
    
    def calculate_baseline_metrics(self, predictions: List[str], ground_truths: List[str]) -> Dict[str, float]:
        """Calculate metrics for baseline methods"""
        correct = 0
        total_f1 = 0.0
        
        for pred, truth in zip(predictions, ground_truths):
            pred_norm = self.normalize_answer(pred)
            truth_norm = self.normalize_answer(truth)
            
            if pred_norm == truth_norm:
                correct += 1
            
            total_f1 += self.compute_f1(pred_norm, truth_norm)
        
        return {
            'accuracy': correct / len(predictions),
            'f1_score': total_f1 / len(predictions)
        }
    
    def load_test_datasets(self) -> Dict[str, List[Dict]]:
        """Load available test datasets"""
        datasets = {}
        
        # Load SQuAD dev set
        if DATASET_CONFIG['squad_dev'].exists():
            datasets['squad_dev'] = self.preprocessor.load_squad_dataset(
                DATASET_CONFIG['squad_dev']
            )[:500]  # Limit for faster evaluation
        
        # Load educational test set
        if DATASET_CONFIG['educational_qa'].exists():
            edu_data = self.preprocessor.load_educational_dataset(
                DATASET_CONFIG['educational_qa']
            )
            # Use last 20% as test set
            test_size = int(len(edu_data) * 0.2)
            datasets['educational_test'] = edu_data[-test_size:]
        
        # Load sample dataset
        if DATASET_CONFIG['squad_sample'].exists():
            datasets['squad_sample'] = self.preprocessor.load_squad_dataset(
                DATASET_CONFIG['squad_sample']
            )
        
        return datasets

if __name__ == "__main__":
    # Example usage
    from pathlib import Path
    
    # Initialize evaluator
    evaluator = ModelEvaluator()
    
    # Load test datasets
    test_datasets = evaluator.load_test_datasets()
    
    if test_datasets:
        print(f"Loaded {len(test_datasets)} test datasets:")
        for name, examples in test_datasets.items():
            print(f"  {name}: {len(examples)} examples")
        
        # Example evaluation (replace with your model path)
        model_path = "bert-base-uncased"  # or path to your fine-tuned model
        output_dir = Path("evaluation_results")
        
        # Run comprehensive evaluation
        results = evaluator.comprehensive_evaluation(
            model_path=model_path,
            test_datasets=test_datasets,
            output_dir=output_dir
        )
        
        print("\nEvaluation completed!")
        print(f"Overall Accuracy: {results['overall_metrics']['overall_accuracy']:.4f}")
        print(f"Overall F1 Score: {results['overall_metrics']['overall_f1']:.4f}")
        print(f"Results saved to: {output_dir}")
    else:
        print("No test datasets found. Please check your dataset files.")