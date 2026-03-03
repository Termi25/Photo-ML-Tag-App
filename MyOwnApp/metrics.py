"""
Metrics Tracking Module
Tracks and measures system performance including inference latency, 
training convergence speed, and sorting accuracy.
"""

import time
import json
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime
from collections import defaultdict

from data_layer import metadata_store


class MetricsCollector:
    """Collects and stores performance metrics"""
    
    def __init__(self):
        self.metrics_file = Path("./data/metrics.json")
        self.metrics_file.parent.mkdir(parents=True, exist_ok=True)
        
        self.current_session = {
            'session_id': datetime.now().strftime("%Y%m%d_%H%M%S"),
            'start_time': time.time(),
            'inference_metrics': {
                'latencies': [],
                'batch_sizes': [],
                'total_images_processed': 0,
                'avg_latency_per_image': 0.0
            },
            'training_metrics': {
                'convergence_times': [],
                'epochs_to_converge': [],
                'final_accuracies': [],
                'total_training_time': 0.0
            },
            'accuracy_metrics': {
                'predictions_vs_ground_truth': [],
                'per_tag_accuracy': {},
                'overall_accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0
            },
            'preprocessing_metrics': {
                'avg_preprocessing_time': 0.0,
                'total_images_preprocessed': 0
            }
        }
    
    def start_inference_measurement(self):
        """Start measuring inference"""
        return time.time()
    
    def record_inference(self, start_time: float, batch_size: int = 1):
        """Record inference timing"""
        elapsed = time.time() - start_time
        self.current_session['inference_metrics']['latencies'].append(elapsed)
        self.current_session['inference_metrics']['batch_sizes'].append(batch_size)
        self.current_session['inference_metrics']['total_images_processed'] += batch_size
        
        # Update average
        self._update_avg_inference_latency()
    
    def _update_avg_inference_latency(self):
        """Calculate average inference latency per image"""
        total_images = self.current_session['inference_metrics']['total_images_processed']
        total_time = sum(self.current_session['inference_metrics']['latencies'])
        
        if total_images > 0:
            self.current_session['inference_metrics']['avg_latency_per_image'] = total_time / total_images
    
    def start_training_measurement(self):
        """Start measuring training"""
        return time.time()
    
    def record_training(self, start_time: float, epochs_completed: int, final_accuracy: float):
        """Record training metrics"""
        elapsed = time.time() - start_time
        self.current_session['training_metrics']['convergence_times'].append(elapsed)
        self.current_session['training_metrics']['epochs_to_converge'].append(epochs_completed)
        self.current_session['training_metrics']['final_accuracies'].append(final_accuracy)
        self.current_session['training_metrics']['total_training_time'] += elapsed
    
    def calculate_sorting_accuracy(self, ground_truth_available: bool = True):
        """Calculate sorting accuracy against ground truth"""
        if not ground_truth_available:
            print("No ground truth available for accuracy calculation")
            return
        
        # Query all images with both predictions and ground truth
        cursor = metadata_store.connection.cursor()
        
        # Get images that have ground truth
        cursor.execute('''
            SELECT DISTINCT i.id, i.file_path
            FROM images i
            JOIN ground_truth gt ON i.id = gt.image_id
        ''')
        
        images_with_ground_truth = cursor.fetchall()
        
        if len(images_with_ground_truth) == 0:
            print("No ground truth annotations found")
            return
        
        correct_predictions = 0
        total_predictions = 0
        per_tag_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
        
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        
        for image_id, image_path in images_with_ground_truth:
            # Get ground truth tags
            gt_tags = metadata_store.get_ground_truth_tags(image_id)
            
            # Get predicted tags
            predicted_tags_data = metadata_store.get_image_tags(image_id)
            predicted_tags = {tag['id'] for tag in predicted_tags_data if tag['source'] == 'model_prediction'}
            
            # Calculate metrics
            for tag_id in gt_tags:
                total_predictions += 1
                per_tag_stats[tag_id]['total'] += 1
                
                if tag_id in predicted_tags:
                    correct_predictions += 1
                    per_tag_stats[tag_id]['correct'] += 1
                    true_positives += 1
                else:
                    false_negatives += 1
            
            # Count false positives
            for tag_id in predicted_tags:
                if tag_id not in gt_tags:
                    false_positives += 1
        
        # Calculate overall accuracy
        overall_accuracy = 0.0
        if total_predictions > 0:
            overall_accuracy = (correct_predictions / total_predictions) * 100
            self.current_session['accuracy_metrics']['overall_accuracy'] = overall_accuracy
        
        # Calculate precision, recall, F1
        precision = 0.0
        recall = 0.0
        f1_score = 0.0
        if (true_positives + false_positives) > 0:
            precision = true_positives / (true_positives + false_positives)
            self.current_session['accuracy_metrics']['precision'] = precision
        
        if (true_positives + false_negatives) > 0:
            recall = true_positives / (true_positives + false_negatives)
            self.current_session['accuracy_metrics']['recall'] = recall
        
        if precision + recall > 0:
            f1_score = 2 * (precision * recall) / (precision + recall)
            self.current_session['accuracy_metrics']['f1_score'] = f1_score
        
        # Per-tag accuracy
        cursor.execute('SELECT id, tag_name FROM tags')
        tag_names = {row[0]: row[1] for row in cursor.fetchall()}
        
        for tag_id, stats in per_tag_stats.items():
            tag_name = tag_names.get(tag_id, f'tag_{tag_id}')
            if stats['total'] > 0:
                tag_accuracy = (stats['correct'] / stats['total']) * 100
                self.current_session['accuracy_metrics']['per_tag_accuracy'][tag_name] = {
                    'accuracy': tag_accuracy,
                    'correct': stats['correct'],
                    'total': stats['total']
                }
        
        print(f"Accuracy calculated: {overall_accuracy:.2f}%")
        print(f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1: {f1_score:.2f}")
    
    def record_preprocessing_metrics(self, preprocessor):
        """Record preprocessing metrics from preprocessing engine"""
        avg_time = preprocessor.get_avg_preprocessing_time()
        self.current_session['preprocessing_metrics']['avg_preprocessing_time'] = avg_time
        self.current_session['preprocessing_metrics']['total_images_preprocessed'] = len(preprocessor.preprocessing_times)
    
    def save_metrics(self):
        """Save metrics to file"""
        self.current_session['end_time'] = time.time()
        self.current_session['total_duration'] = self.current_session['end_time'] - self.current_session['start_time']
        
        # Load existing metrics
        all_metrics = []
        if self.metrics_file.exists():
            try:
                with open(self.metrics_file, 'r') as f:
                    all_metrics = json.load(f)
            except:
                all_metrics = []
        
        # Append current session
        all_metrics.append(self.current_session)
        
        # Save
        with open(self.metrics_file, 'w') as f:
            json.dump(all_metrics, f, indent=2)
        
        print(f"Metrics saved to {self.metrics_file}")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of current metrics"""
        return {
            'session_id': self.current_session['session_id'],
            'duration': time.time() - self.current_session['start_time'],
            'images_processed': self.current_session['inference_metrics']['total_images_processed'],
            'avg_inference_latency': self.current_session['inference_metrics']['avg_latency_per_image'],
            'total_training_time': self.current_session['training_metrics']['total_training_time'],
            'overall_accuracy': self.current_session['accuracy_metrics']['overall_accuracy'],
            'precision': self.current_session['accuracy_metrics']['precision'],
            'recall': self.current_session['accuracy_metrics']['recall'],
            'f1_score': self.current_session['accuracy_metrics']['f1_score']
        }
    
    def print_summary(self):
        """Print metrics summary"""
        summary = self.get_summary()
        
        print("\n" + "="*50)
        print("PERFORMANCE METRICS SUMMARY")
        print("="*50)
        print(f"Session ID: {summary['session_id']}")
        print(f"Duration: {summary['duration']:.2f} seconds")
        print(f"\nINFERENCE METRICS:")
        print(f"  Total Images Processed: {summary['images_processed']}")
        print(f"  Avg Latency per Image: {summary['avg_inference_latency']*1000:.2f} ms")
        print(f"\nTRAINING METRICS:")
        print(f"  Total Training Time: {summary['total_training_time']:.2f} seconds")
        if len(self.current_session['training_metrics']['final_accuracies']) > 0:
            print(f"  Latest Training Accuracy: {self.current_session['training_metrics']['final_accuracies'][-1]:.2f}%")
        print(f"\nACCURACY METRICS:")
        print(f"  Overall Accuracy: {summary['overall_accuracy']:.2f}%")
        print(f"  Precision: {summary['precision']:.4f}")
        print(f"  Recall: {summary['recall']:.4f}")
        print(f"  F1 Score: {summary['f1_score']:.4f}")
        
        if len(self.current_session['accuracy_metrics']['per_tag_accuracy']) > 0:
            print(f"\nPER-TAG ACCURACY:")
            for tag_name, stats in self.current_session['accuracy_metrics']['per_tag_accuracy'].items():
                print(f"  {tag_name}: {stats['accuracy']:.2f}% ({stats['correct']}/{stats['total']})")
        
        print("="*50 + "\n")
    
    def compare_with_previous_sessions(self) -> Dict[str, Any]:
        """Compare current session with previous sessions"""
        if not self.metrics_file.exists():
            return {}
        
        try:
            with open(self.metrics_file, 'r') as f:
                all_metrics = json.load(f)
            
            if len(all_metrics) < 2:
                return {}
            
            # Get averages from previous sessions
            prev_sessions = all_metrics[:-1]  # Exclude current session if already saved
            
            avg_inference = sum(s['inference_metrics']['avg_latency_per_image'] for s in prev_sessions) / len(prev_sessions)
            avg_accuracy = sum(s['accuracy_metrics']['overall_accuracy'] for s in prev_sessions) / len(prev_sessions)
            
            current_inference = self.current_session['inference_metrics']['avg_latency_per_image']
            current_accuracy = self.current_session['accuracy_metrics']['overall_accuracy']
            
            return {
                'inference_improvement': ((avg_inference - current_inference) / avg_inference * 100) if avg_inference > 0 else 0,
                'accuracy_improvement': current_accuracy - avg_accuracy,
                'previous_avg_inference': avg_inference,
                'previous_avg_accuracy': avg_accuracy
            }
        except Exception as e:
            print(f"Error comparing with previous sessions: {e}")
            return {}


class BenchmarkRunner:
    """Runs automated benchmarks to test system performance"""
    
    def __init__(self, controller, metrics_collector: MetricsCollector):
        self.controller = controller
        self.metrics = metrics_collector
    
    def run_full_benchmark(self, hyperparameter_configs: List[Dict[str, Any]] | None = None):
        """Run full benchmark with different configurations"""
        if hyperparameter_configs is None:
            # Default configurations to test
            hyperparameter_configs = [
                {'learning_rate': 0.001, 'batch_size': 16},
                {'learning_rate': 0.001, 'batch_size': 32},
                {'learning_rate': 0.0001, 'batch_size': 32},
            ]
        
        results = []
        
        for i, config in enumerate(hyperparameter_configs):
            print(f"\n{'='*60}")
            print(f"BENCHMARK {i+1}/{len(hyperparameter_configs)}")
            print(f"Configuration: {config}")
            print(f"{'='*60}\n")
            
            # Apply configuration
            from config import config_manager
            config_manager.update_hyperparameters(**config)
            
            # Run training
            start_time = self.metrics.start_training_measurement()
            self.controller.bootstrap_from_folders()
            
            # Wait for training to complete
            while self.controller.training_manager.is_training:
                time.sleep(1)
            
            training_metrics = self.controller.training_manager.training_loop.training_metrics
            self.metrics.record_training(
                start_time,
                training_metrics.get('epochs_completed', 0),
                training_metrics.get('accuracies', [0])[-1] if training_metrics.get('accuracies') else 0
            )
            
            # Run inference
            start_time = self.metrics.start_inference_measurement()
            self.controller.start_initial_sorting(auto_apply=True)
            self.metrics.record_inference(start_time, len(self.controller.current_images))
            
            # Calculate accuracy
            self.metrics.calculate_sorting_accuracy(ground_truth_available=True)
            
            # Store results
            results.append({
                'config': config,
                'summary': self.metrics.get_summary()
            })
            
            # Save metrics for this run
            self.metrics.save_metrics()
            
            # Reset for next run
            self.metrics = MetricsCollector()
        
        print("\n" + "="*60)
        print("BENCHMARK RESULTS COMPARISON")
        print("="*60)
        for i, result in enumerate(results):
            print(f"\nConfiguration {i+1}: {result['config']}")
            print(f"  Accuracy: {result['summary']['overall_accuracy']:.2f}%")
            print(f"  Avg Inference Latency: {result['summary']['avg_inference_latency']*1000:.2f} ms")
            print(f"  Training Time: {result['summary']['total_training_time']:.2f} s")
        
        return results


# Global metrics collector
metrics_collector = MetricsCollector()
