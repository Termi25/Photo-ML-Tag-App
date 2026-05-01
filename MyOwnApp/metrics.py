import time
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
from datetime import datetime
from collections import defaultdict

from data_layer import metadata_store


class MetricsCollector:
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
        return time.time()
    
    def record_inference(self, start_time: float, batch_size: int = 1):
        elapsed = time.time() - start_time
        self.current_session['inference_metrics']['latencies'].append(elapsed)
        self.current_session['inference_metrics']['batch_sizes'].append(batch_size)
        self.current_session['inference_metrics']['total_images_processed'] += batch_size
        
        # Update average
        self._update_avg_inference_latency()
    
    def _update_avg_inference_latency(self):
        total_images = self.current_session['inference_metrics']['total_images_processed']
        total_time = sum(self.current_session['inference_metrics']['latencies'])
        
        if total_images > 0:
            self.current_session['inference_metrics']['avg_latency_per_image'] = total_time / total_images
    
    def start_training_measurement(self):
        return time.time()
    
    def record_training(self, start_time: float, epochs_completed: int, final_accuracy: float):
        elapsed = time.time() - start_time
        self.current_session['training_metrics']['convergence_times'].append(elapsed)
        self.current_session['training_metrics']['epochs_to_converge'].append(epochs_completed)
        self.current_session['training_metrics']['final_accuracies'].append(final_accuracy)
        self.current_session['training_metrics']['total_training_time'] += elapsed
    
    def calculate_sorting_accuracy(self, ground_truth_available: bool = True):
        """Calculate sorting accuracy, per-class F1, macro/micro F1, and confusion pairs."""
        if not ground_truth_available:
            print("No ground truth available for accuracy calculation")
            return

        cursor = metadata_store.connection.cursor()
        cursor.execute('''
            SELECT DISTINCT i.id, i.file_path
            FROM images i
            JOIN ground_truth gt ON i.id = gt.image_id
        ''')
        images_with_ground_truth = cursor.fetchall()

        if len(images_with_ground_truth) == 0:
            print("No ground truth annotations found")
            return

        cursor.execute('SELECT id, tag_name FROM tags')
        tag_names: Dict[int, str] = {row[0]: row[1] for row in cursor.fetchall()}

        correct_predictions = 0
        total_predictions   = 0
        true_positives  = 0
        false_positives = 0
        false_negatives = 0

        # per_tag_stats[tag_id] = {tp, fp, fn}
        per_tag_stats: Dict[int, Dict[str, int]] = defaultdict(
            lambda: {'tp': 0, 'fp': 0, 'fn': 0}
        )
        # confusion_pairs[(true_tag_id, predicted_tag_id)] = count of co-occurrences
        # where true_tag was missed (FN) and predicted_tag was a FP on the same image
        confusion_pairs: Dict[Tuple[int, int], int] = defaultdict(int)

        for image_id, _image_path in images_with_ground_truth:
            gt_tags       = set(metadata_store.get_ground_truth_tags(image_id))
            predicted_data = metadata_store.get_image_tags(image_id)
            predicted_tags = {tag['id'] for tag in predicted_data
                              if tag['source'] == 'model_prediction'}

            for tag_id in gt_tags:
                total_predictions += 1
                per_tag_stats[tag_id]['tp' if tag_id in predicted_tags else 'fn'] += 1
                if tag_id in predicted_tags:
                    correct_predictions += 1
                    true_positives += 1
                else:
                    false_negatives += 1

            fp_tags = predicted_tags - gt_tags
            fn_tags = gt_tags - predicted_tags
            false_positives += len(fp_tags)

            for tag_id in fp_tags:
                per_tag_stats[tag_id]['fp'] += 1
            # Build confusion pairs: for each missed true tag, record what was predicted instead
            for true_tag in fn_tags:
                for pred_tag in fp_tags:
                    confusion_pairs[(true_tag, pred_tag)] += 1

        # Micro precision / recall / F1
        precision = (true_positives / (true_positives + false_positives)
                     if (true_positives + false_positives) > 0 else 0.0)
        recall    = (true_positives / (true_positives + false_negatives)
                     if (true_positives + false_negatives) > 0 else 0.0)
        micro_f1  = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0.0)

        # Per-class F1 and macro F1
        per_class_f1_scores: List[float] = []
        per_tag_accuracy_out: Dict[str, Any] = {}
        for tag_id, stats in per_tag_stats.items():
            tag_name = tag_names.get(tag_id, f'tag_{tag_id}')
            tp, fp, fn = stats['tp'], stats['fp'], stats['fn']
            total = tp + fn
            p_k   = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r_k   = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1_k  = 2 * p_k * r_k / (p_k + r_k) if (p_k + r_k) > 0 else 0.0
            per_class_f1_scores.append(f1_k)
            per_tag_accuracy_out[tag_name] = {
                'accuracy':  (tp / total * 100) if total > 0 else 0.0,
                'correct':   tp,
                'total':     total,
                'precision': p_k,
                'recall':    r_k,
                'f1':        f1_k,
            }

        macro_f1 = (sum(per_class_f1_scores) / len(per_class_f1_scores)
                    if per_class_f1_scores else 0.0)
        overall_accuracy = (correct_predictions / total_predictions * 100
                            if total_predictions > 0 else 0.0)

        # Store confusion pairs (top 20 by count) for later export
        top_confusion: List[Dict[str, Any]] = sorted(
            [{'true_tag':  tag_names.get(t, f'tag_{t}'),
              'pred_tag':  tag_names.get(p, f'tag_{p}'),
              'count':     cnt}
             for (t, p), cnt in confusion_pairs.items()],
            key=lambda d: d['count'], reverse=True
        )[:20]

        am = self.current_session['accuracy_metrics']
        am['overall_accuracy']  = overall_accuracy
        am['precision']         = precision
        am['recall']            = recall
        am['f1_score']          = micro_f1
        am['macro_f1']          = macro_f1
        am['micro_f1']          = micro_f1
        am['per_tag_accuracy']  = per_tag_accuracy_out
        am['per_class_f1']      = {n: d['f1'] for n, d in per_tag_accuracy_out.items()}
        am['top_confusion_pairs'] = top_confusion

        print(f"Accuracy:  {overall_accuracy:.2f}%")
        print(f"Precision: {precision:.4f}  Recall: {recall:.4f}  Micro F1: {micro_f1:.4f}  Macro F1: {macro_f1:.4f}")

    def export_confusion_matrix(
        self,
        output_path: str,
        top_n: int = 15,
    ) -> None:
        """Save a ranked top-N confused pairs table as a PNG heatmap + CSV.

        Reads from the most recent calculate_sorting_accuracy() call.
        """
        pairs = self.current_session['accuracy_metrics'].get('top_confusion_pairs', [])
        if not pairs:
            print("No confusion data available — run calculate_sorting_accuracy() first.")
            return

        rows = pairs[:top_n]
        out  = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        # --- CSV ---
        csv_path = out.with_suffix('.csv')
        try:
            with open(csv_path, 'w', encoding='utf-8') as f:
                f.write("true_tag,predicted_tag,count\n")
                for r in rows:
                    f.write(f"{r['true_tag']},{r['pred_tag']},{r['count']}\n")
            print(f"Confusion pairs CSV: {csv_path}")
        except Exception as e:
            print(f"Could not write CSV: {e}")

        # --- PNG heatmap (optional, requires matplotlib) ---
        try:
            import matplotlib.pyplot as plt
            import numpy as np

            true_labels = [r['true_tag']  for r in rows]
            pred_labels = [r['pred_tag']  for r in rows]
            counts      = [r['count']     for r in rows]

            all_labels = sorted(set(true_labels) | set(pred_labels))
            n = len(all_labels)
            label_idx = {lbl: i for i, lbl in enumerate(all_labels)}
            matrix = np.zeros((n, n), dtype=int)
            for r in rows:
                matrix[label_idx[r['true_tag']], label_idx[r['pred_tag']]] = r['count']

            fig, ax = plt.subplots(figsize=(max(8, n), max(6, n)))
            im = ax.imshow(matrix, cmap='YlOrRd')
            ax.set_xticks(range(n)); ax.set_yticks(range(n))
            ax.set_xticklabels(all_labels, rotation=45, ha='right', fontsize=8)
            ax.set_yticklabels(all_labels, fontsize=8)
            ax.set_xlabel('Predicted tag')
            ax.set_ylabel('True tag')
            ax.set_title(f'Top-{top_n} misclassified tag pairs')
            plt.colorbar(im, ax=ax)
            plt.tight_layout()
            png_path = out.with_suffix('.png')
            fig.savefig(str(png_path), dpi=120)
            plt.close(fig)
            print(f"Confusion matrix PNG: {png_path}")
        except Exception as e:
            print(f"Could not generate PNG (matplotlib required): {e}")
    
    def record_preprocessing_metrics(self, preprocessor):
        avg_time = preprocessor.get_avg_preprocessing_time()
        self.current_session['preprocessing_metrics']['avg_preprocessing_time'] = avg_time
        self.current_session['preprocessing_metrics']['total_images_preprocessed'] = len(preprocessor.preprocessing_times)
    
    def save_metrics(self):
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
    def __init__(self, controller, metrics_collector: MetricsCollector) -> None:
        self.controller = controller
        self.metrics    = metrics_collector

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_full_benchmark(
        self,
        dataset: str = 'personal',
        hyperparameter_configs: List[Dict[str, Any]] | None = None,
        model_types: List[str] | None = None,
        test_split: float = 0.15,
        coco_data_dir: str | None = None,
        coco_max_images: int | None = None,
        progress_callback=None,
    ) -> List[Dict[str, Any]]:
        """Run a full benchmark across hyperparameter configs and model architectures.

        Args:
            dataset:               'personal' or 'coco'
            hyperparameter_configs: list of dicts with keys matching HyperParameters fields
            model_types:           list of model_type strings to compare (e.g. ['resnet18', 'efficientnet_b0'])
            test_split:            fraction of data to hold out as a test set (0 to disable)
            coco_data_dir:         local directory for COCO data cache
            coco_max_images:       max COCO val2017 images to use
            progress_callback:     optional callable(event, value) for UI updates
        """
        if dataset not in ('personal', 'coco'):
            raise ValueError(f"dataset must be 'personal' or 'coco', got '{dataset}'")

        if hyperparameter_configs is None:
            hyperparameter_configs = [
                {'learning_rate': 0.001,  'batch_size': 32},
            ]
        if model_types is None:
            model_types = [config_manager.app_config.model_type]

        from config import config_manager

        # ------ COCO: load data once before the loops ------
        coco_image_paths:  List[str]         = []
        coco_multi_labels: List[List[float]] = []

        if dataset == 'coco':
            from coco_benchmark import COCOBenchmarkLoader

            data_dir   = coco_data_dir   or config_manager.app_config.coco_data_dir   or './data/coco'
            max_images = coco_max_images or config_manager.app_config.coco_max_images or 500

            msg = f"Loading COCO benchmark dataset (max {max_images} images)..."
            print(f"\n{msg}")
            if progress_callback:
                progress_callback('status', msg)

            loader = COCOBenchmarkLoader(data_dir, max_images=max_images)
            coco_image_paths, coco_multi_labels = loader.load_data(progress_callback)

            if not coco_image_paths:
                print("No COCO images available — aborting benchmark.")
                return []

        # ------ Run one pass per (model_type × hyperparameter_config) ------
        results: List[Dict[str, Any]] = []
        total_runs = len(model_types) * len(hyperparameter_configs)
        run_idx    = 0

        for model_type in model_types:
            config_manager.app_config.model_type = model_type

            for hp_config in hyperparameter_configs:
                run_idx += 1
                print(f"\n{'='*60}")
                print(f"BENCHMARK {run_idx}/{total_runs}  [{dataset.upper()}]  model={model_type}")
                print(f"Config: {hp_config}")
                print(f"{'='*60}\n")

                if progress_callback:
                    progress_callback('status',
                        f'Benchmark {run_idx}/{total_runs}: {model_type}  {hp_config}')

                config_manager.update_hyperparameters(**hp_config)
                self._clear_benchmark_state()

                t0 = time.time()
                if dataset == 'personal':
                    training_loop = self._run_personal_benchmark(test_split=test_split)
                else:
                    training_loop = self._run_coco_benchmark(
                        coco_image_paths, coco_multi_labels, test_split=test_split
                    )
                elapsed = time.time() - t0

                summary = self.metrics.get_summary()
                test_results = (training_loop.training_metrics.get('test_results', {})
                                if training_loop else {})

                results.append({
                    'model_type':    model_type,
                    'config':        hp_config,
                    'dataset':       dataset,
                    'summary':       summary,
                    'test_results':  test_results,
                    'elapsed_s':     elapsed,
                })

                self.metrics.save_metrics()
                self.metrics = MetricsCollector()

        # ------ Print multi-model comparison table ------
        print("\n" + "=" * 70)
        print(f"BENCHMARK RESULTS  [{dataset.upper()}]  ({total_runs} runs)")
        print("=" * 70)
        header = (f"{'Model':<20} {'Params':>7}  {'Val Acc':>8}  "
                  f"{'Test Acc':>9}  {'MacroF1':>8}  {'MicroF1':>8}  {'Time(s)':>8}")
        print(header)
        print("-" * 70)

        _PARAM_COUNTS = {
            'resnet18': 11.7, 'resnet34': 21.8, 'resnet50': 25.6,
            'efficientnet_b0': 5.3, 'efficientnet_b1': 7.8,
            'mobilenet_v2': 3.4, 'vgg16': 138.4,
        }
        for r in results:
            s     = r['summary']
            tr    = r.get('test_results', {})
            params = _PARAM_COUNTS.get(r['model_type'], 0.0)
            val_acc  = s.get('overall_accuracy', 0.0)
            test_acc = tr.get('test_accuracy',  float('nan'))
            macro_f1 = tr.get('macro_f1',        s.get('macro_f1', float('nan')))
            micro_f1 = tr.get('micro_f1',        s.get('micro_f1', float('nan')))
            test_str = f"{test_acc:9.2f}%" if test_acc == test_acc else "      n/a"
            mf1_str  = f"{macro_f1:8.4f}"  if macro_f1 == macro_f1 else "     n/a"
            mi_str   = f"{micro_f1:8.4f}"  if micro_f1 == micro_f1 else "     n/a"
            print(f"{r['model_type']:<20} {params:>6.1f}M  {val_acc:8.2f}%  "
                  f"{test_str}  {mf1_str}  {mi_str}  {r['elapsed_s']:8.1f}")

        if progress_callback:
            progress_callback('complete', results)

        return results

    # ------------------------------------------------------------------
    # Per-dataset benchmark passes
    # ------------------------------------------------------------------

    def _run_personal_benchmark(self, test_split: float = 0.15):
        from config import config_manager
        from ml_module import TrainingLoop

        # Prepare data from folder structure
        training_loop = TrainingLoop()
        image_paths, labels = training_loop.prepare_data_from_folders(
            str(self.controller.root_folder)
        )

        start_time = self.metrics.start_training_measurement()
        tm = training_loop.train(image_paths, labels, test_split=test_split)
        self.metrics.record_training(
            start_time,
            tm.get('epochs_completed', 0),
            tm.get('accuracies', [0])[-1] if tm.get('accuracies') else 0,
        )

        if test_split > 0:
            training_loop.evaluate_on_test_set()

        self._seed_personal_ground_truth()

        start_time = self.metrics.start_inference_measurement()
        self.controller.start_initial_sorting(auto_apply=True)
        self.metrics.record_inference(start_time, len(self.controller.current_images))

        self.metrics.calculate_sorting_accuracy(ground_truth_available=True)
        return training_loop

    def _run_coco_benchmark(
        self,
        image_paths:  List[str],
        multi_labels: List[List[float]],
        test_split:   float = 0.15,
    ):
        from ml_module import TrainingLoop, InferenceEngine

        training_loop = TrainingLoop()
        start_time    = self.metrics.start_training_measurement()
        tm            = training_loop.train(image_paths, multi_labels, test_split=test_split)
        self.metrics.record_training(
            start_time,
            tm.get('epochs_completed', 0),
            tm.get('accuracies', [0])[-1] if tm.get('accuracies') else 0,
        )

        if test_split > 0:
            training_loop.evaluate_on_test_set()

        inference_engine = InferenceEngine()
        if not inference_engine.load_model('latest'):
            print("Could not load model after COCO training — skipping inference.")
            return training_loop

        start_time       = self.metrics.start_inference_measurement()
        predictions_list = inference_engine.predict_batch(image_paths)
        self.metrics.record_inference(start_time, len(image_paths))

        for img_path, preds in zip(image_paths, predictions_list):
            if preds:
                self.controller.tagging_engine.apply_predicted_tags(img_path, preds)

        self.metrics.calculate_sorting_accuracy(ground_truth_available=True)
        return training_loop

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _clear_benchmark_state(self) -> None:
        cursor = metadata_store.connection.cursor()
        cursor.execute("DELETE FROM ground_truth")
        cursor.execute(
            "DELETE FROM image_tags WHERE source IN "
            "('model_prediction', 'coco_ground_truth')"
        )
        metadata_store.connection.commit()
        print("Benchmark state cleared (ground truth + predictions removed).")

    def _seed_personal_ground_truth(self) -> None:
        cursor = metadata_store.connection.cursor()
        now    = datetime.now().isoformat()
        cursor.execute(
            """
            INSERT OR IGNORE INTO ground_truth (image_id, tag_id, verified_by, verified_at)
            SELECT image_id, tag_id, 'folder_structure', ?
            FROM   image_tags
            WHERE  source = 'folder_supervised'
            """,
            (now,),
        )
        metadata_store.connection.commit()
        print(f"Seeded {cursor.rowcount} ground-truth entries from folder structure.")


# Global metrics collector
metrics_collector = MetricsCollector()
