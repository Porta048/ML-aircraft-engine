"""
Performance Comparison Script
Compares original vs optimized RUL prediction system performance

This script benchmarks:
- Model loading time
- Data preprocessing time  
- Feature engineering time
- Inference latency
- Memory usage
- Overall accuracy

Author: ML Engineer
Date: 2025-08-04
"""

import time
import psutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from memory_profiler import profile
import warnings
warnings.filterwarnings('ignore')

# Import both versions
from production_ready_rul_predictor import ProductionRULPredictor
from optimized_production_rul_predictor import OptimizedRULPredictor

class PerformanceBenchmark:
    """
    Comprehensive performance benchmarking suite
    """
    
    def __init__(self):
        self.results = {
            'original': {},
            'optimized': {}
        }
        
        # Data paths
        self.train_path = "CMaps/train_FD001.txt"
        self.test_path = "CMaps/test_FD001.txt"
        self.rul_path = "CMaps/RUL_FD001.txt"
    
    def get_memory_usage(self):
        """Get current memory usage in MB"""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    
    def benchmark_data_loading(self, predictor_class, version_name):
        """Benchmark data loading performance"""
        print(f"\n🔄 Benchmarking {version_name} - Data Loading")
        
        predictor = predictor_class()
        initial_memory = self.get_memory_usage()
        
        start_time = time.time()
        
        if version_name == "Original":
            train_df, test_df, true_rul = predictor.load_data(
                self.train_path, self.test_path, self.rul_path
            )
        else:
            train_df, test_df, true_rul = predictor.load_data_optimized(
                self.train_path, self.test_path, self.rul_path
            )
        
        loading_time = time.time() - start_time
        final_memory = self.get_memory_usage()
        memory_used = final_memory - initial_memory
        
        self.results[version_name.lower()]['data_loading'] = {
            'time': loading_time,
            'memory_mb': memory_used,
            'train_shape': train_df.shape,
            'test_shape': test_df.shape
        }
        
        print(f"   ⏱️  Loading time: {loading_time:.2f}s")
        print(f"   💾 Memory used: {memory_used:.1f}MB")
        
        return predictor, train_df, test_df, true_rul
    
    def benchmark_preprocessing(self, predictor, train_df, test_df, version_name):
        """Benchmark data preprocessing performance"""
        print(f"\n🔄 Benchmarking {version_name} - Data Preprocessing")
        
        initial_memory = self.get_memory_usage()
        start_time = time.time()
        
        if version_name == "Original":
            # Simulate original preprocessing steps
            train_processed = predictor.calculate_rul(train_df.copy())
            train_processed = predictor.advanced_feature_engineering(train_processed)
            
            # Simulate test preprocessing  
            test_processed = predictor.advanced_feature_engineering(test_df.copy())
            
            # Simulate scaling
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
            feature_cols = [col for col in train_processed.columns 
                           if col not in ['unit_number', 'time_in_cycles', 'RUL']]
            
            train_processed[feature_cols] = scaler.fit_transform(train_processed[feature_cols])
            test_processed[feature_cols] = scaler.transform(test_processed[feature_cols])
            
        else:
            train_processed, test_processed = predictor.preprocess_data_inplace(train_df, test_df)
        
        preprocessing_time = time.time() - start_time
        final_memory = self.get_memory_usage()
        memory_used = final_memory - initial_memory
        
        self.results[version_name.lower()]['preprocessing'] = {
            'time': preprocessing_time,
            'memory_mb': memory_used,
            'features_created': train_processed.shape[1] - 3  # -3 for unit, time, RUL
        }
        
        print(f"   ⏱️  Preprocessing time: {preprocessing_time:.2f}s")
        print(f"   💾 Memory used: {memory_used:.1f}MB")
        print(f"   🔢 Features created: {train_processed.shape[1] - 3}")
        
        return train_processed, test_processed
    
    def benchmark_sequence_creation(self, predictor, train_processed, test_processed, version_name):
        """Benchmark sequence creation performance"""
        print(f"\n🔄 Benchmarking {version_name} - Sequence Creation")
        
        start_time = time.time()
        
        if version_name == "Original":
            X_train, y_train = predictor.create_sequences(train_processed)
            X_test, _ = predictor.create_sequences(test_processed)
        else:
            X_train, y_train = predictor.create_sequences_vectorized(train_processed)  
            X_test, _ = predictor.create_sequences_vectorized(test_processed)
        
        sequence_time = time.time() - start_time
        
        self.results[version_name.lower()]['sequence_creation'] = {
            'time': sequence_time,
            'train_sequences': len(X_train),
            'test_sequences': len(X_test),
            'sequence_shape': X_train.shape
        }
        
        print(f"   ⏱️  Sequence creation time: {sequence_time:.2f}s")
        print(f"   📊 Train sequences: {len(X_train)}")
        print(f"   📊 Test sequences: {len(X_test)}")
        
        return X_train, y_train, X_test
    
    def benchmark_model_training(self, predictor, X_train, y_train, version_name):
        """Benchmark model training performance"""
        print(f"\n🔄 Benchmarking {version_name} - Model Training")
        
        # Use smaller training set for benchmark
        train_size = min(1000, len(X_train))
        X_train_sample = X_train[:train_size]
        y_train_sample = y_train[:train_size]
        
        # Split for validation
        split_idx = int(0.8 * len(X_train_sample))
        X_train_split = X_train_sample[:split_idx]
        y_train_split = y_train_sample[:split_idx]
        X_val = X_train_sample[split_idx:]
        y_val = y_train_sample[split_idx:]
        
        start_time = time.time()
        
        if version_name == "Original":
            # Simulate original training (single model)
            model = predictor.build_lstm_model((X_train_split.shape[1], X_train_split.shape[2]))
            history = model.fit(
                X_train_split, y_train_split,
                validation_data=(X_val, y_val),
                epochs=10,  # Reduced for benchmark
                batch_size=32,
                verbose=0
            )
            predictor.models = {'primary': model}
            
        else:
            # Use reduced ensemble size for benchmark
            original_ensemble_size = predictor.config['ensemble_size']
            predictor.config['ensemble_size'] = 2  # Reduced for benchmark
            predictor.config['epochs'] = 10  # Reduced for benchmark
            
            models = predictor.train_ensemble_parallel(X_train_split, y_train_split, X_val, y_val)
            
            # Restore original config
            predictor.config['ensemble_size'] = original_ensemble_size
        
        training_time = time.time() - start_time
        
        self.results[version_name.lower()]['training'] = {
            'time': training_time,
            'samples_used': train_size,
            'models_trained': 1 if version_name == "Original" else 2
        }
        
        print(f"   ⏱️  Training time: {training_time:.2f}s")
        print(f"   📈 Samples used: {train_size}")
        
        return predictor
    
    def benchmark_inference(self, predictor, X_test, version_name, num_runs=100):
        """Benchmark inference performance"""
        print(f"\n🔄 Benchmarking {version_name} - Inference Performance")
        
        # Single prediction benchmark
        single_sample = X_test[:1]
        inference_times = []
        
        for _ in range(num_runs):
            start_time = time.time()
            
            if version_name == "Original":
                if hasattr(predictor.models, 'get'):
                    prediction = predictor.models['primary'].predict(single_sample, verbose=0)
                else:
                    prediction = list(predictor.models.values())[0].predict(single_sample, verbose=0)
            else:
                prediction = predictor.predict_ensemble_parallel(single_sample)
            
            inference_time = time.time() - start_time
            inference_times.append(inference_time * 1000)  # Convert to ms
        
        # Batch prediction benchmark  
        batch_size = min(100, len(X_test))
        batch_sample = X_test[:batch_size]
        
        start_time = time.time()
        
        if version_name == "Original":
            if hasattr(predictor, 'models') and predictor.models:
                if isinstance(predictor.models, dict):
                    model = list(predictor.models.values())[0]
                else:
                    model = predictor.models
                batch_predictions = model.predict(batch_sample, verbose=0)
        else:
            batch_predictions = predictor.predict_ensemble_parallel(batch_sample)
        
        batch_time = time.time() - start_time
        
        self.results[version_name.lower()]['inference'] = {
            'single_prediction_ms': {
                'mean': np.mean(inference_times),
                'std': np.std(inference_times),
                'min': np.min(inference_times),
                'max': np.max(inference_times)
            },
            'batch_prediction_time': batch_time,
            'batch_size': batch_size,
            'throughput_per_second': batch_size / batch_time
        }
        
        print(f"   ⏱️  Single prediction: {np.mean(inference_times):.1f}ms (±{np.std(inference_times):.1f}ms)")
        print(f"   ⏱️  Batch prediction ({batch_size} samples): {batch_time:.2f}s")
        print(f"   🚀 Throughput: {batch_size / batch_time:.1f} predictions/second")
    
    def create_performance_visualization(self):
        """Create comprehensive performance comparison visualization"""
        print(f"\n📊 Creating performance visualization...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Performance Comparison: Original vs Optimized RUL Predictor', 
                     fontsize=16, fontweight='bold')
        
        # 1. Loading Time Comparison
        loading_times = [
            self.results['original']['data_loading']['time'],
            self.results['optimized']['data_loading']['time']
        ]
        axes[0, 0].bar(['Original', 'Optimized'], loading_times, 
                       color=['red', 'green'], alpha=0.7)
        axes[0, 0].set_title('Data Loading Time')
        axes[0, 0].set_ylabel('Time (seconds)')
        
        # Add improvement annotation
        improvement_loading = loading_times[0] / loading_times[1]
        axes[0, 0].text(0.5, max(loading_times) * 0.8, 
                        f'{improvement_loading:.1f}x faster', 
                        ha='center', fontweight='bold', fontsize=12)
        
        # 2. Preprocessing Time Comparison
        prep_times = [
            self.results['original']['preprocessing']['time'],
            self.results['optimized']['preprocessing']['time']
        ]
        axes[0, 1].bar(['Original', 'Optimized'], prep_times,
                       color=['red', 'green'], alpha=0.7)
        axes[0, 1].set_title('Preprocessing Time')
        axes[0, 1].set_ylabel('Time (seconds)')
        
        improvement_prep = prep_times[0] / prep_times[1]
        axes[0, 1].text(0.5, max(prep_times) * 0.8,
                        f'{improvement_prep:.1f}x faster',
                        ha='center', fontweight='bold', fontsize=12)
        
        # 3. Memory Usage Comparison
        memory_usage = [
            self.results['original']['data_loading']['memory_mb'] + 
            self.results['original']['preprocessing']['memory_mb'],
            self.results['optimized']['data_loading']['memory_mb'] + 
            self.results['optimized']['preprocessing']['memory_mb']
        ]
        axes[0, 2].bar(['Original', 'Optimized'], memory_usage,
                       color=['red', 'green'], alpha=0.7)
        axes[0, 2].set_title('Memory Usage')
        axes[0, 2].set_ylabel('Memory (MB)')
        
        reduction_memory = (memory_usage[0] - memory_usage[1]) / memory_usage[0] * 100
        axes[0, 2].text(0.5, max(memory_usage) * 0.8,
                        f'{reduction_memory:.1f}% less',
                        ha='center', fontweight='bold', fontsize=12)
        
        # 4. Feature Count Comparison
        feature_counts = [
            self.results['original']['preprocessing']['features_created'],
            self.results['optimized']['preprocessing']['features_created']
        ]
        axes[1, 0].bar(['Original', 'Optimized'], feature_counts,
                       color=['red', 'green'], alpha=0.7)
        axes[1, 0].set_title('Feature Count')
        axes[1, 0].set_ylabel('Number of Features')
        
        # 5. Inference Latency Comparison
        inference_times = [
            self.results['original']['inference']['single_prediction_ms']['mean'],
            self.results['optimized']['inference']['single_prediction_ms']['mean']
        ]
        axes[1, 1].bar(['Original', 'Optimized'], inference_times,
                       color=['red', 'green'], alpha=0.7)
        axes[1, 1].set_title('Single Prediction Latency')
        axes[1, 1].set_ylabel('Time (milliseconds)')
        
        improvement_inference = inference_times[0] / inference_times[1]
        axes[1, 1].text(0.5, max(inference_times) * 0.8,
                        f'{improvement_inference:.1f}x faster',
                        ha='center', fontweight='bold', fontsize=12)
        
        # 6. Throughput Comparison
        throughputs = [
            self.results['original']['inference']['throughput_per_second'],
            self.results['optimized']['inference']['throughput_per_second']
        ]
        axes[1, 2].bar(['Original', 'Optimized'], throughputs,
                       color=['red', 'green'], alpha=0.7)
        axes[1, 2].set_title('Inference Throughput')
        axes[1, 2].set_ylabel('Predictions per Second')
        
        improvement_throughput = throughputs[1] / throughputs[0]
        axes[1, 2].text(0.5, max(throughputs) * 0.8,
                        f'{improvement_throughput:.1f}x faster',
                        ha='center', fontweight='bold', fontsize=12)
        
        plt.tight_layout()
        plt.savefig('performance_comparison_chart.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_performance_report(self):
        """Generate comprehensive performance report"""
        print(f"\n" + "="*80)
        print("COMPREHENSIVE PERFORMANCE COMPARISON REPORT")
        print("="*80)
        
        # Overall performance summary
        print(f"\n📊 OVERALL PERFORMANCE IMPROVEMENTS:")
        print("-" * 50)
        
        # Loading time improvement
        loading_improvement = (self.results['original']['data_loading']['time'] / 
                              self.results['optimized']['data_loading']['time'])
        print(f"Data Loading:     {loading_improvement:.1f}x faster")
        
        # Preprocessing improvement
        prep_improvement = (self.results['original']['preprocessing']['time'] / 
                           self.results['optimized']['preprocessing']['time'])
        print(f"Preprocessing:    {prep_improvement:.1f}x faster")
        
        # Memory reduction
        orig_memory = (self.results['original']['data_loading']['memory_mb'] + 
                      self.results['original']['preprocessing']['memory_mb'])
        opt_memory = (self.results['optimized']['data_loading']['memory_mb'] + 
                     self.results['optimized']['preprocessing']['memory_mb'])
        memory_reduction = ((orig_memory - opt_memory) / orig_memory) * 100
        print(f"Memory Usage:     {memory_reduction:.1f}% reduction")
        
        # Inference improvement
        inference_improvement = (self.results['original']['inference']['single_prediction_ms']['mean'] / 
                                self.results['optimized']['inference']['single_prediction_ms']['mean'])
        print(f"Inference Speed:  {inference_improvement:.1f}x faster")
        
        # Feature reduction
        orig_features = self.results['original']['preprocessing']['features_created']
        opt_features = self.results['optimized']['preprocessing']['features_created']
        feature_reduction = ((orig_features - opt_features) / orig_features) * 100
        print(f"Feature Count:    {feature_reduction:.1f}% reduction")
        
        print(f"\n💡 PRODUCTION READINESS METRICS:")
        print("-" * 50)
        
        opt_latency = self.results['optimized']['inference']['single_prediction_ms']['mean']
        opt_throughput = self.results['optimized']['inference']['throughput_per_second']
        
        print(f"Single Prediction Latency:  {opt_latency:.1f}ms")
        print(f"Batch Throughput:          {opt_throughput:.1f} predictions/second")
        print(f"Real-time Capable:         {'✓ YES' if opt_latency < 200 else '✗ NO'}")
        print(f"High-volume Capable:       {'✓ YES' if opt_throughput > 10 else '✗ NO'}")
        
        print(f"\n🎯 DEPLOYMENT RECOMMENDATIONS:")
        print("-" * 50)
        
        if opt_latency < 100:
            print("✓ Suitable for real-time critical applications")
        elif opt_latency < 500:
            print("✓ Suitable for near real-time applications")
        else:
            print("⚠ Consider further optimization for real-time use")
        
        if memory_reduction > 50:
            print("✓ Excellent memory efficiency for cloud deployment")
        elif memory_reduction > 25:
            print("✓ Good memory efficiency")
        else:
            print("⚠ Consider memory optimization")
        
        if opt_throughput > 50:
            print("✓ Excellent scalability for high-volume scenarios")
        elif opt_throughput > 10:
            print("✓ Good scalability")
        else:
            print("⚠ Consider throughput optimization")
        
        print("="*80)
    
    def run_full_benchmark(self):
        """Run complete performance benchmark"""
        print("🚀 Starting Comprehensive Performance Benchmark")
        print("="*80)
        
        # Benchmark Original Version
        print(f"\n{'='*20} ORIGINAL VERSION BENCHMARK {'='*20}")
        try:
            predictor_orig, train_df_orig, test_df_orig, true_rul_orig = self.benchmark_data_loading(
                ProductionRULPredictor, "Original"
            )
            
            train_processed_orig, test_processed_orig = self.benchmark_preprocessing(
                predictor_orig, train_df_orig, test_df_orig, "Original"
            )
            
            X_train_orig, y_train_orig, X_test_orig = self.benchmark_sequence_creation(
                predictor_orig, train_processed_orig, test_processed_orig, "Original"
            )
            
            predictor_orig = self.benchmark_model_training(
                predictor_orig, X_train_orig, y_train_orig, "Original"
            )
            
            self.benchmark_inference(predictor_orig, X_test_orig, "Original")
            
        except Exception as e:
            print(f"❌ Original version benchmark failed: {e}")
        
        # Benchmark Optimized Version
        print(f"\n{'='*20} OPTIMIZED VERSION BENCHMARK {'='*20}")
        try:
            predictor_opt, train_df_opt, test_df_opt, true_rul_opt = self.benchmark_data_loading(
                OptimizedRULPredictor, "Optimized"
            )
            
            train_processed_opt, test_processed_opt = self.benchmark_preprocessing(
                predictor_opt, train_df_opt, test_df_opt, "Optimized"
            )
            
            X_train_opt, y_train_opt, X_test_opt = self.benchmark_sequence_creation(
                predictor_opt, train_processed_opt, test_processed_opt, "Optimized"
            )
            
            predictor_opt = self.benchmark_model_training(
                predictor_opt, X_train_opt, y_train_opt, "Optimized"
            )
            
            self.benchmark_inference(predictor_opt, X_test_opt, "Optimized")
            
        except Exception as e:
            print(f"❌ Optimized version benchmark failed: {e}")
        
        # Generate reports and visualizations
        self.create_performance_visualization()
        self.generate_performance_report()

def main():
    """
    Main function to run performance comparison
    """
    benchmark = PerformanceBenchmark()
    benchmark.run_full_benchmark()

if __name__ == "__main__":
    main()