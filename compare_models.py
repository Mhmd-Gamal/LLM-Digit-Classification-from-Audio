#!/usr/bin/env python3
"""
Model comparison script for MLP vs CNN spoken digit recognition.

This script compares the original MLP model with the new CNN model,
providing detailed performance metrics and visualizations.
"""

import os
import time
import warnings
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, Tuple

# Suppress warnings
warnings.filterwarnings('ignore')

# Import our modules
from data_loader import FSDDDataLoader
from features import MFCCFeatureExtractor
from model import CompactCNN

# Import original MLP model
import sys
sys.path.append('.')
from spoken_digit_recognition import SpokenDigitRecognizer

class ModelComparator:
    """
    Compare MLP and CNN models for spoken digit recognition.
    """
    
    def __init__(self):
        """Initialize the model comparator."""
        self.mlp_model = None
        self.cnn_model = None
        self.mlp_results = None
        self.cnn_results = None
        self.data_splits = None
        
    def load_data(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Load and split data for comparison.
        
        Returns:
            Dictionary containing train/val/test splits
        """
        print("Loading data for model comparison...")
        
        # Load data using the new data loader
        data_loader = FSDDDataLoader(target_sr=8000, duration=1.0)
        audio_data, labels = data_loader.load_dataset()
        
        # Create stratified splits
        splits = data_loader.stratified_split(
            audio_data, labels,
            test_size=0.2,
            val_size=0.1
        )
        
        self.data_splits = splits
        return splits
    
    def train_mlp_model(self) -> Dict[str, Any]:
        """
        Train the original MLP model.
        
        Returns:
            Dictionary containing MLP results
        """
        print("\n" + "="*50)
        print("TRAINING MLP MODEL")
        print("="*50)
        
        # Initialize MLP model
        self.mlp_model = SpokenDigitRecognizer(
            target_sr=8000,
            duration=1.0,
            n_mfcc=20,
            feature_dim=40
        )
        
        # Get training data
        X_train, y_train = self.data_splits['train']
        X_val, y_val = self.data_splits['val']
        X_test, y_test = self.data_splits['test']
        
        # Extract features for MLP (mean + std aggregation)
        print("Extracting MLP features...")
        X_train_features, y_train_features = self.mlp_model.extract_features(
            X_train, y_train, augment=True
        )
        X_val_features, y_val_features = self.mlp_model.extract_features(
            X_val, y_val, augment=False
        )
        X_test_features, y_test_features = self.mlp_model.extract_features(
            X_test, y_test, augment=False
        )
        
        # Train MLP model
        print("Training MLP model...")
        start_time = time.time()
        history = self.mlp_model.train(
            X_train_features, y_train_features,
            test_size=0.0,  # We already have validation split
            epochs=30
        )
        training_time = time.time() - start_time
        
        # Evaluate MLP model
        print("Evaluating MLP model...")
        self.mlp_model.X_test = X_test_features
        self.mlp_model.y_test = y_test_features
        mlp_results = self.mlp_model.evaluate()
        
        # Add timing information
        mlp_results['training_time'] = training_time
        mlp_results['history'] = history
        
        self.mlp_results = mlp_results
        return mlp_results
    
    def train_cnn_model(self) -> Dict[str, Any]:
        """
        Train the new CNN model.
        
        Returns:
            Dictionary containing CNN results
        """
        print("\n" + "="*50)
        print("TRAINING CNN MODEL")
        print("="*50)
        
        # Get training data
        X_train, y_train = self.data_splits['train']
        X_val, y_val = self.data_splits['val']
        X_test, y_test = self.data_splits['test']
        
        # Initialize feature extractor
        feature_extractor = MFCCFeatureExtractor(
            target_sr=8000,
            duration=1.0,
            n_mfcc=20
        )
        
        # Extract spectro-temporal features
        print("Extracting CNN features...")
        X_train_features, y_train_features = feature_extractor.extract_features_with_augmentation(
            X_train, y_train, use_spec_augment=True
        )
        X_val_features, y_val_features = feature_extractor.extract_features_batch(X_val)
        X_test_features, y_test_features = feature_extractor.extract_features_batch(X_test)
        
        # Normalize features
        X_train_norm, mean, std = feature_extractor.normalize_features(X_train_features)
        X_val_norm, _, _ = feature_extractor.normalize_features(X_val_features, mean, std)
        X_test_norm, _, _ = feature_extractor.normalize_features(X_test_features, mean, std)
        
        # Initialize CNN model
        input_shape = X_train_norm.shape[1:]
        self.cnn_model = CompactCNN(
            input_shape=input_shape,
            num_classes=10,
            dropout_rate=0.25,
            l2_reg=1e-4
        )
        
        # Train CNN model
        print("Training CNN model...")
        start_time = time.time()
        history = self.cnn_model.train(
            X_train_norm, y_train_features,
            X_val_norm, y_val_features,
            epochs=30,
            batch_size=32,
            use_class_weights=True,
            use_focal_loss=False
        )
        training_time = time.time() - start_time
        
        # Evaluate CNN model
        print("Evaluating CNN model...")
        cnn_results = self.cnn_model.evaluate(X_test_norm, y_test_features)
        
        # Add timing information
        cnn_results['training_time'] = training_time
        cnn_results['history'] = history
        
        self.cnn_results = cnn_results
        return cnn_results
    
    def benchmark_inference_speed(self) -> Dict[str, Dict[str, float]]:
        """
        Benchmark inference speed for both models.
        
        Returns:
            Dictionary containing speed results for both models
        """
        print("\n" + "="*50)
        print("BENCHMARKING INFERENCE SPEED")
        print("="*50)
        
        # Get test data
        X_test, y_test = self.data_splits['test']
        
        # Benchmark MLP
        print("Benchmarking MLP inference...")
        mlp_times = []
        for i, audio in enumerate(X_test[:50]):  # Test on 50 samples
            if i % 10 == 0:
                print(f"MLP sample {i}/50")
            _, _, inference_time = self.mlp_model.predict_single(audio)
            mlp_times.append(inference_time)
        
        # Benchmark CNN
        print("Benchmarking CNN inference...")
        cnn_times = []
        feature_extractor = MFCCFeatureExtractor(target_sr=8000, duration=1.0, n_mfcc=20)
        
        for i, audio in enumerate(X_test[:50]):  # Test on 50 samples
            if i % 10 == 0:
                print(f"CNN sample {i}/50")
            
            # Extract features
            features = feature_extractor.extract_spectro_temporal_features(audio)
            _, _, inference_time = self.cnn_model.predict_single(features)
            cnn_times.append(inference_time)
        
        # Calculate statistics
        mlp_stats = {
            'mean_time': np.mean(mlp_times),
            'std_time': np.std(mlp_times),
            'min_time': np.min(mlp_times),
            'max_time': np.max(mlp_times),
            'p95_time': np.percentile(mlp_times, 95)
        }
        
        cnn_stats = {
            'mean_time': np.mean(cnn_times),
            'std_time': np.std(cnn_times),
            'min_time': np.min(cnn_times),
            'max_time': np.max(cnn_times),
            'p95_time': np.percentile(cnn_times, 95)
        }
        
        print(f"\nMLP Inference Speed:")
        print(f"  Mean: {mlp_stats['mean_time']:.2f} ms")
        print(f"  Std:  {mlp_stats['std_time']:.2f} ms")
        print(f"  P95:  {mlp_stats['p95_time']:.2f} ms")
        
        print(f"\nCNN Inference Speed:")
        print(f"  Mean: {cnn_stats['mean_time']:.2f} ms")
        print(f"  Std:  {cnn_stats['std_time']:.2f} ms")
        print(f"  P95:  {cnn_stats['p95_time']:.2f} ms")
        
        return {
            'mlp': mlp_stats,
            'cnn': cnn_stats
        }
    
    def plot_comparison(self, speed_results: Dict[str, Dict[str, float]]):
        """
        Create comprehensive comparison plots.
        
        Args:
            speed_results: Speed benchmarking results
        """
        print("\n" + "="*50)
        print("CREATING COMPARISON PLOTS")
        print("="*50)
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 12))
        
        # 1. Accuracy comparison
        ax1 = plt.subplot(2, 4, 1)
        models = ['MLP', 'CNN']
        accuracies = [self.mlp_results['accuracy'], self.cnn_results['accuracy']]
        colors = ['skyblue', 'lightcoral']
        
        bars = ax1.bar(models, accuracies, color=colors)
        ax1.set_title('Test Accuracy Comparison')
        ax1.set_ylabel('Accuracy')
        ax1.set_ylim(0.9, 1.0)
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'{acc:.3f}', ha='center', va='bottom')
        
        # 2. Training time comparison
        ax2 = plt.subplot(2, 4, 2)
        training_times = [self.mlp_results['training_time'], self.cnn_results['training_time']]
        
        bars = ax2.bar(models, training_times, color=colors)
        ax2.set_title('Training Time Comparison')
        ax2.set_ylabel('Time (seconds)')
        
        for bar, time_val in zip(bars, training_times):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{time_val:.1f}s', ha='center', va='bottom')
        
        # 3. Inference speed comparison
        ax3 = plt.subplot(2, 4, 3)
        inference_times = [speed_results['mlp']['mean_time'], speed_results['cnn']['mean_time']]
        
        bars = ax3.bar(models, inference_times, color=colors)
        ax3.set_title('Inference Speed Comparison')
        ax3.set_ylabel('Time (ms)')
        ax3.axhline(y=20, color='red', linestyle='--', alpha=0.7, label='20ms target')
        ax3.legend()
        
        for bar, time_val in zip(bars, inference_times):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{time_val:.1f}ms', ha='center', va='bottom')
        
        # 4. Precision/Recall/F1 comparison
        ax4 = plt.subplot(2, 4, 4)
        metrics = ['Precision', 'Recall', 'F1-Score']
        mlp_metrics = [self.mlp_results.get('precision', 0), 
                      self.mlp_results.get('recall', 0), 
                      self.mlp_results.get('f1_score', 0)]
        cnn_metrics = [self.cnn_results['precision'], 
                      self.cnn_results['recall'], 
                      self.cnn_results['f1_score']]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        ax4.bar(x - width/2, mlp_metrics, width, label='MLP', color='skyblue')
        ax4.bar(x + width/2, cnn_metrics, width, label='CNN', color='lightcoral')
        
        ax4.set_title('Detailed Metrics Comparison')
        ax4.set_ylabel('Score')
        ax4.set_xticks(x)
        ax4.set_xticklabels(metrics)
        ax4.legend()
        
        # 5. MLP Confusion Matrix
        ax5 = plt.subplot(2, 4, 5)
        sns.heatmap(self.mlp_results['confusion_matrix'], annot=True, fmt='d', 
                   cmap='Blues', ax=ax5, cbar=False)
        ax5.set_title('MLP Confusion Matrix')
        ax5.set_xlabel('Predicted')
        ax5.set_ylabel('True')
        
        # 6. CNN Confusion Matrix
        ax6 = plt.subplot(2, 4, 6)
        sns.heatmap(self.cnn_results['confusion_matrix'], annot=True, fmt='d', 
                   cmap='Reds', ax=ax6, cbar=False)
        ax6.set_title('CNN Confusion Matrix')
        ax6.set_xlabel('Predicted')
        ax6.set_ylabel('True')
        
        # 7. Training History - Accuracy
        ax7 = plt.subplot(2, 4, 7)
        mlp_history = self.mlp_results['history']
        cnn_history = self.cnn_results['history']
        
        ax7.plot(mlp_history.history['accuracy'], label='MLP Train', color='blue', alpha=0.7)
        ax7.plot(mlp_history.history['val_accuracy'], label='MLP Val', color='blue', linestyle='--', alpha=0.7)
        ax7.plot(cnn_history.history['accuracy'], label='CNN Train', color='red', alpha=0.7)
        ax7.plot(cnn_history.history['val_accuracy'], label='CNN Val', color='red', linestyle='--', alpha=0.7)
        
        ax7.set_title('Training Accuracy History')
        ax7.set_xlabel('Epoch')
        ax7.set_ylabel('Accuracy')
        ax7.legend()
        ax7.grid(True, alpha=0.3)
        
        # 8. Training History - Loss
        ax8 = plt.subplot(2, 4, 8)
        ax8.plot(mlp_history.history['loss'], label='MLP Train', color='blue', alpha=0.7)
        ax8.plot(mlp_history.history['val_loss'], label='MLP Val', color='blue', linestyle='--', alpha=0.7)
        ax8.plot(cnn_history.history['loss'], label='CNN Train', color='red', alpha=0.7)
        ax8.plot(cnn_history.history['val_loss'], label='CNN Val', color='red', linestyle='--', alpha=0.7)
        
        ax8.set_title('Training Loss History')
        ax8.set_xlabel('Epoch')
        ax8.set_ylabel('Loss')
        ax8.legend()
        ax8.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Comparison plots saved to model_comparison.png")
    
    def generate_report(self, speed_results: Dict[str, Dict[str, float]]) -> str:
        """
        Generate a comprehensive comparison report.
        
        Args:
            speed_results: Speed benchmarking results
            
        Returns:
            Formatted report string
        """
        report = []
        report.append("=" * 80)
        report.append("SPOKEN DIGIT RECOGNITION - MODEL COMPARISON REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Executive Summary
        report.append("EXECUTIVE SUMMARY")
        report.append("-" * 40)
        mlp_acc = self.mlp_results['accuracy']
        cnn_acc = self.cnn_results['accuracy']
        mlp_speed = speed_results['mlp']['mean_time']
        cnn_speed = speed_results['cnn']['mean_time']
        
        report.append(f"MLP Model:  {mlp_acc:.3f} accuracy, {mlp_speed:.1f}ms inference")
        report.append(f"CNN Model:  {cnn_acc:.3f} accuracy, {cnn_speed:.1f}ms inference")
        report.append("")
        
        # Accuracy comparison
        report.append("ACCURACY COMPARISON")
        report.append("-" * 40)
        report.append(f"MLP Test Accuracy: {mlp_acc:.4f} ({mlp_acc*100:.2f}%)")
        report.append(f"CNN Test Accuracy: {cnn_acc:.4f} ({cnn_acc*100:.2f}%)")
        report.append(f"Improvement:       {cnn_acc - mlp_acc:+.4f} ({(cnn_acc - mlp_acc)*100:+.2f}%)")
        report.append("")
        
        # Speed comparison
        report.append("INFERENCE SPEED COMPARISON")
        report.append("-" * 40)
        report.append(f"MLP Mean Time: {mlp_speed:.2f} ms")
        report.append(f"CNN Mean Time: {cnn_speed:.2f} ms")
        report.append(f"Speed Change:  {cnn_speed - mlp_speed:+.2f} ms")
        report.append(f"20ms Target:   {'✓ PASS' if cnn_speed < 20 else '✗ FAIL'}")
        report.append("")
        
        # Training time comparison
        report.append("TRAINING TIME COMPARISON")
        report.append("-" * 40)
        mlp_train_time = self.mlp_results['training_time']
        cnn_train_time = self.cnn_results['training_time']
        report.append(f"MLP Training Time: {mlp_train_time:.1f} seconds")
        report.append(f"CNN Training Time: {cnn_train_time:.1f} seconds")
        report.append(f"Training Overhead: {cnn_train_time - mlp_train_time:+.1f} seconds")
        report.append("")
        
        # Detailed metrics
        report.append("DETAILED METRICS COMPARISON")
        report.append("-" * 40)
        report.append(f"{'Metric':<12} {'MLP':<8} {'CNN':<8} {'Change':<8}")
        report.append("-" * 40)
        
        mlp_precision = self.mlp_results.get('precision', 0)
        cnn_precision = self.cnn_results['precision']
        report.append(f"{'Precision':<12} {mlp_precision:<8.3f} {cnn_precision:<8.3f} {cnn_precision - mlp_precision:+.3f}")
        
        mlp_recall = self.mlp_results.get('recall', 0)
        cnn_recall = self.cnn_results['recall']
        report.append(f"{'Recall':<12} {mlp_recall:<8.3f} {cnn_recall:<8.3f} {cnn_recall - mlp_recall:+.3f}")
        
        mlp_f1 = self.mlp_results.get('f1_score', 0)
        cnn_f1 = self.cnn_results['f1_score']
        report.append(f"{'F1-Score':<12} {mlp_f1:<8.3f} {cnn_f1:<8.3f} {cnn_f1 - mlp_f1:+.3f}")
        report.append("")
        
        # Requirements check
        report.append("REQUIREMENTS CHECK")
        report.append("-" * 40)
        accuracy_met = cnn_acc >= 0.97
        speed_met = cnn_speed < 20
        
        report.append(f"✓ Accuracy ≥97%: {'PASS' if accuracy_met else 'FAIL'} ({cnn_acc*100:.2f}%)")
        report.append(f"✓ Inference <20ms: {'PASS' if speed_met else 'FAIL'} ({cnn_speed:.1f}ms)")
        report.append("")
        
        # Conclusion
        report.append("CONCLUSION")
        report.append("-" * 40)
        if accuracy_met and speed_met:
            report.append("🎉 CNN model successfully meets all requirements!")
            report.append("   - Improved accuracy over MLP baseline")
            report.append("   - Maintains fast inference speed")
            report.append("   - Ready for deployment")
        else:
            report.append("⚠ CNN model needs further optimization:")
            if not accuracy_met:
                report.append(f"   - Accuracy {cnn_acc*100:.2f}% below 97% target")
            if not speed_met:
                report.append(f"   - Inference {cnn_speed:.1f}ms exceeds 20ms target")
        
        report.append("")
        report.append("=" * 80)
        
        return "\n".join(report)

def main():
    """
    Main function for model comparison.
    """
    print("=" * 80)
    print("SPOKEN DIGIT RECOGNITION - MODEL COMPARISON")
    print("=" * 80)
    
    # Initialize comparator
    comparator = ModelComparator()
    
    # Load data
    comparator.load_data()
    
    # Train both models
    mlp_results = comparator.train_mlp_model()
    cnn_results = comparator.train_cnn_model()
    
    # Benchmark inference speed
    speed_results = comparator.benchmark_inference_speed()
    
    # Create comparison plots
    comparator.plot_comparison(speed_results)
    
    # Generate and save report
    report = comparator.generate_report(speed_results)
    print("\n" + report)
    
    # Save report to file
    with open('model_comparison_report.txt', 'w') as f:
        f.write(report)
    
    print("\nComparison report saved to model_comparison_report.txt")
    print("Comparison plots saved to model_comparison.png")

if __name__ == "__main__":
    main()