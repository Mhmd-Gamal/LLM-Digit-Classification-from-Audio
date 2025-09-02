#!/usr/bin/env python3
"""
Training script for the CNN-based spoken digit recognition system.

This script orchestrates the complete training pipeline including data loading,
feature extraction, model training, and evaluation.
"""

import os
import time
import warnings
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any

# Suppress warnings
warnings.filterwarnings('ignore')

# Import our modules
from data_loader import FSDDDataLoader
from features import MFCCFeatureExtractor
from model import CompactCNN

def plot_training_history(history: Any, save_path: str = "training_history_cnn.png"):
    """
    Plot training history for the CNN model.
    
    Args:
        history: Keras training history object
        save_path: Path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Plot loss
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Training history plot saved to {save_path}")

def plot_confusion_matrix(cm: np.ndarray, save_path: str = "confusion_matrix_cnn.png"):
    """
    Plot confusion matrix for the CNN model.
    
    Args:
        cm: Confusion matrix array
        save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=range(10), yticklabels=range(10))
    plt.title('Confusion Matrix - CNN Spoken Digit Recognition')
    plt.xlabel('Predicted Digit')
    plt.ylabel('True Digit')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Confusion matrix plot saved to {save_path}")

def benchmark_inference_speed(model: CompactCNN, X_test: np.ndarray, 
                            num_samples: int = 100) -> Dict[str, float]:
    """
    Benchmark inference speed of the CNN model.
    
    Args:
        model: Trained CNN model
        X_test: Test features for benchmarking
        num_samples: Number of samples to test
        
    Returns:
        Dictionary containing timing statistics
    """
    print(f"Benchmarking inference speed on {num_samples} samples...")
    
    # Select random samples for benchmarking
    indices = np.random.choice(len(X_test), min(num_samples, len(X_test)), replace=False)
    test_samples = X_test[indices]
    
    inference_times = []
    
    for i, sample in enumerate(test_samples):
        if i % 20 == 0:
            print(f"Benchmarking sample {i}/{len(test_samples)}")
        
        _, _, inference_time = model.predict_single(sample)
        inference_times.append(inference_time)
    
    # Calculate statistics
    mean_time = np.mean(inference_times)
    std_time = np.std(inference_times)
    min_time = np.min(inference_times)
    max_time = np.max(inference_times)
    p95_time = np.percentile(inference_times, 95)
    
    print(f"\nInference Speed Benchmark Results:")
    print(f"Mean time: {mean_time:.2f} ms")
    print(f"Std time:  {std_time:.2f} ms")
    print(f"Min time:  {min_time:.2f} ms")
    print(f"Max time:  {max_time:.2f} ms")
    print(f"95th percentile: {p95_time:.2f} ms")
    
    # Check if meets requirement
    if mean_time < 20:
        print("✓ Inference speed meets <20ms requirement")
    else:
        print("⚠ Inference speed exceeds 20ms requirement")
    
    return {
        'mean_time': mean_time,
        'std_time': std_time,
        'min_time': min_time,
        'max_time': max_time,
        'p95_time': p95_time,
        'meets_requirement': mean_time < 20
    }

def main():
    """
    Main training pipeline for the CNN-based spoken digit recognition system.
    """
    print("=" * 80)
    print("CNN-BASED SPOKEN DIGIT RECOGNITION TRAINING")
    print("=" * 80)
    
    # Configuration
    config = {
        'target_sr': 8000,
        'duration': 1.0,
        'n_mfcc': 20,
        'test_size': 0.2,
        'val_size': 0.1,
        'epochs': 50,
        'batch_size': 32,
        'use_class_weights': True,
        'use_focal_loss': False,
        'use_spec_augment': True
    }
    
    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Step 1: Load and split data
    print("\n" + "="*50)
    print("STEP 1: Loading and splitting data")
    print("="*50)
    
    data_loader = FSDDDataLoader(
        target_sr=config['target_sr'],
        duration=config['duration']
    )
    
    audio_data, labels = data_loader.load_dataset()
    splits = data_loader.stratified_split(
        audio_data, labels,
        test_size=config['test_size'],
        val_size=config['val_size']
    )
    
    # Step 2: Extract features
    print("\n" + "="*50)
    print("STEP 2: Extracting spectro-temporal features")
    print("="*50)
    
    feature_extractor = MFCCFeatureExtractor(
        target_sr=config['target_sr'],
        duration=config['duration'],
        n_mfcc=config['n_mfcc']
    )
    
    # Extract features for each split
    X_train, y_train = splits['train']
    X_val, y_val = splits['val']
    X_test, y_test = splits['test']
    
    print("Extracting training features...")
    X_train_features, y_train_features = feature_extractor.extract_features_with_augmentation(
        X_train, y_train, use_spec_augment=config['use_spec_augment']
    )
    
    print("Extracting validation features...")
    X_val_features, y_val_features = feature_extractor.extract_features_batch(X_val)
    
    print("Extracting test features...")
    X_test_features, y_test_features = feature_extractor.extract_features_batch(X_test)
    
    # Normalize features
    print("Normalizing features...")
    X_train_norm, mean, std = feature_extractor.normalize_features(X_train_features)
    X_val_norm, _, _ = feature_extractor.normalize_features(X_val_features, mean, std)
    X_test_norm, _, _ = feature_extractor.normalize_features(X_test_features, mean, std)
    
    print(f"Final feature shapes:")
    print(f"  Train: {X_train_norm.shape}")
    print(f"  Val:   {X_val_norm.shape}")
    print(f"  Test:  {X_test_norm.shape}")
    
    # Step 3: Build and train model
    print("\n" + "="*50)
    print("STEP 3: Building and training CNN model")
    print("="*50)
    
    input_shape = X_train_norm.shape[1:]  # (n_mfcc*3, time_frames)
    cnn_model = CompactCNN(
        input_shape=input_shape,
        num_classes=10,
        dropout_rate=0.25,
        l2_reg=1e-4
    )
    
    # Train the model
    start_time = time.time()
    history = cnn_model.train(
        X_train_norm, y_train_features,
        X_val_norm, y_val_features,
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        use_class_weights=config['use_class_weights'],
        use_focal_loss=config['use_focal_loss']
    )
    training_time = time.time() - start_time
    
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Step 4: Evaluate model
    print("\n" + "="*50)
    print("STEP 4: Evaluating CNN model")
    print("="*50)
    
    results = cnn_model.evaluate(X_test_norm, y_test_features)
    
    # Plot results
    plot_training_history(history)
    plot_confusion_matrix(results['confusion_matrix'])
    
    # Step 5: Benchmark inference speed
    print("\n" + "="*50)
    print("STEP 5: Benchmarking inference speed")
    print("="*50)
    
    speed_results = benchmark_inference_speed(cnn_model, X_test_norm)
    
    # Step 6: Save model and results
    print("\n" + "="*50)
    print("STEP 6: Saving model and results")
    print("="*50)
    
    cnn_model.save_model("spoken_digit_cnn_model.h5")
    
    # Save normalization parameters
    np.savez("normalization_params.npz", mean=mean, std=std)
    print("Normalization parameters saved to normalization_params.npz")
    
    # Print final summary
    print("\n" + "="*80)
    print("TRAINING COMPLETED - FINAL SUMMARY")
    print("="*80)
    print(f"Test Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    print(f"Test Precision: {results['precision']:.4f}")
    print(f"Test Recall: {results['recall']:.4f}")
    print(f"Test F1-Score: {results['f1_score']:.4f}")
    print(f"Mean Inference Time: {speed_results['mean_time']:.2f} ms")
    print(f"Training Time: {training_time:.2f} seconds")
    
    # Check if requirements are met
    accuracy_met = results['accuracy'] >= 0.97
    speed_met = speed_results['mean_time'] < 20
    
    print(f"\nRequirements Check:")
    print(f"✓ Accuracy ≥97%: {'PASS' if accuracy_met else 'FAIL'} ({results['accuracy']*100:.2f}%)")
    print(f"✓ Inference <20ms: {'PASS' if speed_met else 'FAIL'} ({speed_results['mean_time']:.2f}ms)")
    
    if accuracy_met and speed_met:
        print("\n🎉 ALL REQUIREMENTS MET! Model is ready for deployment.")
    else:
        print("\n⚠ Some requirements not met. Consider hyperparameter tuning.")
    
    return {
        'model': cnn_model,
        'results': results,
        'speed_results': speed_results,
        'history': history,
        'feature_extractor': feature_extractor,
        'normalization_params': {'mean': mean, 'std': std}
    }

if __name__ == "__main__":
    main()