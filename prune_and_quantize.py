#!/usr/bin/env python3
"""
Pruning and Quantization Script for Spoken Digit Recognition.

This script implements iterative magnitude-based weight pruning and
post-training dynamic INT8 quantization for model compression and speed.
"""

import os
import argparse
import time
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.quantization as quantization
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Tuple
import json

# Suppress warnings
warnings.filterwarnings('ignore')

# Import our modules
from model_student import create_student_model

class MagnitudePruner:
    """
    Iterative magnitude-based weight pruning.
    """
    
    def __init__(self, model: nn.Module, target_sparsity: float = 0.4):
        """
        Initialize the pruner.
        
        Args:
            model: Model to prune
            target_sparsity: Target global sparsity (0.0 to 1.0)
        """
        self.model = model
        self.target_sparsity = target_sparsity
        self.pruning_schedule = []
        self.current_sparsity = 0.0
        
    def get_current_sparsity(self) -> float:
        """Calculate current model sparsity."""
        total_params = 0
        zero_params = 0
        
        for name, param in self.model.named_parameters():
            if 'weight' in name and 'conv' in name.lower():
                total_params += param.numel()
                zero_params += (param == 0).sum().item()
        
        return zero_params / total_params if total_params > 0 else 0.0
    
    def iterative_prune(self, train_loader: DataLoader, val_loader: DataLoader,
                       device: torch.device, epochs_per_iter: int = 5,
                       pruning_steps: int = 8) -> Dict[str, Any]:
        """
        Perform iterative magnitude-based pruning.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            device: Device to train on
            epochs_per_iter: Epochs per pruning iteration
            pruning_steps: Number of pruning steps
            
        Returns:
            Pruning history
        """
        print(f"Starting iterative pruning to {self.target_sparsity*100:.1f}% sparsity...")
        
        history = {
            'sparsity': [],
            'accuracy': [],
            'parameters': []
        }
        
        # Calculate sparsity increment per step
        sparsity_increment = self.target_sparsity / pruning_steps
        
        for step in range(pruning_steps):
            current_target_sparsity = (step + 1) * sparsity_increment
            
            print(f"\nPruning step {step + 1}/{pruning_steps}")
            print(f"Target sparsity: {current_target_sparsity*100:.1f}%")
            
            # Prune the model
            self._prune_model(current_target_sparsity)
            
            # Fine-tune the pruned model
            self._fine_tune(train_loader, val_loader, device, epochs_per_iter)
            
            # Evaluate
            accuracy = self._evaluate(val_loader, device)
            current_sparsity = self.get_current_sparsity()
            param_count = sum(p.numel() for p in self.model.parameters())
            
            history['sparsity'].append(current_sparsity)
            history['accuracy'].append(accuracy)
            history['parameters'].append(param_count)
            
            print(f"Current sparsity: {current_sparsity*100:.1f}%")
            print(f"Accuracy: {accuracy:.2f}%")
            print(f"Parameters: {param_count:,}")
            
            # Early stopping if accuracy drops too much
            if accuracy < 95.0:
                print("Accuracy dropped below 95%, stopping pruning")
                break
        
        return history
    
    def _prune_model(self, target_sparsity: float):
        """Prune model to target sparsity using magnitude-based pruning."""
        # Collect all weights from convolutional layers
        weights = []
        for name, param in self.model.named_parameters():
            if 'weight' in name and 'conv' in name.lower():
                weights.append(param.data.view(-1))
        
        if not weights:
            return
        
        # Concatenate all weights
        all_weights = torch.cat(weights)
        
        # Calculate threshold for target sparsity
        threshold = torch.quantile(torch.abs(all_weights), target_sparsity)
        
        # Apply pruning mask
        for name, param in self.model.named_parameters():
            if 'weight' in name and 'conv' in name.lower():
                mask = torch.abs(param.data) > threshold
                param.data *= mask
    
    def _fine_tune(self, train_loader: DataLoader, val_loader: DataLoader,
                   device: torch.device, epochs: int):
        """Fine-tune the pruned model."""
        self.model.train()
        optimizer = optim.Adam(self.model.parameters(), lr=0.0001)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(epochs):
            # Training
            for batch_idx, (data, targets) in enumerate(train_loader):
                data, targets = data.to(device), targets.to(device)
                
                optimizer.zero_grad()
                outputs = self.model(data)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
            
            # Validation
            if epoch % 2 == 0:
                accuracy = self._evaluate(val_loader, device)
                print(f"  Epoch {epoch + 1}/{epochs}, Accuracy: {accuracy:.2f}%")
    
    def _evaluate(self, val_loader: DataLoader, device: torch.device) -> float:
        """Evaluate model accuracy."""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, targets in val_loader:
                data, targets = data.to(device), targets.to(device)
                outputs = self.model(data)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        return 100. * correct / total

class QuantizedModel(nn.Module):
    """
    Wrapper for quantized model with benchmarking capabilities.
    """
    
    def __init__(self, model: nn.Module):
        super(QuantizedModel, self).__init__()
        self.model = model
        self.quantized = False
    
    def quantize_dynamic(self):
        """Apply dynamic INT8 quantization."""
        print("Applying dynamic INT8 quantization...")
        
        # Quantize the model
        self.model = quantization.quantize_dynamic(
            self.model, {nn.Linear, nn.Conv2d}, dtype=torch.qint8
        )
        
        self.quantized = True
        print("Dynamic quantization applied successfully")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.model(x)
    
    def benchmark_inference(self, input_tensor: torch.Tensor, 
                           num_runs: int = 100) -> Dict[str, float]:
        """
        Benchmark inference speed.
        
        Args:
            input_tensor: Input tensor for benchmarking
            num_runs: Number of inference runs
            
        Returns:
            Timing statistics
        """
        self.eval()
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = self.forward(input_tensor)
        
        # Benchmark
        times = []
        with torch.no_grad():
            for _ in range(num_runs):
                start_time = time.time()
                _ = self.forward(input_tensor)
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                end_time = time.time()
                times.append((end_time - start_time) * 1000)  # Convert to ms
        
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        
        return {
            'avg_inference_time_ms': avg_time,
            'min_inference_time_ms': min_time,
            'max_inference_time_ms': max_time,
            'std_inference_time_ms': torch.tensor(times).std().item()
        }

def load_and_preprocess_data() -> Tuple[torch.Tensor, torch.Tensor]:
    """Load synthetic data for demonstration."""
    print("Loading synthetic data for pruning/quantization demo...")
    
    num_samples = 1000
    input_shape = (1, 60, 80)
    
    # Generate synthetic MFCC features
    features = torch.randn(num_samples, *input_shape)
    labels = torch.randint(0, 10, (num_samples,))
    
    print(f"Loaded {num_samples} samples with shape {input_shape}")
    return features, labels

def plot_pruning_history(history: Dict[str, Any], save_path: str = "pruning_history.png"):
    """Plot pruning history."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot sparsity vs accuracy
    ax1.plot([s*100 for s in history['sparsity']], history['accuracy'], 'bo-')
    ax1.set_title('Accuracy vs Sparsity')
    ax1.set_xlabel('Sparsity (%)')
    ax1.set_ylabel('Accuracy (%)')
    ax1.grid(True)
    
    # Plot sparsity vs parameters
    ax2.plot([s*100 for s in history['sparsity']], history['parameters'], 'ro-')
    ax2.set_title('Parameters vs Sparsity')
    ax2.set_xlabel('Sparsity (%)')
    ax2.set_ylabel('Parameters')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Pruning history plot saved to {save_path}")

def main():
    """Main pruning and quantization function."""
    parser = argparse.ArgumentParser(description='Model Pruning and Quantization')
    
    # Model parameters
    parser.add_argument('--input_shape', type=int, nargs=3, default=[1, 60, 80],
                       help='Input shape (channels, height, width)')
    parser.add_argument('--num_classes', type=int, default=10,
                       help='Number of output classes')
    
    # Pruning parameters
    parser.add_argument('--target_sparsity', type=float, default=0.4,
                       help='Target global sparsity (0.0 to 1.0)')
    parser.add_argument('--pruning_steps', type=int, default=8,
                       help='Number of pruning steps')
    parser.add_argument('--epochs_per_iter', type=int, default=5,
                       help='Epochs per pruning iteration')
    
    # Benchmarking parameters
    parser.add_argument('--benchmark_runs', type=int, default=100,
                       help='Number of runs for benchmarking')
    
    # System parameters
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cpu, cuda)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Load data
    features, labels = load_and_preprocess_data()
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        features, labels, test_size=0.2, random_state=args.seed, stratify=labels
    )
    
    # Create data loaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Create model
    input_shape = tuple(args.input_shape)
    model = create_student_model(input_shape=input_shape, num_classes=args.num_classes)
    model = model.to(device)
    
    # Benchmark original model
    print("\n=== ORIGINAL MODEL BENCHMARK ===")
    original_params = sum(p.numel() for p in model.parameters())
    print(f"Original parameters: {original_params:,}")
    
    # Create sample input for benchmarking
    sample_input = torch.randn(1, *input_shape).to(device)
    
    # Benchmark original model
    original_stats = model.benchmark_inference(sample_input, args.benchmark_runs)
    print(f"Original model inference time: {original_stats['avg_inference_time_ms']:.2f} ms")
    
    # Prune the model
    print("\n=== PRUNING MODEL ===")
    pruner = MagnitudePruner(model, target_sparsity=args.target_sparsity)
    pruning_history = pruner.iterative_prune(
        train_loader, val_loader, device, 
        epochs_per_iter=args.epochs_per_iter,
        pruning_steps=args.pruning_steps
    )
    
    # Benchmark pruned model
    print("\n=== PRUNED MODEL BENCHMARK ===")
    pruned_params = sum(p.numel() for p in model.parameters())
    print(f"Pruned parameters: {pruned_params:,}")
    print(f"Parameter reduction: {((original_params - pruned_params) / original_params * 100):.1f}%")
    
    pruned_stats = model.benchmark_inference(sample_input, args.benchmark_runs)
    print(f"Pruned model inference time: {pruned_stats['avg_inference_time_ms']:.2f} ms")
    
    # Quantize the model
    print("\n=== QUANTIZING MODEL ===")
    quantized_model = QuantizedModel(model)
    quantized_model.quantize_dynamic()
    
    # Benchmark quantized model
    print("\n=== QUANTIZED MODEL BENCHMARK ===")
    quantized_stats = quantized_model.benchmark_inference(sample_input, args.benchmark_runs)
    print(f"Quantized model inference time: {quantized_stats['avg_inference_time_ms']:.2f} ms")
    
    # Print final comparison
    print("\n=== FINAL COMPARISON ===")
    print(f"{'Metric':<20} {'Original':<15} {'Pruned':<15} {'Quantized':<15}")
    print("-" * 65)
    print(f"{'Parameters':<20} {original_params:<15,} {pruned_params:<15,} {'N/A':<15}")
    print(f"{'Inference (ms)':<20} {original_stats['avg_inference_time_ms']:<15.2f} {pruned_stats['avg_inference_time_ms']:<15.2f} {quantized_stats['avg_inference_time_ms']:<15.2f}")
    
    # Calculate speedup
    original_time = original_stats['avg_inference_time_ms']
    pruned_time = pruned_stats['avg_inference_time_ms']
    quantized_time = quantized_stats['avg_inference_time_ms']
    
    print(f"\nSpeedup vs Original:")
    print(f"  Pruned: {original_time/pruned_time:.2f}x")
    print(f"  Quantized: {original_time/quantized_time:.2f}x")
    
    # Check target constraints
    print(f"\n=== TARGET CONSTRAINT CHECK ===")
    print(f"Target: ≤15ms CPU latency")
    print(f"Quantized model: {quantized_time:.2f}ms")
    
    if quantized_time <= 15.0:
        print("✓ Target latency achieved!")
    else:
        print("⚠️  Target latency not achieved")
    
    # Plot pruning history
    plot_pruning_history(pruning_history)
    
    # Save results
    results = {
        'original': {
            'parameters': original_params,
            'inference_time_ms': original_stats['avg_inference_time_ms']
        },
        'pruned': {
            'parameters': pruned_params,
            'inference_time_ms': pruned_stats['avg_inference_time_ms'],
            'sparsity': pruning_history['sparsity'][-1] if pruning_history['sparsity'] else 0.0
        },
        'quantized': {
            'inference_time_ms': quantized_stats['avg_inference_time_ms']
        },
        'pruning_history': pruning_history
    }
    
    with open('pruning_quantization_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save models
    torch.save(model.state_dict(), 'model_pruned.pth')
    torch.save(quantized_model.state_dict(), 'model_quantized.pth')
    
    print("\nModels saved:")
    print("  - model_pruned.pth")
    print("  - model_quantized.pth")
    print("  - pruning_quantization_results.json")
    
    print("\nPruning and quantization completed!")

if __name__ == '__main__':
    main()