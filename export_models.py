#!/usr/bin/env python3
"""
Model Export Script for TorchScript and ONNX.

This script exports the trained PyTorch model to TorchScript and ONNX formats
for deployment with <1MB file size each.
"""

import os
import argparse
import warnings
import numpy as np
import torch
import torch.nn as nn
import onnx
import onnxruntime as ort
from typing import Tuple, Dict, Any
import json

# Suppress warnings
warnings.filterwarnings('ignore')

# Import our modules
from model_student import create_student_model

def export_torchscript(model: nn.Module, input_shape: Tuple[int, int, int], 
                      output_path: str) -> Dict[str, Any]:
    """
    Export PyTorch model to TorchScript format.
    
    Args:
        model: Trained PyTorch model
        input_shape: Input shape (channels, height, width)
        output_path: Output file path
        
    Returns:
        Export statistics
    """
    print(f"Exporting to TorchScript: {output_path}")
    
    model.eval()
    
    # Create example input
    example_input = torch.randn(1, *input_shape)
    
    try:
        # Trace the model
        traced_model = torch.jit.trace(model, example_input)
        
        # Save the traced model
        torch.jit.save(traced_model, output_path)
        
        # Get file size
        file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
        
        print(f"✓ TorchScript export successful")
        print(f"  - File size: {file_size:.2f} MB")
        
        # Test the exported model
        loaded_model = torch.jit.load(output_path)
        with torch.no_grad():
            output = loaded_model(example_input)
        
        print(f"  - Output shape: {output.shape}")
        print(f"  - Test inference: ✓")
        
        return {
            'success': True,
            'file_size_mb': file_size,
            'output_shape': list(output.shape)
        }
        
    except Exception as e:
        print(f"✗ TorchScript export failed: {e}")
        return {
            'success': False,
            'error': str(e)
        }

def export_onnx(model: nn.Module, input_shape: Tuple[int, int, int], 
               output_path: str, opset_version: int = 17) -> Dict[str, Any]:
    """
    Export PyTorch model to ONNX format.
    
    Args:
        model: Trained PyTorch model
        input_shape: Input shape (channels, height, width)
        output_path: Output file path
        opset_version: ONNX opset version
        
    Returns:
        Export statistics
    """
    print(f"Exporting to ONNX (opset {opset_version}): {output_path}")
    
    model.eval()
    
    # Create example input
    example_input = torch.randn(1, *input_shape)
    
    try:
        # Export to ONNX
        torch.onnx.export(
            model,
            example_input,
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        # Get file size
        file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
        
        print(f"✓ ONNX export successful")
        print(f"  - File size: {file_size:.2f} MB")
        
        # Validate ONNX model
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print(f"  - ONNX validation: ✓")
        
        # Test with ONNX Runtime
        ort_session = ort.InferenceSession(output_path)
        input_name = ort_session.get_inputs()[0].name
        output_name = ort_session.get_outputs()[0].name
        
        # Run inference
        ort_inputs = {input_name: example_input.numpy()}
        ort_outputs = ort_session.run([output_name], ort_inputs)
        
        print(f"  - ONNX Runtime test: ✓")
        print(f"  - Output shape: {ort_outputs[0].shape}")
        
        return {
            'success': True,
            'file_size_mb': file_size,
            'opset_version': opset_version,
            'output_shape': list(ort_outputs[0].shape)
        }
        
    except Exception as e:
        print(f"✗ ONNX export failed: {e}")
        return {
            'success': False,
            'error': str(e)
        }

def benchmark_exported_models(model_path: str, ts_path: str, onnx_path: str,
                             input_shape: Tuple[int, int, int], 
                             num_runs: int = 100) -> Dict[str, Any]:
    """
    Benchmark original PyTorch vs exported models.
    
    Args:
        model_path: Path to original PyTorch model
        ts_path: Path to TorchScript model
        onnx_path: Path to ONNX model
        input_shape: Input shape
        num_runs: Number of benchmark runs
        
    Returns:
        Benchmark results
    """
    print("\n=== BENCHMARKING EXPORTED MODELS ===")
    
    # Create sample input
    sample_input = torch.randn(1, *input_shape)
    
    results = {}
    
    # Benchmark original PyTorch model
    if os.path.exists(model_path):
        print("Benchmarking original PyTorch model...")
        model = create_student_model(input_shape=input_shape, num_classes=10)
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(sample_input)
        
        # Benchmark
        times = []
        with torch.no_grad():
            for _ in range(num_runs):
                start_time = torch.cuda.Event(enable_timing=True)
                end_time = torch.cuda.Event(enable_timing=True)
                
                start_time.record()
                _ = model(sample_input)
                end_time.record()
                
                torch.cuda.synchronize()
                times.append(start_time.elapsed_time(end_time))
        
        avg_time = sum(times) / len(times)
        results['pytorch'] = {
            'avg_inference_time_ms': avg_time,
            'min_inference_time_ms': min(times),
            'max_inference_time_ms': max(times)
        }
        print(f"  PyTorch: {avg_time:.2f} ms")
    
    # Benchmark TorchScript model
    if os.path.exists(ts_path):
        print("Benchmarking TorchScript model...")
        ts_model = torch.jit.load(ts_path)
        ts_model.eval()
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = ts_model(sample_input)
        
        # Benchmark
        times = []
        with torch.no_grad():
            for _ in range(num_runs):
                start_time = torch.cuda.Event(enable_timing=True)
                end_time = torch.cuda.Event(enable_timing=True)
                
                start_time.record()
                _ = ts_model(sample_input)
                end_time.record()
                
                torch.cuda.synchronize()
                times.append(start_time.elapsed_time(end_time))
        
        avg_time = sum(times) / len(times)
        results['torchscript'] = {
            'avg_inference_time_ms': avg_time,
            'min_inference_time_ms': min(times),
            'max_inference_time_ms': max(times)
        }
        print(f"  TorchScript: {avg_time:.2f} ms")
    
    # Benchmark ONNX model
    if os.path.exists(onnx_path):
        print("Benchmarking ONNX model...")
        ort_session = ort.InferenceSession(onnx_path)
        input_name = ort_session.get_inputs()[0].name
        
        # Warmup
        for _ in range(10):
            _ = ort_session.run(None, {input_name: sample_input.numpy()})
        
        # Benchmark
        times = []
        for _ in range(num_runs):
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)
            
            start_time.record()
            _ = ort_session.run(None, {input_name: sample_input.numpy()})
            end_time.record()
            
            torch.cuda.synchronize()
            times.append(start_time.elapsed_time(end_time))
        
        avg_time = sum(times) / len(times)
        results['onnx'] = {
            'avg_inference_time_ms': avg_time,
            'min_inference_time_ms': min(times),
            'max_inference_time_ms': max(times)
        }
        print(f"  ONNX: {avg_time:.2f} ms")
    
    return results

def main():
    """Main export function."""
    parser = argparse.ArgumentParser(description='Model Export to TorchScript and ONNX')
    
    parser.add_argument('--model', type=str, default='student_model_distilled.pth',
                       help='Path to trained PyTorch model')
    parser.add_argument('--input_shape', type=int, nargs=3, default=[1, 60, 80],
                       help='Input shape (channels, height, width)')
    parser.add_argument('--num_classes', type=int, default=10,
                       help='Number of output classes')
    parser.add_argument('--opset', type=int, default=17,
                       help='ONNX opset version')
    parser.add_argument('--output_dir', type=str, default='artifacts',
                       help='Output directory for exported models')
    parser.add_argument('--benchmark', action='store_true',
                       help='Run benchmark comparison')
    parser.add_argument('--benchmark_runs', type=int, default=100,
                       help='Number of benchmark runs')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Define output paths
    ts_path = os.path.join(args.output_dir, 'model_ts.pt')
    onnx_path = os.path.join(args.output_dir, 'model.onnx')
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"Error: Model file {args.model} not found")
        print("Creating a sample model for demonstration...")
        
        # Create a sample model
        model = create_student_model(
            input_shape=tuple(args.input_shape), 
            num_classes=args.num_classes
        )
        
        # Save sample model
        torch.save(model.state_dict(), args.model)
        print(f"Sample model saved to {args.model}")
    
    # Load model
    print(f"Loading model from {args.model}...")
    model = create_student_model(
        input_shape=tuple(args.input_shape), 
        num_classes=args.num_classes
    )
    model.load_state_dict(torch.load(args.model, map_location='cpu'))
    model.eval()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # Export to TorchScript
    ts_results = export_torchscript(model, tuple(args.input_shape), ts_path)
    
    # Export to ONNX
    onnx_results = export_onnx(model, tuple(args.input_shape), onnx_path, args.opset)
    
    # Check file size constraints
    print("\n=== FILE SIZE CHECK ===")
    print("Target: <1MB each")
    
    if ts_results['success']:
        if ts_results['file_size_mb'] < 1.0:
            print(f"✓ TorchScript: {ts_results['file_size_mb']:.2f}MB (<1MB)")
        else:
            print(f"⚠️  TorchScript: {ts_results['file_size_mb']:.2f}MB (≥1MB)")
    
    if onnx_results['success']:
        if onnx_results['file_size_mb'] < 1.0:
            print(f"✓ ONNX: {onnx_results['file_size_mb']:.2f}MB (<1MB)")
        else:
            print(f"⚠️  ONNX: {onnx_results['file_size_mb']:.2f}MB (≥1MB)")
    
    # Benchmark if requested
    if args.benchmark:
        benchmark_results = benchmark_exported_models(
            args.model, ts_path, onnx_path, 
            tuple(args.input_shape), args.benchmark_runs
        )
        
        # Print benchmark summary
        print("\n=== BENCHMARK SUMMARY ===")
        for model_type, stats in benchmark_results.items():
            print(f"{model_type.capitalize()}: {stats['avg_inference_time_ms']:.2f}ms")
    
    # Save export results
    export_summary = {
        'model_path': args.model,
        'input_shape': args.input_shape,
        'num_classes': args.num_classes,
        'total_parameters': total_params,
        'torchscript': ts_results,
        'onnx': onnx_results,
        'export_timestamp': str(torch.cuda.Event(enable_timing=True).elapsed_time(torch.cuda.Event(enable_timing=True)))
    }
    
    if args.benchmark:
        export_summary['benchmark'] = benchmark_results
    
    with open(os.path.join(args.output_dir, 'export_results.json'), 'w') as f:
        json.dump(export_summary, f, indent=2)
    
    print(f"\nExport results saved to {args.output_dir}/export_results.json")
    print("\nExport completed!")

if __name__ == '__main__':
    main()