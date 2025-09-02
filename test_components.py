#!/usr/bin/env python3
"""
Test script to verify all components work correctly.
"""

import os
import sys
import torch
import numpy as np
from model_student import create_student_model

def test_model_creation():
    """Test student model creation and parameter count."""
    print("Testing model creation...")
    
    model = create_student_model(input_shape=(1, 60, 80), num_classes=10)
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"Model parameters: {total_params:,}")
    print(f"Target: ≤90,000 parameters")
    
    if total_params <= 90000:
        print("✓ Model within parameter limit")
        return True
    else:
        print("✗ Model exceeds parameter limit")
        return False

def test_model_inference():
    """Test model inference with sample data."""
    print("\nTesting model inference...")
    
    model = create_student_model(input_shape=(1, 60, 80), num_classes=10)
    model.eval()
    
    # Create sample input
    sample_input = torch.randn(1, 1, 60, 80)
    
    # Test inference
    with torch.no_grad():
        output = model(sample_input)
        predicted = torch.argmax(output, dim=1).item()
        confidence = torch.softmax(output, dim=1).max().item()
    
    print(f"Output shape: {output.shape}")
    print(f"Predicted digit: {predicted}")
    print(f"Confidence: {confidence:.3f}")
    print("✓ Model inference working")

def test_benchmark():
    """Test model benchmarking."""
    print("\nTesting model benchmarking...")
    
    model = create_student_model(input_shape=(1, 60, 80), num_classes=10)
    sample_input = torch.randn(1, 1, 60, 80)
    
    stats = model.benchmark_inference(sample_input, num_runs=10)
    
    print(f"Average inference time: {stats['avg_inference_time_ms']:.2f} ms")
    print(f"Target: ≤15ms")
    
    if stats['avg_inference_time_ms'] <= 15.0:
        print("✓ Inference time within target")
        return True
    else:
        print("✗ Inference time exceeds target")
        return False

def test_file_structure():
    """Test that all required files exist."""
    print("\nTesting file structure...")
    
    required_files = [
        'model_student.py',
        'train_distill.py',
        'prune_and_quantize.py',
        'export_models.py',
        'infer.py',
        'requirements.txt',
        'README.md'
    ]
    
    required_dirs = [
        'artifacts',
        'logs',
        'notebooks'
    ]
    
    all_good = True
    
    for file in required_files:
        if os.path.exists(file):
            print(f"✓ {file}")
        else:
            print(f"✗ {file} (missing)")
            all_good = False
    
    for dir in required_dirs:
        if os.path.exists(dir):
            print(f"✓ {dir}/")
        else:
            print(f"✗ {dir}/ (missing)")
            all_good = False
    
    return all_good

def main():
    """Run all tests."""
    print("LLM-Digit-Classification-from-Audio - Component Test")
    print("=" * 50)
    
    tests = [
        test_model_creation,
        test_model_inference,
        test_benchmark,
        test_file_structure
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"✗ Test failed with error: {e}")
            results.append(False)
    
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! System ready for use.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the issues above.")
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)