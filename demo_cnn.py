#!/usr/bin/env python3
"""
Demo script for the CNN-based spoken digit recognition system.

This script provides a simple interface to test the trained CNN model
with real-time microphone input and voice activity detection.
"""

import os
import sys
import time
import warnings
from pathlib import Path

# Suppress warnings
warnings.filterwarnings('ignore')

# Import our modules
from inference import CNNInferenceEngine

def check_model_files():
    """Check if required model files exist."""
    model_path = "spoken_digit_cnn_model.h5"
    norm_path = "normalization_params.npz"
    
    if not os.path.exists(model_path):
        print(f"❌ Model file {model_path} not found!")
        print("Please run 'python train.py' first to train the CNN model.")
        return False
    
    if not os.path.exists(norm_path):
        print(f"❌ Normalization file {norm_path} not found!")
        print("Please run 'python train.py' first to train the CNN model.")
        return False
    
    print("✅ Model files found!")
    return True

def demo_inference():
    """Run the CNN inference demo."""
    print("=" * 60)
    print("CNN SPOKEN DIGIT RECOGNITION DEMO")
    print("=" * 60)
    
    # Check if model files exist
    if not check_model_files():
        return
    
    # Initialize inference engine
    print("\n🚀 Initializing CNN inference engine...")
    try:
        inference_engine = CNNInferenceEngine(
            model_path="spoken_digit_cnn_model.h5",
            normalization_path="normalization_params.npz"
        )
        print("✅ Inference engine initialized successfully!")
    except Exception as e:
        print(f"❌ Failed to initialize inference engine: {e}")
        return
    
    # Benchmark inference speed
    print("\n⏱️  Benchmarking inference speed...")
    try:
        speed_results = inference_engine.benchmark_inference(num_samples=20)
        print(f"✅ Mean inference time: {speed_results['mean_total_time']:.2f}ms")
        
        if speed_results['mean_total_time'] < 20:
            print("🎉 Inference speed meets <20ms requirement!")
        else:
            print("⚠️  Inference speed exceeds 20ms requirement")
    except Exception as e:
        print(f"⚠️  Speed benchmark failed: {e}")
    
    # Interactive microphone demo
    print("\n🎤 Interactive microphone demo")
    print("=" * 40)
    print("Instructions:")
    print("1. Make sure your microphone is working")
    print("2. Speak a digit (0-9) clearly")
    print("3. The system will detect voice activity and predict the digit")
    print("4. Type 'quit' to exit")
    print("=" * 40)
    
    try:
        while True:
            response = input("\nPress Enter to record (or 'quit' to exit): ").strip().lower()
            
            if response == 'quit':
                break
            
            print("\n🎙️  Recording 2 seconds of audio...")
            print("Speak a digit (0-9) now!")
            
            result = inference_engine.predict_from_microphone(duration=2.0, use_vad=True)
            
            if result:
                prediction, confidence, inference_time, metadata = result
                
                if prediction == -1:
                    print(f"❌ No voice activity detected")
                    print(f"   Energy level: {metadata['energy']:.4f}")
                    print(f"   Try speaking louder or closer to the microphone")
                else:
                    print(f"🎯 Predicted digit: {prediction}")
                    print(f"📊 Confidence: {confidence:.3f}")
                    print(f"⏱️  Inference time: {inference_time:.2f}ms")
                    print(f"🔊 Voice detected: {metadata['has_voice']}")
                    print(f"📈 Energy level: {metadata['energy']:.4f}")
                    
                    # Provide feedback based on confidence
                    if confidence > 0.9:
                        print("🎉 High confidence prediction!")
                    elif confidence > 0.7:
                        print("✅ Good confidence prediction")
                    else:
                        print("⚠️  Low confidence - try speaking more clearly")
            else:
                print("❌ Recording failed. Please check your microphone.")
    
    except KeyboardInterrupt:
        print("\n\n👋 Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
    
    print("\n" + "=" * 60)
    print("DEMO COMPLETED")
    print("=" * 60)

def main():
    """Main demo function."""
    try:
        demo_inference()
    except Exception as e:
        print(f"❌ Demo failed with error: {e}")
        print("Please ensure all dependencies are installed and model files exist.")

if __name__ == "__main__":
    main()