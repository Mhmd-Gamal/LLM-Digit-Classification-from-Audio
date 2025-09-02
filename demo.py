#!/usr/bin/env python3
"""
Demo script for the Spoken Digit Recognition System

This script provides a quick demonstration of the system's capabilities
without requiring the full training process.
"""

import numpy as np
import matplotlib.pyplot as plt
from spoken_digit_recognition import SpokenDigitRecognizer

def demo_with_sample_data():
    """
    Demonstrate the system with synthetic sample data.
    """
    print("=" * 50)
    print("SPOKEN DIGIT RECOGNITION DEMO")
    print("=" * 50)
    
    # Initialize recognizer
    recognizer = SpokenDigitRecognizer()
    
    # Create synthetic audio data (simulating FSDD samples)
    print("\n1. Creating synthetic audio data...")
    np.random.seed(42)
    
    # Generate synthetic audio samples for each digit
    n_samples_per_digit = 50
    audio_data = []
    labels = []
    
    for digit in range(10):
        for _ in range(n_samples_per_digit):
            # Create synthetic audio with different characteristics per digit
            duration = 1.0
            sr = 8000
            t = np.linspace(0, duration, int(sr * duration))
            
            # Different frequency patterns for different digits
            base_freq = 200 + digit * 50  # Vary base frequency
            audio = np.sin(2 * np.pi * base_freq * t)
            
            # Add some harmonics and noise to make it more realistic
            audio += 0.3 * np.sin(2 * np.pi * base_freq * 2 * t)
            audio += 0.1 * np.random.normal(0, 0.1, len(t))
            
            # Normalize
            audio = audio / np.max(np.abs(audio))
            
            audio_data.append(audio)
            labels.append(digit)
    
    audio_data = np.array(audio_data)
    labels = np.array(labels)
    
    print(f"Generated {len(audio_data)} synthetic audio samples")
    print(f"Label distribution: {np.bincount(labels)}")
    
    # Extract features
    print("\n2. Extracting MFCC features...")
    features, labels = recognizer.extract_features(audio_data, labels, augment=False)
    
    # Train model
    print("\n3. Training model...")
    history = recognizer.train(features, labels, epochs=20)
    
    # Evaluate
    print("\n4. Evaluating model...")
    results = recognizer.evaluate()
    
    # Test responsiveness
    print("\n5. Testing responsiveness...")
    test_audio = audio_data[0]
    pred, conf, time_ms = recognizer.predict_single(test_audio)
    print(f"Single prediction time: {time_ms:.2f} ms")
    print(f"Predicted digit: {pred}, Confidence: {conf:.3f}")
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n" + "=" * 50)
    print("DEMO COMPLETED SUCCESSFULLY!")
    print("=" * 50)
    
    return recognizer, results

def interactive_demo():
    """
    Interactive demo with user input.
    """
    print("\n" + "=" * 50)
    print("INTERACTIVE DEMO")
    print("=" * 50)
    
    try:
        # Try to load a pre-trained model
        recognizer = SpokenDigitRecognizer()
        recognizer.load_model("spoken_digit_model.h5")
        print("Loaded pre-trained model!")
        
        while True:
            print("\nOptions:")
            print("1. Test with microphone")
            print("2. Exit")
            
            choice = input("Enter your choice (1-2): ").strip()
            
            if choice == '1':
                result = recognizer.predict_from_microphone()
                if result:
                    pred, conf, time_ms = result
                    print(f"\n🎯 Result: Digit {pred} (Confidence: {conf:.1%}, Time: {time_ms:.1f}ms)")
            
            elif choice == '2':
                print("Goodbye!")
                break
            
            else:
                print("Invalid choice. Please try again.")
                
    except FileNotFoundError:
        print("No pre-trained model found. Please run the main script first to train a model.")
    except Exception as e:
        print(f"Error in interactive demo: {e}")

if __name__ == "__main__":
    # Run the demo
    recognizer, results = demo_with_sample_data()
    
    # Ask if user wants to try interactive demo
    try:
        response = input("\nWould you like to try the interactive microphone demo? (y/n): ").lower()
        if response == 'y':
            interactive_demo()
    except KeyboardInterrupt:
        print("\nDemo interrupted by user.")
    except Exception as e:
        print(f"Interactive demo not available: {e}")