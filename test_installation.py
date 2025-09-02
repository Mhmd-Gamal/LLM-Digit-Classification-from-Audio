#!/usr/bin/env python3
"""
Test script to verify installation and basic functionality
of the Spoken Digit Recognition System.
"""

import sys
import importlib
import numpy as np

def test_imports():
    """Test if all required packages can be imported."""
    print("Testing package imports...")
    
    required_packages = [
        'numpy', 'pandas', 'matplotlib', 'seaborn',
        'librosa', 'sklearn', 'tensorflow', 'sounddevice',
        'datasets', 'huggingface_hub'
    ]
    
    failed_imports = []
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {package}")
        except ImportError as e:
            print(f"❌ {package}: {e}")
            failed_imports.append(package)
    
    if failed_imports:
        print(f"\n❌ Failed to import: {failed_imports}")
        print("Please install missing packages with: pip install -r requirements.txt")
        return False
    else:
        print("\n✅ All packages imported successfully!")
        return True

def test_audio_processing():
    """Test basic audio processing functionality."""
    print("\nTesting audio processing...")
    
    try:
        import librosa
        import numpy as np
        
        # Create synthetic audio
        duration = 1.0
        sr = 8000
        t = np.linspace(0, duration, int(sr * duration))
        audio = np.sin(2 * np.pi * 440 * t)  # 440 Hz tone
        
        # Test MFCC extraction
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)
        
        # Test feature aggregation
        mfcc_mean = np.mean(mfccs, axis=1)
        mfcc_std = np.std(mfccs, axis=1)
        features = np.concatenate([mfcc_mean, mfcc_std])
        
        print(f"✅ Audio processing successful")
        print(f"   - MFCC shape: {mfccs.shape}")
        print(f"   - Feature vector shape: {features.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Audio processing failed: {e}")
        return False

def test_model_creation():
    """Test neural network model creation."""
    print("\nTesting model creation...")
    
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, Dropout
        
        # Create a simple model
        model = Sequential([
            Dense(64, activation='relu', input_shape=(40,)),
            Dropout(0.25),
            Dense(32, activation='relu'),
            Dense(10, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Test with dummy data
        X_dummy = np.random.random((10, 40))
        y_dummy = np.random.randint(0, 10, 10)
        
        # Test prediction
        predictions = model.predict(X_dummy, verbose=0)
        
        print(f"✅ Model creation successful")
        print(f"   - Model parameters: {model.count_params()}")
        print(f"   - Prediction shape: {predictions.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        return False

def test_microphone():
    """Test microphone functionality (optional)."""
    print("\nTesting microphone access...")
    
    try:
        import sounddevice as sd
        
        # Test if we can query audio devices
        devices = sd.query_devices()
        print(f"✅ Microphone access available")
        print(f"   - Found {len(devices)} audio devices")
        
        # Test if we can get default input device
        default_input = sd.default.device[0]
        print(f"   - Default input device: {default_input}")
        
        return True
        
    except Exception as e:
        print(f"⚠️  Microphone access limited: {e}")
        print("   - This is optional and won't affect core functionality")
        return False

def test_dataset_loading():
    """Test dataset loading capability."""
    print("\nTesting dataset loading...")
    
    try:
        from datasets import load_dataset
        
        # Test if we can access Hugging Face datasets
        # This is a lightweight test that doesn't actually download FSDD
        print("✅ Dataset loading capability available")
        print("   - Hugging Face datasets library working")
        
        return True
        
    except Exception as e:
        print(f"❌ Dataset loading failed: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("SPOKEN DIGIT RECOGNITION - INSTALLATION TEST")
    print("=" * 60)
    
    tests = [
        ("Package Imports", test_imports),
        ("Audio Processing", test_audio_processing),
        ("Model Creation", test_model_creation),
        ("Dataset Loading", test_dataset_loading),
        ("Microphone Access", test_microphone),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:.<40} {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! The system is ready to use.")
        print("You can now run: python spoken_digit_recognition.py")
    elif passed >= total - 1:  # Allow microphone test to fail
        print("\n✅ Core functionality is working!")
        print("You can run the main script, but microphone demo may not work.")
    else:
        print("\n❌ Some core tests failed. Please check the installation.")
        print("Try: pip install -r requirements.txt")

if __name__ == "__main__":
    main()