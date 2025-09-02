#!/usr/bin/env python3
"""
Simple test to verify the core functionality works.
"""

import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tensorflow as tf

def test_basic_functionality():
    """Test basic functionality without full dataset loading."""
    print("Testing basic functionality...")
    
    # Test 1: Audio processing
    print("1. Testing audio processing...")
    duration = 1.0
    sr = 8000
    t = np.linspace(0, duration, int(sr * duration))
    
    # Create synthetic audio for each digit
    audio_data = []
    labels = []
    
    for digit in range(10):
        for _ in range(20):  # 20 samples per digit
            # Different frequency patterns for different digits
            base_freq = 200 + digit * 50
            audio = np.sin(2 * np.pi * base_freq * t)
            audio += 0.3 * np.sin(2 * np.pi * base_freq * 2 * t)
            audio += 0.1 * np.random.normal(0, 0.1, len(t))
            audio = audio / np.max(np.abs(audio))
            
            audio_data.append(audio)
            labels.append(digit)
    
    audio_data = np.array(audio_data)
    labels = np.array(labels)
    
    print(f"   Created {len(audio_data)} synthetic audio samples")
    
    # Test 2: Feature extraction
    print("2. Testing feature extraction...")
    features_list = []
    
    for audio in audio_data:
        # Extract MFCCs
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)
        
        # Extract mean and std
        mfcc_mean = np.mean(mfccs, axis=1)
        mfcc_std = np.std(mfccs, axis=1)
        features = np.concatenate([mfcc_mean, mfcc_std])
        
        features_list.append(features)
    
    features = np.array(features_list)
    print(f"   Extracted features shape: {features.shape}")
    
    # Test 3: Model creation and training
    print("3. Testing model creation and training...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Create simple model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(40,)),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"   Model parameters: {model.count_params()}")
    
    # Train model
    history = model.fit(
        X_train, y_train,
        epochs=5,  # Quick training
        batch_size=16,
        validation_data=(X_test, y_test),
        verbose=0
    )
    
    # Test 4: Evaluation
    print("4. Testing evaluation...")
    y_pred = model.predict(X_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    accuracy = accuracy_score(y_test, y_pred_classes)
    
    print(f"   Test accuracy: {accuracy:.3f}")
    
    # Test 5: Inference speed
    print("5. Testing inference speed...")
    import time
    
    start_time = time.time()
    for _ in range(100):
        _ = model.predict(X_test[:1], verbose=0)
    end_time = time.time()
    
    avg_time = (end_time - start_time) / 100 * 1000  # Convert to ms
    print(f"   Average inference time: {avg_time:.2f} ms")
    
    print("\n✅ All tests passed!")
    return True

if __name__ == "__main__":
    test_basic_functionality()