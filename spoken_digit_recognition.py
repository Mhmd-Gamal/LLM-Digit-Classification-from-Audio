#!/usr/bin/env python3
"""
Spoken Digit Recognition System using FSDD Dataset

This script implements a lightweight neural network for recognizing spoken digits (0-9)
using Mel-Frequency Cepstral Coefficients (MFCCs) as features.

Author: AI Assistant
Date: 2024
"""

import os
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Tuple, List, Optional

# Audio processing
import librosa
import sounddevice as sd

# Machine learning
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder

# Deep learning
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

# Dataset loading
from datasets import load_dataset

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

class SpokenDigitRecognizer:
    """
    A lightweight spoken digit recognition system using MFCC features and MLP.
    
    This class handles the complete pipeline from data loading to real-time prediction,
    including audio preprocessing, model training, and evaluation.
    """
    
    def __init__(self, target_sr: int = 8000, duration: float = 1.0, 
                 n_mfcc: int = 20, feature_dim: int = 40):
        """
        Initialize the spoken digit recognizer.
        
        Args:
            target_sr: Target sampling rate (Hz)
            duration: Target audio duration (seconds)
            n_mfcc: Number of MFCC coefficients to extract
            feature_dim: Final feature dimension (n_mfcc * 2 for mean + std)
        """
        self.target_sr = target_sr
        self.duration = duration
        self.n_mfcc = n_mfcc
        self.feature_dim = feature_dim
        self.model = None
        self.label_encoder = LabelEncoder()
        self.is_trained = False
        
        print(f"Initialized SpokenDigitRecognizer with:")
        print(f"  - Target sampling rate: {target_sr} Hz")
        print(f"  - Target duration: {duration} seconds")
        print(f"  - MFCC coefficients: {n_mfcc}")
        print(f"  - Feature dimension: {feature_dim}")
    
    def load_fsdd_dataset(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load the Free Spoken Digit Dataset from Hugging Face.
        
        Returns:
            Tuple of (audio_data, labels) where audio_data contains raw waveforms
            and labels contains corresponding digit labels (0-9)
        """
        print("Loading FSDD dataset from Hugging Face...")
        
        try:
            # Load the dataset from Hugging Face
            dataset = load_dataset("freespoken-digit", split="train")
            print(f"Loaded {len(dataset)} audio samples")
            
            audio_data = []
            labels = []
            
            for item in dataset:
                # Extract audio array and label
                audio = np.array(item['audio']['array'])
                label = int(item['label'])
                
                audio_data.append(audio)
                labels.append(label)
            
            print(f"Successfully loaded {len(audio_data)} samples")
            print(f"Label distribution: {np.bincount(labels)}")
            
            return np.array(audio_data), np.array(labels)
            
        except Exception as e:
            print(f"Error loading dataset: {e}")
            print("Falling back to manual dataset loading...")
            return self._load_fsdd_manual()
    
    def _load_fsdd_manual(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fallback method to load FSDD dataset manually.
        This would require the dataset to be downloaded locally.
        """
        # This is a placeholder for manual loading
        # In practice, you would download the dataset and load it here
        raise NotImplementedError("Manual dataset loading not implemented. Please ensure internet connection for Hugging Face dataset loading.")
    
    def preprocess_audio(self, audio: np.ndarray, add_noise: bool = False, 
                        noise_factor: float = 0.01) -> np.ndarray:
        """
        Preprocess a single audio sample into MFCC features.
        
        Args:
            audio: Raw audio waveform
            add_noise: Whether to add Gaussian noise for augmentation
            noise_factor: Standard deviation of noise to add
            
        Returns:
            MFCC feature vector of shape (feature_dim,)
        """
        # Resample to target sampling rate if needed
        if len(audio) != self.target_sr * self.duration:
            # Pad or truncate to target length
            target_length = int(self.target_sr * self.duration)
            if len(audio) < target_length:
                # Pad with zeros
                audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')
            else:
                # Truncate
                audio = audio[:target_length]
        
        # Add noise for data augmentation if requested
        if add_noise:
            noise = np.random.normal(0, noise_factor, audio.shape)
            audio = audio + noise
        
        # Compute MFCCs
        mfccs = librosa.feature.mfcc(
            y=audio, 
            sr=self.target_sr, 
            n_mfcc=self.n_mfcc,
            n_fft=2048,
            hop_length=512
        )
        
        # Extract mean and standard deviation for each MFCC coefficient
        # This creates a fixed-size feature vector regardless of audio length
        mfcc_mean = np.mean(mfccs, axis=1)
        mfcc_std = np.std(mfccs, axis=1)
        
        # Combine mean and std to create final feature vector
        features = np.concatenate([mfcc_mean, mfcc_std])
        
        return features
    
    def extract_features(self, audio_data: np.ndarray, labels: np.ndarray, 
                        augment: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract MFCC features from all audio samples.
        
        Args:
            audio_data: Array of raw audio waveforms
            labels: Array of corresponding labels
            augment: Whether to include augmented samples
            
        Returns:
            Tuple of (features, labels) where features is (n_samples, feature_dim)
        """
        print("Extracting MFCC features...")
        
        features_list = []
        labels_list = []
        
        for i, (audio, label) in enumerate(zip(audio_data, labels)):
            if i % 100 == 0:
                print(f"Processing sample {i}/{len(audio_data)}")
            
            # Extract features from original audio
            features = self.preprocess_audio(audio, add_noise=False)
            features_list.append(features)
            labels_list.append(label)
            
            # Add augmented version if requested
            if augment:
                aug_features = self.preprocess_audio(audio, add_noise=True)
                features_list.append(aug_features)
                labels_list.append(label)
        
        features_array = np.array(features_list)
        labels_array = np.array(labels_list)
        
        print(f"Extracted features shape: {features_array.shape}")
        print(f"Labels shape: {labels_array.shape}")
        
        return features_array, labels_array
    
    def build_model(self) -> tf.keras.Model:
        """
        Build a lightweight MLP model for digit classification.
        
        Returns:
            Compiled Keras model
        """
        print("Building neural network model...")
        
        model = Sequential([
            # Input layer
            Dense(128, activation='relu', input_shape=(self.feature_dim,)),
            Dropout(0.25),
            
            # Hidden layers
            Dense(256, activation='relu'),
            Dropout(0.25),
            
            Dense(128, activation='relu'),
            Dropout(0.1),
            
            # Output layer for 10 digits (0-9)
            Dense(10, activation='softmax')
        ])
        
        # Compile the model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("Model architecture:")
        model.summary()
        
        return model
    
    def train(self, features: np.ndarray, labels: np.ndarray, 
              test_size: float = 0.2, epochs: int = 50, batch_size: int = 32):
        """
        Train the model on the provided features and labels.
        
        Args:
            features: MFCC feature matrix
            labels: Corresponding digit labels
            test_size: Fraction of data to use for testing
            epochs: Number of training epochs
            batch_size: Training batch size
        """
        print("Splitting data into train/test sets...")
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=test_size, random_state=42, stratify=labels
        )
        
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        
        # Build the model
        self.model = self.build_model()
        
        # Set up early stopping
        early_stopping = EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
        
        print("Starting training...")
        
        # Train the model
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            callbacks=[early_stopping],
            verbose=1
        )
        
        self.is_trained = True
        
        # Store test data for evaluation
        self.X_test = X_test
        self.y_test = y_test
        
        return history
    
    def evaluate(self) -> dict:
        """
        Evaluate the trained model on the test set.
        
        Returns:
            Dictionary containing evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        print("Evaluating model on test set...")
        
        # Make predictions
        y_pred_proba = self.model.predict(self.X_test)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Calculate metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        
        print(f"\nTest Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(self.y_test, y_pred, 
                                  target_names=[str(i) for i in range(10)]))
        
        # Confusion matrix
        cm = confusion_matrix(self.y_test, y_pred)
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=range(10), yticklabels=range(10))
        plt.title('Confusion Matrix - Spoken Digit Recognition')
        plt.xlabel('Predicted Digit')
        plt.ylabel('True Digit')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return {
            'accuracy': accuracy,
            'confusion_matrix': cm,
            'predictions': y_pred,
            'true_labels': self.y_test
        }
    
    def predict_single(self, audio: np.ndarray) -> Tuple[int, float, float]:
        """
        Predict digit from a single audio sample.
        
        Args:
            audio: Raw audio waveform
            
        Returns:
            Tuple of (predicted_digit, confidence, inference_time_ms)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        # Measure inference time
        start_time = time.time()
        
        # Extract features
        features = self.preprocess_audio(audio, add_noise=False)
        features = features.reshape(1, -1)  # Add batch dimension
        
        # Make prediction
        prediction_proba = self.model.predict(features, verbose=0)
        predicted_digit = np.argmax(prediction_proba)
        confidence = np.max(prediction_proba)
        
        # Calculate inference time
        inference_time = (time.time() - start_time) * 1000  # Convert to ms
        
        return predicted_digit, confidence, inference_time
    
    def predict_from_microphone(self, duration: float = 1.0) -> Optional[Tuple[int, float, float]]:
        """
        Record audio from microphone and predict the spoken digit.
        
        Args:
            duration: Recording duration in seconds
            
        Returns:
            Tuple of (predicted_digit, confidence, inference_time_ms) or None if failed
        """
        try:
            print(f"Recording {duration} seconds of audio...")
            print("Speak a digit (0-9) now!")
            
            # Record audio
            audio = sd.rec(
                int(duration * self.target_sr),
                samplerate=self.target_sr,
                channels=1,
                dtype='float64'
            )
            sd.wait()  # Wait until recording is finished
            
            print("Processing audio...")
            
            # Flatten the audio array
            audio = audio.flatten()
            
            # Make prediction
            prediction, confidence, inference_time = self.predict_single(audio)
            
            print(f"Predicted digit: {prediction}")
            print(f"Confidence: {confidence:.3f}")
            print(f"Inference time: {inference_time:.2f} ms")
            
            return prediction, confidence, inference_time
            
        except Exception as e:
            print(f"Microphone recording failed: {e}")
            print("Make sure you have a microphone connected and sounddevice is properly installed.")
            return None
    
    def save_model(self, filepath: str):
        """Save the trained model to disk."""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model from disk."""
        self.model = tf.keras.models.load_model(filepath)
        self.is_trained = True
        print(f"Model loaded from {filepath}")


def main():
    """
    Main function to demonstrate the spoken digit recognition system.
    """
    print("=" * 60)
    print("SPOKEN DIGIT RECOGNITION SYSTEM")
    print("=" * 60)
    
    # Initialize the recognizer
    recognizer = SpokenDigitRecognizer()
    
    # Load and preprocess data
    print("\n1. Loading and preprocessing data...")
    audio_data, labels = recognizer.load_fsdd_dataset()
    features, labels = recognizer.extract_features(audio_data, labels, augment=True)
    
    # Train the model
    print("\n2. Training the model...")
    history = recognizer.train(features, labels, epochs=30)
    
    # Evaluate the model
    print("\n3. Evaluating the model...")
    results = recognizer.evaluate()
    
    # Test responsiveness
    print("\n4. Testing responsiveness...")
    test_audio = audio_data[0]  # Use first sample for testing
    pred, conf, time_ms = recognizer.predict_single(test_audio)
    print(f"Single prediction time: {time_ms:.2f} ms")
    
    # Save the model
    print("\n5. Saving the model...")
    recognizer.save_model("spoken_digit_model.h5")
    
    # Optional: Real-time microphone demo
    print("\n6. Real-time microphone demo (optional)...")
    try:
        response = input("Would you like to test with microphone? (y/n): ").lower()
        if response == 'y':
            recognizer.predict_from_microphone()
    except KeyboardInterrupt:
        print("\nDemo interrupted by user.")
    except Exception as e:
        print(f"Microphone demo not available: {e}")
    
    print("\n" + "=" * 60)
    print("DEMO COMPLETED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    main()