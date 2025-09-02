#!/usr/bin/env python3
"""
Inference utilities for the CNN-based spoken digit recognition system.

This module provides real-time inference capabilities including microphone
integration and voice activity detection.
"""

import os
import time
import warnings
import numpy as np
from typing import Tuple, Optional, Dict, Any
import sounddevice as sd
import librosa

# Suppress warnings
warnings.filterwarnings('ignore')

# Import our modules
from features import MFCCFeatureExtractor
from model import CompactCNN

class VoiceActivityDetector:
    """
    Simple voice activity detection for real-time audio processing.
    """
    
    def __init__(self, energy_threshold: float = 0.01, 
                 frame_length: int = 1024, hop_length: int = 512):
        """
        Initialize the voice activity detector.
        
        Args:
            energy_threshold: Energy threshold for voice detection
            frame_length: Frame length for energy calculation
            hop_length: Hop length for energy calculation
        """
        self.energy_threshold = energy_threshold
        self.frame_length = frame_length
        self.hop_length = hop_length
    
    def detect_voice_activity(self, audio: np.ndarray) -> Tuple[bool, float]:
        """
        Detect voice activity in audio.
        
        Args:
            audio: Audio waveform
            
        Returns:
            Tuple of (has_voice, energy_level)
        """
        # Calculate RMS energy
        energy = librosa.feature.rms(
            y=audio, 
            frame_length=self.frame_length, 
            hop_length=self.hop_length
        )
        
        # Get maximum energy
        max_energy = np.max(energy)
        
        # Check if energy exceeds threshold
        has_voice = max_energy > self.energy_threshold
        
        return has_voice, max_energy
    
    def find_most_energetic_window(self, audio: np.ndarray, window_duration: float = 1.0,
                                 sr: int = 8000) -> np.ndarray:
        """
        Find the most energetic 1-second window in the audio.
        
        Args:
            audio: Audio waveform
            window_duration: Duration of the window to extract (seconds)
            sr: Sampling rate
            
        Returns:
            Most energetic window of audio
        """
        window_samples = int(window_duration * sr)
        
        if len(audio) <= window_samples:
            return audio
        
        # Calculate energy for each window
        max_energy = 0
        best_start = 0
        
        for start in range(0, len(audio) - window_samples + 1, hop_length):
            window = audio[start:start + window_samples]
            has_voice, energy = self.detect_voice_activity(window)
            
            if energy > max_energy:
                max_energy = energy
                best_start = start
        
        return audio[best_start:best_start + window_samples]

class CNNInferenceEngine:
    """
    Inference engine for the CNN-based spoken digit recognition system.
    """
    
    def __init__(self, model_path: str, normalization_path: str,
                 target_sr: int = 8000, duration: float = 1.0, n_mfcc: int = 20):
        """
        Initialize the inference engine.
        
        Args:
            model_path: Path to the trained CNN model
            normalization_path: Path to normalization parameters
            target_sr: Target sampling rate
            duration: Target audio duration
            n_mfcc: Number of MFCC coefficients
        """
        self.target_sr = target_sr
        self.duration = duration
        self.n_mfcc = n_mfcc
        
        # Load model
        self.model = CompactCNN(
            input_shape=(n_mfcc * 3, int(target_sr * duration / 512) + 1),
            num_classes=10
        )
        self.model.load_model(model_path)
        
        # Load normalization parameters
        norm_params = np.load(normalization_path)
        self.norm_mean = norm_params['mean']
        self.norm_std = norm_params['std']
        
        # Initialize feature extractor
        self.feature_extractor = MFCCFeatureExtractor(
            target_sr=target_sr,
            duration=duration,
            n_mfcc=n_mfcc
        )
        
        # Initialize voice activity detector
        self.vad = VoiceActivityDetector()
        
        print("CNN Inference Engine initialized successfully")
    
    def preprocess_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Preprocess audio for inference.
        
        Args:
            audio: Raw audio waveform
            
        Returns:
            Preprocessed audio
        """
        # Resample if needed
        if len(audio) != self.target_sr * self.duration:
            target_length = int(self.target_sr * self.duration)
            if len(audio) < target_length:
                audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')
            else:
                audio = audio[:target_length]
        
        return audio
    
    def predict_from_audio(self, audio: np.ndarray, 
                          use_vad: bool = True) -> Tuple[int, float, float, Dict[str, Any]]:
        """
        Predict digit from audio waveform.
        
        Args:
            audio: Raw audio waveform
            use_vad: Whether to use voice activity detection
            
        Returns:
            Tuple of (predicted_digit, confidence, inference_time_ms, metadata)
        """
        start_time = time.time()
        
        # Preprocess audio
        audio = self.preprocess_audio(audio)
        
        # Voice activity detection
        has_voice, energy = self.vad.detect_voice_activity(audio)
        
        if use_vad and not has_voice:
            return -1, 0.0, 0.0, {
                'has_voice': False,
                'energy': energy,
                'message': 'No voice activity detected'
            }
        
        # Find most energetic window if audio is longer than target duration
        if len(audio) > self.target_sr * self.duration:
            audio = self.vad.find_most_energetic_window(audio, self.duration, self.target_sr)
        
        # Extract features
        features = self.feature_extractor.extract_spectro_temporal_features(audio)
        
        # Normalize features
        features_norm = (features - self.norm_mean) / self.norm_std
        
        # Make prediction
        predicted_digit, confidence, inference_time = self.model.predict_single(features_norm)
        
        total_time = (time.time() - start_time) * 1000
        
        metadata = {
            'has_voice': has_voice,
            'energy': energy,
            'feature_extraction_time': total_time - inference_time,
            'total_time': total_time
        }
        
        return predicted_digit, confidence, total_time, metadata
    
    def predict_from_microphone(self, duration: float = 2.0, 
                               use_vad: bool = True) -> Optional[Tuple[int, float, float, Dict[str, Any]]]:
        """
        Record audio from microphone and predict the spoken digit.
        
        Args:
            duration: Recording duration in seconds
            use_vad: Whether to use voice activity detection
            
        Returns:
            Tuple of (predicted_digit, confidence, inference_time_ms, metadata) or None if failed
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
            result = self.predict_from_audio(audio, use_vad=use_vad)
            
            predicted_digit, confidence, inference_time, metadata = result
            
            if predicted_digit == -1:
                print(f"No voice activity detected (energy: {metadata['energy']:.4f})")
                return result
            
            print(f"Predicted digit: {predicted_digit}")
            print(f"Confidence: {confidence:.3f}")
            print(f"Total time: {inference_time:.2f} ms")
            print(f"Voice activity: {metadata['has_voice']}")
            print(f"Energy level: {metadata['energy']:.4f}")
            
            return result
            
        except Exception as e:
            print(f"Microphone recording failed: {e}")
            print("Make sure you have a microphone connected and sounddevice is properly installed.")
            return None
    
    def benchmark_inference(self, num_samples: int = 100) -> Dict[str, float]:
        """
        Benchmark inference speed with random audio samples.
        
        Args:
            num_samples: Number of samples to test
            
        Returns:
            Dictionary containing timing statistics
        """
        print(f"Benchmarking inference speed on {num_samples} samples...")
        
        inference_times = []
        feature_times = []
        total_times = []
        
        for i in range(num_samples):
            if i % 20 == 0:
                print(f"Benchmarking sample {i}/{num_samples}")
            
            # Generate random audio for testing
            audio = np.random.randn(int(self.target_sr * self.duration))
            
            # Make prediction
            _, _, total_time, metadata = self.predict_from_audio(audio, use_vad=False)
            
            inference_times.append(metadata['total_time'] - metadata['feature_extraction_time'])
            feature_times.append(metadata['feature_extraction_time'])
            total_times.append(total_time)
        
        # Calculate statistics
        stats = {
            'mean_inference_time': np.mean(inference_times),
            'std_inference_time': np.std(inference_times),
            'mean_feature_time': np.mean(feature_times),
            'std_feature_time': np.std(feature_times),
            'mean_total_time': np.mean(total_times),
            'std_total_time': np.std(total_times),
            'p95_total_time': np.percentile(total_times, 95)
        }
        
        print(f"\nInference Benchmark Results:")
        print(f"Feature extraction: {stats['mean_feature_time']:.2f} ± {stats['std_feature_time']:.2f} ms")
        print(f"Model inference:    {stats['mean_inference_time']:.2f} ± {stats['std_inference_time']:.2f} ms")
        print(f"Total time:         {stats['mean_total_time']:.2f} ± {stats['std_total_time']:.2f} ms")
        print(f"95th percentile:    {stats['p95_total_time']:.2f} ms")
        
        # Check if meets requirement
        if stats['mean_total_time'] < 20:
            print("✓ Total inference time meets <20ms requirement")
        else:
            print("⚠ Total inference time exceeds 20ms requirement")
        
        return stats

def main():
    """
    Main function for testing the inference engine.
    """
    print("=" * 60)
    print("CNN INFERENCE ENGINE TEST")
    print("=" * 60)
    
    # Check if model files exist
    model_path = "spoken_digit_cnn_model.h5"
    norm_path = "normalization_params.npz"
    
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found!")
        print("Please run train.py first to train the model.")
        return
    
    if not os.path.exists(norm_path):
        print(f"Error: Normalization file {norm_path} not found!")
        print("Please run train.py first to train the model.")
        return
    
    # Initialize inference engine
    inference_engine = CNNInferenceEngine(model_path, norm_path)
    
    # Benchmark inference speed
    print("\n1. Benchmarking inference speed...")
    speed_results = inference_engine.benchmark_inference(num_samples=50)
    
    # Test with microphone
    print("\n2. Testing with microphone...")
    try:
        response = input("Would you like to test with microphone? (y/n): ").lower()
        if response == 'y':
            result = inference_engine.predict_from_microphone(duration=2.0, use_vad=True)
            if result:
                predicted_digit, confidence, inference_time, metadata = result
                if predicted_digit != -1:
                    print(f"\nFinal Result:")
                    print(f"Predicted digit: {predicted_digit}")
                    print(f"Confidence: {confidence:.3f}")
                    print(f"Inference time: {inference_time:.2f} ms")
    except KeyboardInterrupt:
        print("\nTest interrupted by user.")
    except Exception as e:
        print(f"Microphone test failed: {e}")
    
    print("\n" + "=" * 60)
    print("INFERENCE TEST COMPLETED")
    print("=" * 60)

if __name__ == "__main__":
    main()