#!/usr/bin/env python3
"""
Data loading utilities for spoken digit recognition.

This module handles loading and preprocessing of the Free Spoken Digit Dataset (FSDD)
with proper stratified splitting and data augmentation.
"""

import os
import warnings
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional, Dict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Audio processing
import librosa

# Dataset loading
from datasets import load_dataset

warnings.filterwarnings('ignore')

class FSDDDataLoader:
    """
    Data loader for the Free Spoken Digit Dataset with stratified splitting.
    """
    
    def __init__(self, target_sr: int = 8000, duration: float = 1.0):
        """
        Initialize the data loader.
        
        Args:
            target_sr: Target sampling rate (Hz)
            duration: Target audio duration (seconds)
        """
        self.target_sr = target_sr
        self.duration = duration
        self.label_encoder = LabelEncoder()
        
    def load_dataset(self) -> Tuple[np.ndarray, np.ndarray]:
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
            
            audio_data = np.array(audio_data)
            labels = np.array(labels)
            
            print(f"Successfully loaded {len(audio_data)} samples")
            print(f"Label distribution: {np.bincount(labels)}")
            
            return audio_data, labels
            
        except Exception as e:
            print(f"Error loading dataset: {e}")
            raise RuntimeError("Failed to load FSDD dataset. Please check internet connection.")
    
    def stratified_split(self, audio_data: np.ndarray, labels: np.ndarray, 
                        test_size: float = 0.2, val_size: float = 0.1, 
                        random_state: int = 42) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Create stratified train/validation/test splits to prevent data leakage.
        
        Args:
            audio_data: Array of raw audio waveforms
            labels: Array of corresponding labels
            test_size: Fraction of data to use for testing
            val_size: Fraction of remaining data to use for validation
            random_state: Random seed for reproducibility
            
        Returns:
            Dictionary containing train, val, and test splits
        """
        print("Creating stratified train/validation/test splits...")
        
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            audio_data, labels, 
            test_size=test_size, 
            random_state=random_state, 
            stratify=labels
        )
        
        # Second split: separate validation set from remaining data
        val_size_adjusted = val_size / (1 - test_size)  # Adjust for remaining data
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size_adjusted,
            random_state=random_state,
            stratify=y_temp
        )
        
        print(f"Training set: {len(X_train)} samples")
        print(f"Validation set: {len(X_val)} samples")
        print(f"Test set: {len(X_test)} samples")
        
        # Verify stratification
        print("\nLabel distribution in splits:")
        print(f"Train: {np.bincount(y_train)}")
        print(f"Val:   {np.bincount(y_val)}")
        print(f"Test:  {np.bincount(y_test)}")
        
        return {
            'train': (X_train, y_train),
            'val': (X_val, y_val),
            'test': (X_test, y_test)
        }
    
    def preprocess_audio_basic(self, audio: np.ndarray) -> np.ndarray:
        """
        Basic audio preprocessing: resample and normalize length.
        
        Args:
            audio: Raw audio waveform
            
        Returns:
            Preprocessed audio waveform
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
        
        return audio
    
    def augment_audio(self, audio: np.ndarray, augmentation_type: str = 'noise') -> np.ndarray:
        """
        Apply data augmentation to audio.
        
        Args:
            audio: Raw audio waveform
            augmentation_type: Type of augmentation to apply
            
        Returns:
            Augmented audio waveform
        """
        if augmentation_type == 'noise':
            # Add Gaussian noise
            noise = np.random.normal(0, 0.01, audio.shape)
            return audio + noise
            
        elif augmentation_type == 'time_shift':
            # Time shifting
            shift = np.random.randint(-int(0.1 * self.target_sr), int(0.1 * self.target_sr))
            return np.roll(audio, shift)
            
        elif augmentation_type == 'pitch_shift':
            # Pitch shifting using librosa
            pitch_shift = np.random.uniform(-2, 2)  # Semitones
            return librosa.effects.pitch_shift(audio, sr=self.target_sr, n_steps=pitch_shift)
            
        elif augmentation_type == 'time_stretch':
            # Time stretching
            stretch_factor = np.random.uniform(0.8, 1.2)
            return librosa.effects.time_stretch(audio, rate=stretch_factor)
            
        else:
            return audio
    
    def create_augmented_dataset(self, audio_data: np.ndarray, labels: np.ndarray,
                                augmentation_types: List[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create augmented dataset with multiple augmentation types.
        
        Args:
            audio_data: Array of raw audio waveforms
            labels: Array of corresponding labels
            augmentation_types: List of augmentation types to apply
            
        Returns:
            Tuple of (augmented_audio_data, augmented_labels)
        """
        if augmentation_types is None:
            augmentation_types = ['noise', 'time_shift', 'pitch_shift']
        
        print(f"Creating augmented dataset with {len(augmentation_types)} augmentation types...")
        
        augmented_audio = []
        augmented_labels = []
        
        for audio, label in zip(audio_data, labels):
            # Add original sample
            augmented_audio.append(audio)
            augmented_labels.append(label)
            
            # Add augmented samples
            for aug_type in augmentation_types:
                try:
                    aug_audio = self.augment_audio(audio, aug_type)
                    augmented_audio.append(aug_audio)
                    augmented_labels.append(label)
                except Exception as e:
                    print(f"Warning: Failed to apply {aug_type} augmentation: {e}")
                    # Fallback to original audio
                    augmented_audio.append(audio)
                    augmented_labels.append(label)
        
        print(f"Original samples: {len(audio_data)}")
        print(f"Augmented samples: {len(augmented_audio)}")
        print(f"Total samples: {len(augmented_audio)}")
        
        return np.array(augmented_audio), np.array(augmented_labels)