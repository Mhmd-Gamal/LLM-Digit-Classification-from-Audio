#!/usr/bin/env python3
"""
Feature extraction utilities for spoken digit recognition.

This module implements enhanced MFCC feature extraction with delta and delta-delta features
for CNN-based models that treat MFCCs as spectro-temporal images.
"""

import warnings
import numpy as np
from typing import Tuple, Optional
import librosa

warnings.filterwarnings('ignore')

class MFCCFeatureExtractor:
    """
    Enhanced MFCC feature extractor for CNN-based models.
    """
    
    def __init__(self, target_sr: int = 8000, duration: float = 1.0, 
                 n_mfcc: int = 20, n_fft: int = 2048, hop_length: int = 512):
        """
        Initialize the MFCC feature extractor.
        
        Args:
            target_sr: Target sampling rate (Hz)
            duration: Target audio duration (seconds)
            n_mfcc: Number of MFCC coefficients to extract
            n_fft: FFT window size
            hop_length: Number of samples between successive frames
        """
        self.target_sr = target_sr
        self.duration = duration
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        
        # Calculate expected time frames
        self.expected_frames = int((self.target_sr * self.duration) / self.hop_length) + 1
        
        print(f"Initialized MFCCFeatureExtractor:")
        print(f"  - Target sampling rate: {target_sr} Hz")
        print(f"  - Target duration: {duration} seconds")
        print(f"  - MFCC coefficients: {n_mfcc}")
        print(f"  - Expected time frames: {self.expected_frames}")
        print(f"  - Output shape: ({n_mfcc * 3}, {self.expected_frames})")
    
    def extract_mfcc_features(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract MFCC features from audio.
        
        Args:
            audio: Raw audio waveform
            
        Returns:
            MFCC features of shape (n_mfcc, time_frames)
        """
        # Compute MFCCs
        mfccs = librosa.feature.mfcc(
            y=audio,
            sr=self.target_sr,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        
        return mfccs
    
    def extract_delta_features(self, mfccs: np.ndarray) -> np.ndarray:
        """
        Extract delta (first derivative) features.
        
        Args:
            mfccs: MFCC features of shape (n_mfcc, time_frames)
            
        Returns:
            Delta features of shape (n_mfcc, time_frames)
        """
        return librosa.feature.delta(mfccs)
    
    def extract_delta_delta_features(self, mfccs: np.ndarray) -> np.ndarray:
        """
        Extract delta-delta (second derivative) features.
        
        Args:
            mfccs: MFCC features of shape (n_mfcc, time_frames)
            
        Returns:
            Delta-delta features of shape (n_mfcc, time_frames)
        """
        return librosa.feature.delta(mfccs, order=2)
    
    def extract_spectro_temporal_features(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract complete spectro-temporal features (MFCC + delta + delta-delta).
        
        Args:
            audio: Raw audio waveform
            
        Returns:
            Spectro-temporal features of shape (n_mfcc * 3, time_frames)
        """
        # Extract base MFCC features
        mfccs = self.extract_mfcc_features(audio)
        
        # Extract delta features
        delta_mfccs = self.extract_delta_features(mfccs)
        
        # Extract delta-delta features
        delta_delta_mfccs = self.extract_delta_delta_features(mfccs)
        
        # Stack all features vertically
        spectro_temporal = np.vstack([mfccs, delta_mfccs, delta_delta_mfccs])
        
        # Ensure consistent time dimension
        if spectro_temporal.shape[1] != self.expected_frames:
            if spectro_temporal.shape[1] < self.expected_frames:
                # Pad with zeros
                pad_width = self.expected_frames - spectro_temporal.shape[1]
                spectro_temporal = np.pad(spectro_temporal, ((0, 0), (0, pad_width)), mode='constant')
            else:
                # Truncate
                spectro_temporal = spectro_temporal[:, :self.expected_frames]
        
        return spectro_temporal
    
    def extract_features_batch(self, audio_data: np.ndarray, 
                              progress_callback: Optional[callable] = None) -> np.ndarray:
        """
        Extract features from a batch of audio samples.
        
        Args:
            audio_data: Array of raw audio waveforms
            progress_callback: Optional callback function for progress updates
            
        Returns:
            Feature array of shape (n_samples, n_mfcc * 3, time_frames)
        """
        print("Extracting spectro-temporal features...")
        
        features_list = []
        
        for i, audio in enumerate(audio_data):
            if progress_callback and i % 100 == 0:
                progress_callback(i, len(audio_data))
            
            # Extract spectro-temporal features
            features = self.extract_spectro_temporal_features(audio)
            features_list.append(features)
        
        features_array = np.array(features_list)
        
        print(f"Extracted features shape: {features_array.shape}")
        return features_array
    
    def normalize_features(self, features: np.ndarray, 
                          mean: Optional[np.ndarray] = None, 
                          std: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Normalize features using z-score normalization.
        
        Args:
            features: Feature array of shape (n_samples, n_mfcc * 3, time_frames)
            mean: Pre-computed mean for normalization (if None, computed from features)
            std: Pre-computed std for normalization (if None, computed from features)
            
        Returns:
            Tuple of (normalized_features, mean, std)
        """
        if mean is None or std is None:
            # Compute mean and std across all samples and time frames
            mean = np.mean(features, axis=(0, 2), keepdims=True)
            std = np.std(features, axis=(0, 2), keepdims=True)
            
            # Avoid division by zero
            std = np.where(std == 0, 1.0, std)
        
        # Normalize features
        normalized_features = (features - mean) / std
        
        return normalized_features, mean, std
    
    def apply_spec_augment(self, features: np.ndarray, 
                          freq_mask_param: int = 8, 
                          time_mask_param: int = 10,
                          num_freq_masks: int = 2,
                          num_time_masks: int = 2) -> np.ndarray:
        """
        Apply SpecAugment masking to features.
        
        Args:
            features: Feature array of shape (n_mfcc * 3, time_frames)
            freq_mask_param: Maximum frequency mask width
            time_mask_param: Maximum time mask width
            num_freq_masks: Number of frequency masks to apply
            num_time_masks: Number of time masks to apply
            
        Returns:
            Augmented features
        """
        augmented = features.copy()
        
        # Apply frequency masking
        for _ in range(num_freq_masks):
            mask_width = np.random.randint(0, freq_mask_param + 1)
            if mask_width > 0:
                mask_start = np.random.randint(0, features.shape[0] - mask_width)
                augmented[mask_start:mask_start + mask_width, :] = 0
        
        # Apply time masking
        for _ in range(num_time_masks):
            mask_width = np.random.randint(0, time_mask_param + 1)
            if mask_width > 0:
                mask_start = np.random.randint(0, features.shape[1] - mask_width)
                augmented[:, mask_start:mask_start + mask_width] = 0
        
        return augmented
    
    def extract_features_with_augmentation(self, audio_data: np.ndarray, 
                                         labels: np.ndarray,
                                         use_spec_augment: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract features with optional SpecAugment augmentation.
        
        Args:
            audio_data: Array of raw audio waveforms
            labels: Array of corresponding labels
            use_spec_augment: Whether to apply SpecAugment
            
        Returns:
            Tuple of (features, labels) with augmented data
        """
        print("Extracting features with augmentation...")
        
        features_list = []
        labels_list = []
        
        for i, (audio, label) in enumerate(zip(audio_data, labels)):
            if i % 100 == 0:
                print(f"Processing sample {i}/{len(audio_data)}")
            
            # Extract base features
            features = self.extract_spectro_temporal_features(audio)
            features_list.append(features)
            labels_list.append(label)
            
            # Add SpecAugment version if requested
            if use_spec_augment:
                aug_features = self.apply_spec_augment(features)
                features_list.append(aug_features)
                labels_list.append(label)
        
        features_array = np.array(features_list)
        labels_array = np.array(labels_list)
        
        print(f"Final features shape: {features_array.shape}")
        print(f"Final labels shape: {labels_array.shape}")
        
        return features_array, labels_array