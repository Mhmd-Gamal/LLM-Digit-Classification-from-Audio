#!/usr/bin/env python3
"""
Real-time Inference Script for Spoken Digit Recognition.

This script provides a CLI interface for real-time microphone input,
voice activity detection, and inference with ≤150ms end-to-end latency.
"""

import os
import sys
import argparse
import time
import warnings
import numpy as np
import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
import sounddevice as sd
import webrtcvad
from collections import deque
from typing import Optional, Tuple, List
import threading
import queue

# Suppress warnings
warnings.filterwarnings('ignore')

# Import our modules
from model_student import create_student_model

class VoiceActivityDetector:
    """
    Voice Activity Detection using WebRTC VAD.
    """
    
    def __init__(self, sample_rate: int = 8000, frame_duration_ms: int = 30):
        """
        Initialize VAD.
        
        Args:
            sample_rate: Audio sample rate
            frame_duration_ms: Frame duration in milliseconds
        """
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self.frame_size = int(sample_rate * frame_duration_ms / 1000)
        
        # Initialize WebRTC VAD
        self.vad = webrtcvad.Vad(2)  # Aggressiveness level 2
        
        # Audio buffer
        self.audio_buffer = deque(maxlen=int(sample_rate * 2))  # 2 seconds buffer
        self.is_speaking = False
        self.speech_start = None
        self.speech_end = None
        
        # VAD parameters
        self.min_speech_duration = 0.5  # Minimum speech duration in seconds
        self.silence_threshold = 1.0   # Silence threshold in seconds
    
    def add_audio(self, audio_chunk: np.ndarray) -> bool:
        """
        Add audio chunk and detect voice activity.
        
        Args:
            audio_chunk: Audio chunk as numpy array
            
        Returns:
            True if speech is detected
        """
        # Add to buffer
        self.audio_buffer.extend(audio_chunk)
        
        # Process frames for VAD
        if len(self.audio_buffer) >= self.frame_size:
            frame = np.array(list(self.audio_buffer)[:self.frame_size])
            
            # Convert to 16-bit PCM
            frame_16bit = (frame * 32767).astype(np.int16)
            
            # Check VAD
            is_speech = self.vad.is_speech(frame_16bit.tobytes(), self.sample_rate)
            
            current_time = time.time()
            
            if is_speech and not self.is_speaking:
                # Speech started
                self.is_speaking = True
                self.speech_start = current_time
                print("🎤 Speech detected!")
                
            elif not is_speech and self.is_speaking:
                # Speech ended
                speech_duration = current_time - self.speech_start
                
                if speech_duration >= self.min_speech_duration:
                    self.speech_end = current_time
                    self.is_speaking = False
                    return True
                else:
                    # Too short, ignore
                    self.is_speaking = False
                    self.speech_start = None
            
            # Clear buffer if speech ended
            if self.speech_end and (current_time - self.speech_end) > self.silence_threshold:
                self.audio_buffer.clear()
                self.speech_end = None
        
        return False
    
    def get_speech_segment(self) -> Optional[np.ndarray]:
        """
        Get the detected speech segment.
        
        Returns:
            Speech audio segment or None
        """
        if self.speech_end is None:
            return None
        
        # Extract speech segment from buffer
        speech_samples = int((self.speech_end - self.speech_start) * self.sample_rate)
        speech_start_idx = len(self.audio_buffer) - speech_samples
        
        if speech_start_idx >= 0:
            speech_segment = np.array(list(self.audio_buffer)[speech_start_idx:])
            return speech_segment
        
        return None

class AudioProcessor:
    """
    Audio processing for real-time inference.
    """
    
    def __init__(self, sample_rate: int = 8000, target_length: int = 8000):
        """
        Initialize audio processor.
        
        Args:
            sample_rate: Audio sample rate
            target_length: Target audio length in samples
        """
        self.sample_rate = sample_rate
        self.target_length = target_length
        
        # MFCC feature extractor
        self.mfcc_transform = T.MFCC(
            sample_rate=sample_rate,
            n_mfcc=20,
            melkwargs={
                'n_fft': 512,
                'n_mels': 40,
                'hop_length': 160,
                'mel_scale': 'htk',
            }
        )
        
        # Delta and delta-delta features
        self.delta_transform = T.ComputeDeltas(win_length=5)
        self.delta2_transform = T.ComputeDeltas(win_length=5)
    
    def process_audio(self, audio: np.ndarray) -> torch.Tensor:
        """
        Process audio to MFCC features.
        
        Args:
            audio: Input audio as numpy array
            
        Returns:
            MFCC features tensor
        """
        # Convert to torch tensor
        audio_tensor = torch.from_numpy(audio).float()
        
        # Ensure mono
        if audio_tensor.dim() > 1:
            audio_tensor = audio_tensor.mean(dim=-1)
        
        # Resample if needed
        if len(audio_tensor) != self.target_length:
            # Pad or truncate
            if len(audio_tensor) < self.target_length:
                # Pad with zeros
                padding = self.target_length - len(audio_tensor)
                audio_tensor = torch.cat([audio_tensor, torch.zeros(padding)])
            else:
                # Truncate
                audio_tensor = audio_tensor[:self.target_length]
        
        # Extract MFCC features
        mfcc = self.mfcc_transform(audio_tensor.unsqueeze(0))
        
        # Add delta and delta-delta features
        delta = self.delta_transform(mfcc)
        delta2 = self.delta2_transform(delta)
        
        # Concatenate features
        features = torch.cat([mfcc, delta, delta2], dim=1)
        
        # Normalize
        features = (features - features.mean()) / (features.std() + 1e-8)
        
        # Reshape to (1, 60, time_frames)
        features = features.squeeze(0)  # Remove batch dimension
        
        # Pad or truncate time dimension to 80 frames
        if features.shape[1] < 80:
            padding = 80 - features.shape[1]
            features = torch.cat([features, torch.zeros(60, padding)], dim=1)
        else:
            features = features[:, :80]
        
        # Add channel dimension
        features = features.unsqueeze(0)  # (1, 60, 80)
        
        return features

class RealTimeInference:
    """
    Real-time inference system with microphone input.
    """
    
    def __init__(self, model_path: str, sample_rate: int = 8000, 
                 chunk_duration_ms: int = 30):
        """
        Initialize real-time inference.
        
        Args:
            model_path: Path to trained model
            sample_rate: Audio sample rate
            chunk_duration_ms: Audio chunk duration in milliseconds
        """
        self.sample_rate = sample_rate
        self.chunk_size = int(sample_rate * chunk_duration_ms / 1000)
        
        # Load model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model(model_path)
        self.model.eval()
        
        # Initialize components
        self.vad = VoiceActivityDetector(sample_rate=sample_rate)
        self.audio_processor = AudioProcessor(sample_rate=sample_rate)
        
        # Audio queue for processing
        self.audio_queue = queue.Queue()
        self.processing_thread = None
        self.is_running = False
        
        print(f"Real-time inference initialized:")
        print(f"  - Sample rate: {sample_rate} Hz")
        print(f"  - Chunk duration: {chunk_duration_ms} ms")
        print(f"  - Device: {self.device}")
    
    def _load_model(self, model_path: str) -> nn.Module:
        """Load the trained model."""
        print(f"Loading model from {model_path}...")
        
        # Create model architecture
        model = create_student_model(input_shape=(1, 60, 80), num_classes=10)
        
        # Load weights
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            print("Model loaded successfully")
        else:
            print(f"Warning: Model file {model_path} not found. Using random weights.")
        
        model = model.to(self.device)
        return model
    
    def _audio_callback(self, indata: np.ndarray, frames: int, 
                       time_info, status):
        """Audio callback for microphone input."""
        if status:
            print(f"Audio callback status: {status}")
        
        # Add audio to queue
        self.audio_queue.put(indata.copy())
    
    def _process_audio_loop(self):
        """Process audio in background thread."""
        while self.is_running:
            try:
                # Get audio chunk
                audio_chunk = self.audio_queue.get(timeout=0.1)
                
                # Add to VAD
                speech_detected = self.vad.add_audio(audio_chunk.flatten())
                
                if speech_detected:
                    # Get speech segment
                    speech_segment = self.vad.get_speech_segment()
                    
                    if speech_segment is not None:
                        # Process and predict
                        self._predict_digit(speech_segment)
                        
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in audio processing: {e}")
    
    def _predict_digit(self, audio: np.ndarray):
        """Predict digit from audio segment."""
        start_time = time.time()
        
        try:
            # Process audio
            features = self.audio_processor.process_audio(audio)
            features = features.to(self.device)
            
            # Inference
            with torch.no_grad():
                outputs = self.model(features)
                probabilities = torch.softmax(outputs, dim=1)
                predicted_digit = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0, predicted_digit].item()
            
            # Calculate total time
            total_time = (time.time() - start_time) * 1000
            
            # Display result
            print(f"🎯 Predicted: {predicted_digit} (confidence: {confidence:.2f})")
            print(f"⏱️  Total time: {total_time:.1f}ms")
            
            if total_time <= 150:
                print("✅ Target latency achieved (≤150ms)")
            else:
                print("⚠️  Target latency exceeded")
            
        except Exception as e:
            print(f"Error in prediction: {e}")
    
    def start(self):
        """Start real-time inference."""
        print("Starting real-time inference...")
        print("Speak a digit (0-9) into the microphone")
        print("Press Ctrl+C to stop")
        
        self.is_running = True
        
        # Start processing thread
        self.processing_thread = threading.Thread(target=self._process_audio_loop)
        self.processing_thread.start()
        
        try:
            # Start audio stream
            with sd.InputStream(
                callback=self._audio_callback,
                channels=1,
                samplerate=self.sample_rate,
                blocksize=self.chunk_size,
                dtype=np.float32
            ):
                print("🎤 Microphone active. Speak now!")
                
                # Keep running until interrupted
                while True:
                    time.sleep(0.1)
                    
        except KeyboardInterrupt:
            print("\nStopping real-time inference...")
        finally:
            self.stop()
    
    def stop(self):
        """Stop real-time inference."""
        self.is_running = False
        
        if self.processing_thread:
            self.processing_thread.join()
        
        print("Real-time inference stopped")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Real-time Spoken Digit Recognition')
    
    parser.add_argument('--model', type=str, default='student_model_distilled.pth',
                       help='Path to trained model')
    parser.add_argument('--sample_rate', type=int, default=8000,
                       help='Audio sample rate')
    parser.add_argument('--chunk_duration', type=int, default=30,
                       help='Audio chunk duration in milliseconds')
    parser.add_argument('--mic', action='store_true',
                       help='Enable microphone input (default)')
    parser.add_argument('--file', type=str,
                       help='Process audio file instead of microphone')
    
    args = parser.parse_args()
    
    # Check if microphone is requested
    if not args.mic and not args.file:
        args.mic = True  # Default to microphone
    
    # Initialize inference system
    inference = RealTimeInference(
        model_path=args.model,
        sample_rate=args.sample_rate,
        chunk_duration_ms=args.chunk_duration
    )
    
    if args.file:
        # Process audio file
        print(f"Processing audio file: {args.file}")
        
        if not os.path.exists(args.file):
            print(f"Error: File {args.file} not found")
            return
        
        # Load audio file
        audio, sr = torchaudio.load(args.file)
        audio = audio.numpy().flatten()
        
        # Resample if needed
        if sr != args.sample_rate:
            print(f"Resampling from {sr}Hz to {args.sample_rate}Hz")
            # Simple resampling (in practice, use proper resampling)
            audio = audio[::sr//args.sample_rate]
        
        # Process audio
        inference._predict_digit(audio)
        
    else:
        # Start real-time inference
        inference.start()

if __name__ == '__main__':
    main()