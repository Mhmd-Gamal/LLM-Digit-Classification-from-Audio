#!/usr/bin/env python3
"""
Compact PyTorch CNN Student Model for Spoken Digit Recognition.

This module implements a lightweight CNN architecture designed for knowledge
distillation with ≤90k parameters while maintaining high accuracy.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import time

class CompactStudentCNN(nn.Module):
    """
    Compact CNN student model for spoken digit recognition.
    
    Architecture designed for knowledge distillation with ≤90k parameters.
    Uses depthwise separable convolutions and efficient pooling strategies.
    """
    
    def __init__(self, input_shape: Tuple[int, int, int] = (1, 60, 80), 
                 num_classes: int = 10, dropout_rate: float = 0.2):
        """
        Initialize the compact student CNN.
        
        Args:
            input_shape: Input shape (channels, height, width)
            num_classes: Number of output classes (digits 0-9)
            dropout_rate: Dropout rate for regularization
        """
        super(CompactStudentCNN, self).__init__()
        
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        
        # Calculate parameters for efficient architecture
        # Target: ≤90k parameters
        
        # First conv block - reduce spatial dimensions
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_shape[0], 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Second conv block - depthwise separable conv for efficiency
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1, groups=16),  # Depthwise
            nn.Conv2d(16, 32, kernel_size=1),  # Pointwise
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Third conv block - more efficient with smaller filters
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1, groups=16),  # Depthwise
            nn.Conv2d(32, 48, kernel_size=1),  # Pointwise
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))  # Global average pooling
        )
        
        # Calculate flattened size after convolutions
        # Input: (1, 60, 80) -> conv1: (16, 30, 40) -> conv2: (32, 15, 20) -> conv3: (48, 1, 1)
        self.flatten_size = 48
        
        # Dense layers with reduced parameters
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.flatten_size, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(32, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
        
        # Count parameters
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Student CNN initialized:")
        print(f"  - Input shape: {input_shape}")
        print(f"  - Total parameters: {total_params:,}")
        print(f"  - Target: ≤90,000 parameters")
        
        if total_params > 90000:
            print(f"⚠️  WARNING: Model exceeds 90k parameter limit! ({total_params:,} params)")
        else:
            print(f"✓ Model within 90k parameter limit")
    
    def _initialize_weights(self):
        """Initialize model weights for better training."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the student network.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Logits tensor of shape (batch_size, num_classes)
        """
        # Convolutional layers
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        # Flatten and classify
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        
        return x
    
    def get_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Get logits without softmax for knowledge distillation."""
        return self.forward(x)
    
    def get_probabilities(self, x: torch.Tensor) -> torch.Tensor:
        """Get probabilities with softmax."""
        logits = self.forward(x)
        return F.softmax(logits, dim=1)
    
    def count_parameters(self) -> int:
        """Count total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def benchmark_inference(self, input_tensor: torch.Tensor, 
                           num_runs: int = 100) -> dict:
        """
        Benchmark inference speed of the model.
        
        Args:
            input_tensor: Input tensor for benchmarking
            num_runs: Number of inference runs for averaging
            
        Returns:
            Dictionary with timing statistics
        """
        self.eval()
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = self.forward(input_tensor)
        
        # Benchmark
        times = []
        with torch.no_grad():
            for _ in range(num_runs):
                start_time = time.time()
                _ = self.forward(input_tensor)
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                end_time = time.time()
                times.append((end_time - start_time) * 1000)  # Convert to ms
        
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        
        return {
            'avg_inference_time_ms': avg_time,
            'min_inference_time_ms': min_time,
            'max_inference_time_ms': max_time,
            'std_inference_time_ms': torch.tensor(times).std().item()
        }

class TeacherCNN(nn.Module):
    """
    Teacher CNN model for knowledge distillation.
    
    This is a larger model that will guide the student during training.
    """
    
    def __init__(self, input_shape: Tuple[int, int, int] = (1, 60, 80), 
                 num_classes: int = 10):
        """
        Initialize the teacher CNN.
        
        Args:
            input_shape: Input shape (channels, height, width)
            num_classes: Number of output classes
        """
        super(TeacherCNN, self).__init__()
        
        self.input_shape = input_shape
        self.num_classes = num_classes
        
        # Larger architecture for teacher
        self.features = nn.Sequential(
            # First block
            nn.Conv2d(input_shape[0], 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.25),
            
            # Second block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.25),
            
            # Third block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Dropout2d(0.25)
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(64, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
        
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Teacher CNN initialized with {total_params:,} parameters")
    
    def _initialize_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the teacher network."""
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
    def get_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Get logits without softmax."""
        return self.forward(x)
    
    def get_probabilities(self, x: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """Get probabilities with temperature scaling for distillation."""
        logits = self.forward(x)
        return F.softmax(logits / temperature, dim=1)

def create_student_model(input_shape: Tuple[int, int, int] = (1, 60, 80), 
                        num_classes: int = 10) -> CompactStudentCNN:
    """
    Factory function to create a student model.
    
    Args:
        input_shape: Input shape (channels, height, width)
        num_classes: Number of output classes
        
    Returns:
        Initialized student model
    """
    return CompactStudentCNN(input_shape=input_shape, num_classes=num_classes)

def create_teacher_model(input_shape: Tuple[int, int, int] = (1, 60, 80), 
                        num_classes: int = 10) -> TeacherCNN:
    """
    Factory function to create a teacher model.
    
    Args:
        input_shape: Input shape (channels, height, width)
        num_classes: Number of output classes
        
    Returns:
        Initialized teacher model
    """
    return TeacherCNN(input_shape=input_shape, num_classes=num_classes)