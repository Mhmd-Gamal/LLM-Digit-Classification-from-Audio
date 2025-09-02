#!/usr/bin/env python3
"""
CNN model architecture for spoken digit recognition.

This module implements a compact 2D-CNN architecture that treats MFCCs as
spectro-temporal images for improved accuracy while maintaining fast inference.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, GlobalAveragePooling2D, 
    BatchNormalization, Dropout, Dense, Reshape
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical
from typing import Tuple, Dict, Any

class CompactCNN:
    """
    Compact 2D-CNN for spoken digit recognition with spectro-temporal features.
    """
    
    def __init__(self, input_shape: Tuple[int, int, int], num_classes: int = 10,
                 dropout_rate: float = 0.25, l2_reg: float = 1e-4):
        """
        Initialize the CNN model.
        
        Args:
            input_shape: Input shape (channels, height, width) for spectro-temporal features
            num_classes: Number of output classes (digits 0-9)
            dropout_rate: Dropout rate for regularization
            l2_reg: L2 regularization factor
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        self.model = None
        
        print(f"Initialized CompactCNN:")
        print(f"  - Input shape: {input_shape}")
        print(f"  - Number of classes: {num_classes}")
        print(f"  - Dropout rate: {dropout_rate}")
        print(f"  - L2 regularization: {l2_reg}")
    
    def build_model(self) -> tf.keras.Model:
        """
        Build the compact CNN architecture.
        
        Architecture:
        - Input: (1, n_mfcc*3, time_frames) - spectro-temporal image
        - Conv2D(32) + BatchNorm + ReLU + MaxPool
        - Conv2D(64) + BatchNorm + ReLU + MaxPool  
        - Conv2D(128) + BatchNorm + ReLU + GlobalAvgPool
        - Dense(64) + Dropout + ReLU
        - Dense(10) + Softmax
        
        Returns:
            Compiled Keras model
        """
        print("Building compact CNN architecture...")
        
        model = Sequential([
            # Reshape input to add channel dimension for Conv2D
            Reshape((*self.input_shape[1:], 1), input_shape=self.input_shape),
            
            # First convolutional block
            Conv2D(32, (3, 3), activation='relu', 
                   kernel_regularizer=l2(self.l2_reg),
                   padding='same'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(self.dropout_rate),
            
            # Second convolutional block
            Conv2D(64, (3, 3), activation='relu',
                   kernel_regularizer=l2(self.l2_reg),
                   padding='same'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(self.dropout_rate),
            
            # Third convolutional block
            Conv2D(128, (3, 3), activation='relu',
                   kernel_regularizer=l2(self.l2_reg),
                   padding='same'),
            BatchNormalization(),
            GlobalAveragePooling2D(),
            Dropout(self.dropout_rate),
            
            # Dense layers
            Dense(64, activation='relu',
                  kernel_regularizer=l2(self.l2_reg)),
            Dropout(self.dropout_rate),
            
            # Output layer
            Dense(self.num_classes, activation='softmax')
        ])
        
        # Compile the model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("CNN Model architecture:")
        model.summary()
        
        # Count parameters
        total_params = model.count_params()
        print(f"Total parameters: {total_params:,}")
        
        if total_params > 150000:
            print("WARNING: Model exceeds 150k parameter limit!")
        else:
            print("✓ Model within 150k parameter limit")
        
        self.model = model
        return model
    
    def build_model_with_focal_loss(self, alpha: float = 0.25, gamma: float = 2.0) -> tf.keras.Model:
        """
        Build model with focal loss for handling class imbalance.
        
        Args:
            alpha: Weighting factor for rare class
            gamma: Focusing parameter for hard examples
            
        Returns:
            Compiled Keras model with focal loss
        """
        print("Building CNN with focal loss...")
        
        # Build the base model
        model = self.build_model()
        
        # Define focal loss
        def focal_loss(y_true, y_pred):
            epsilon = tf.keras.backend.epsilon()
            y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
            
            # Calculate focal loss
            alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
            p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
            focal_weight = alpha_t * tf.pow((1 - p_t), gamma)
            
            # Calculate cross entropy
            ce = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
            
            return focal_weight * ce
        
        # Recompile with focal loss
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss=focal_loss,
            metrics=['accuracy']
        )
        
        print("Model compiled with focal loss")
        return model
    
    def get_class_weights(self, labels: np.ndarray) -> Dict[int, float]:
        """
        Calculate class weights for handling class imbalance.
        
        Args:
            labels: Array of class labels
            
        Returns:
            Dictionary mapping class indices to weights
        """
        from sklearn.utils.class_weight import compute_class_weight
        
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(labels),
            y=labels
        )
        
        class_weight_dict = dict(zip(np.unique(labels), class_weights))
        
        print("Class weights:")
        for class_idx, weight in class_weight_dict.items():
            print(f"  Class {class_idx}: {weight:.3f}")
        
        return class_weight_dict
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray,
              epochs: int = 50, batch_size: int = 32,
              use_class_weights: bool = True,
              use_focal_loss: bool = False) -> tf.keras.callbacks.History:
        """
        Train the CNN model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            epochs: Number of training epochs
            batch_size: Training batch size
            use_class_weights: Whether to use class weights
            use_focal_loss: Whether to use focal loss
            
        Returns:
            Training history
        """
        if self.model is None:
            if use_focal_loss:
                self.build_model_with_focal_loss()
            else:
                self.build_model()
        
        # Calculate class weights if requested
        class_weights = None
        if use_class_weights and not use_focal_loss:
            class_weights = self.get_class_weights(y_train)
        
        # Set up callbacks
        callbacks = self._get_training_callbacks()
        
        print("Starting CNN training...")
        
        # Train the model
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def _get_training_callbacks(self) -> list:
        """
        Get training callbacks for optimization.
        
        Returns:
            List of Keras callbacks
        """
        from tensorflow.keras.callbacks import (
            EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
        )
        
        callbacks = [
            # Early stopping
            EarlyStopping(
                monitor='val_accuracy',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            
            # Learning rate reduction
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=8,
                min_lr=1e-6,
                verbose=1
            ),
            
            # Model checkpoint
            ModelCheckpoint(
                'best_cnn_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        return callbacks
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate the trained model.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary containing evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model must be trained before evaluation")
        
        print("Evaluating CNN model...")
        
        # Make predictions
        y_pred_proba = self.model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Calculate metrics
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, 
            f1_score, confusion_matrix, classification_report
        )
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        print(f"\nCNN Test Results:")
        print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        
        # Classification report
        print("\nDetailed Classification Report:")
        print(classification_report(y_test, y_pred, 
                                  target_names=[str(i) for i in range(10)]))
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'predictions': y_pred,
            'true_labels': y_test,
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
    
    def predict_single(self, features: np.ndarray) -> Tuple[int, float, float]:
        """
        Predict digit from a single feature sample.
        
        Args:
            features: Spectro-temporal features of shape (n_mfcc*3, time_frames)
            
        Returns:
            Tuple of (predicted_digit, confidence, inference_time_ms)
        """
        if self.model is None:
            raise ValueError("Model must be trained before prediction")
        
        import time
        
        # Measure inference time
        start_time = time.time()
        
        # Add batch and channel dimensions
        features_batch = features.reshape(1, *features.shape, 1)
        
        # Make prediction
        prediction_proba = self.model.predict(features_batch, verbose=0)
        predicted_digit = np.argmax(prediction_proba)
        confidence = np.max(prediction_proba)
        
        # Calculate inference time
        inference_time = (time.time() - start_time) * 1000  # Convert to ms
        
        return predicted_digit, confidence, inference_time
    
    def save_model(self, filepath: str):
        """Save the trained model to disk."""
        if self.model is None:
            raise ValueError("Model must be trained before saving")
        
        self.model.save(filepath)
        print(f"CNN model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model from disk."""
        self.model = tf.keras.models.load_model(filepath)
        print(f"CNN model loaded from {filepath}")