#!/usr/bin/env python3
"""
Hyperparameter tuning script for the CNN-based spoken digit recognition system.

This script performs systematic hyperparameter optimization using grid search
and random search to find the best configuration for the CNN model.
"""

import os
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, List, Tuple
from itertools import product
import json

# Suppress warnings
warnings.filterwarnings('ignore')

# Import our modules
from data_loader import FSDDDataLoader
from features import MFCCFeatureExtractor
from model import CompactCNN

class HyperparameterTuner:
    """
    Hyperparameter tuning for CNN-based spoken digit recognition.
    """
    
    def __init__(self, n_trials: int = 20, cv_folds: int = 3):
        """
        Initialize the hyperparameter tuner.
        
        Args:
            n_trials: Number of random search trials
            cv_folds: Number of cross-validation folds
        """
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.results = []
        self.best_config = None
        self.best_score = 0
        
        # Define search spaces
        self.search_spaces = {
            'dropout_rate': [0.1, 0.2, 0.25, 0.3, 0.4, 0.5],
            'l2_reg': [1e-5, 1e-4, 1e-3, 1e-2],
            'learning_rate': [0.0001, 0.001, 0.01, 0.1],
            'batch_size': [16, 32, 64, 128],
            'n_mfcc': [13, 20, 26, 40],
            'use_class_weights': [True, False],
            'use_focal_loss': [True, False],
            'use_spec_augment': [True, False]
        }
        
        print(f"Initialized HyperparameterTuner:")
        print(f"  - Random search trials: {n_trials}")
        print(f"  - Cross-validation folds: {cv_folds}")
        print(f"  - Search space size: {np.prod([len(v) for v in self.search_spaces.values()])}")
    
    def load_data(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Load and prepare data for hyperparameter tuning.
        
        Returns:
            Dictionary containing data splits
        """
        print("Loading data for hyperparameter tuning...")
        
        # Load data
        data_loader = FSDDDataLoader(target_sr=8000, duration=1.0)
        audio_data, labels = data_loader.load_dataset()
        
        # Create splits
        splits = data_loader.stratified_split(
            audio_data, labels,
            test_size=0.2,
            val_size=0.1
        )
        
        return splits
    
    def generate_random_config(self) -> Dict[str, Any]:
        """
        Generate a random configuration from the search space.
        
        Returns:
            Random hyperparameter configuration
        """
        config = {}
        for param, values in self.search_spaces.items():
            config[param] = np.random.choice(values)
        
        return config
    
    def evaluate_config(self, config: Dict[str, Any], 
                       data_splits: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> Dict[str, Any]:
        """
        Evaluate a single hyperparameter configuration.
        
        Args:
            config: Hyperparameter configuration
            data_splits: Data splits for training/validation
            
        Returns:
            Dictionary containing evaluation results
        """
        print(f"Evaluating config: {config}")
        
        try:
            # Get data
            X_train, y_train = data_splits['train']
            X_val, y_val = data_splits['val']
            
            # Initialize feature extractor
            feature_extractor = MFCCFeatureExtractor(
                target_sr=8000,
                duration=1.0,
                n_mfcc=config['n_mfcc']
            )
            
            # Extract features
            X_train_features, y_train_features = feature_extractor.extract_features_with_augmentation(
                X_train, y_train, use_spec_augment=config['use_spec_augment']
            )
            X_val_features, y_val_features = feature_extractor.extract_features_batch(X_val)
            
            # Normalize features
            X_train_norm, mean, std = feature_extractor.normalize_features(X_train_features)
            X_val_norm, _, _ = feature_extractor.normalize_features(X_val_features, mean, std)
            
            # Initialize model
            input_shape = X_train_norm.shape[1:]
            model = CompactCNN(
                input_shape=input_shape,
                num_classes=10,
                dropout_rate=config['dropout_rate'],
                l2_reg=config['l2_reg']
            )
            
            # Build model with appropriate loss
            if config['use_focal_loss']:
                model.build_model_with_focal_loss()
            else:
                model.build_model()
            
            # Update learning rate
            model.model.compile(
                optimizer=Adam(learning_rate=config['learning_rate']),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # Train model
            start_time = time.time()
            history = model.train(
                X_train_norm, y_train_features,
                X_val_norm, y_val_features,
                epochs=20,  # Reduced for faster tuning
                batch_size=config['batch_size'],
                use_class_weights=config['use_class_weights'],
                use_focal_loss=config['use_focal_loss']
            )
            training_time = time.time() - start_time
            
            # Get best validation accuracy
            best_val_acc = max(history.history['val_accuracy'])
            
            # Calculate inference speed
            inference_times = []
            for i in range(10):  # Test on 10 samples
                sample = X_val_norm[i]
                _, _, inference_time = model.predict_single(sample)
                inference_times.append(inference_time)
            
            mean_inference_time = np.mean(inference_times)
            
            # Calculate score (accuracy with speed penalty)
            speed_penalty = max(0, (mean_inference_time - 20) / 20)  # Penalty if >20ms
            score = best_val_acc - 0.1 * speed_penalty  # 10% penalty for speed
            
            result = {
                'config': config.copy(),
                'val_accuracy': best_val_acc,
                'training_time': training_time,
                'inference_time': mean_inference_time,
                'score': score,
                'history': history.history
            }
            
            print(f"  Val Accuracy: {best_val_acc:.4f}")
            print(f"  Inference Time: {mean_inference_time:.2f}ms")
            print(f"  Score: {score:.4f}")
            
            return result
            
        except Exception as e:
            print(f"  Error evaluating config: {e}")
            return {
                'config': config.copy(),
                'val_accuracy': 0.0,
                'training_time': 0.0,
                'inference_time': 1000.0,
                'score': 0.0,
                'error': str(e)
            }
    
    def random_search(self, data_splits: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> List[Dict[str, Any]]:
        """
        Perform random search hyperparameter optimization.
        
        Args:
            data_splits: Data splits for training/validation
            
        Returns:
            List of evaluation results
        """
        print(f"\nStarting random search with {self.n_trials} trials...")
        
        results = []
        
        for trial in range(self.n_trials):
            print(f"\nTrial {trial + 1}/{self.n_trials}")
            
            # Generate random configuration
            config = self.generate_random_config()
            
            # Evaluate configuration
            result = self.evaluate_config(config, data_splits)
            results.append(result)
            
            # Update best configuration
            if result['score'] > self.best_score:
                self.best_score = result['score']
                self.best_config = config.copy()
                print(f"  New best score: {self.best_score:.4f}")
        
        self.results = results
        return results
    
    def grid_search(self, data_splits: Dict[str, Tuple[np.ndarray, np.ndarray]], 
                   limited_params: Dict[str, List] = None) -> List[Dict[str, Any]]:
        """
        Perform grid search on a limited parameter space.
        
        Args:
            data_splits: Data splits for training/validation
            limited_params: Limited parameter space for grid search
            
        Returns:
            List of evaluation results
        """
        if limited_params is None:
            limited_params = {
                'dropout_rate': [0.2, 0.25, 0.3],
                'l2_reg': [1e-4, 1e-3],
                'learning_rate': [0.001, 0.01],
                'batch_size': [32, 64],
                'n_mfcc': [20, 26],
                'use_class_weights': [True],
                'use_focal_loss': [False],
                'use_spec_augment': [True]
            }
        
        print(f"\nStarting grid search...")
        print(f"Parameter combinations: {np.prod([len(v) for v in limited_params.values()])}")
        
        results = []
        param_names = list(limited_params.keys())
        param_values = list(limited_params.values())
        
        for i, combination in enumerate(product(*param_values)):
            print(f"\nGrid search {i + 1}/{len(list(product(*param_values)))}")
            
            # Create configuration
            config = dict(zip(param_names, combination))
            
            # Evaluate configuration
            result = self.evaluate_config(config, data_splits)
            results.append(result)
            
            # Update best configuration
            if result['score'] > self.best_score:
                self.best_score = result['score']
                self.best_config = config.copy()
                print(f"  New best score: {self.best_score:.4f}")
        
        self.results = results
        return results
    
    def analyze_results(self) -> pd.DataFrame:
        """
        Analyze hyperparameter tuning results.
        
        Returns:
            DataFrame containing results analysis
        """
        if not self.results:
            raise ValueError("No results to analyze. Run tuning first.")
        
        # Convert results to DataFrame
        df_results = []
        for result in self.results:
            row = result['config'].copy()
            row.update({
                'val_accuracy': result['val_accuracy'],
                'training_time': result['training_time'],
                'inference_time': result['inference_time'],
                'score': result['score']
            })
            df_results.append(row)
        
        df = pd.DataFrame(df_results)
        
        # Sort by score
        df = df.sort_values('score', ascending=False)
        
        print("\n" + "="*60)
        print("HYPERPARAMETER TUNING RESULTS")
        print("="*60)
        
        print(f"Total trials: {len(df)}")
        print(f"Best score: {self.best_score:.4f}")
        print(f"Best validation accuracy: {df.iloc[0]['val_accuracy']:.4f}")
        print(f"Best inference time: {df.iloc[0]['inference_time']:.2f}ms")
        
        print("\nTop 5 configurations:")
        print(df.head()[['val_accuracy', 'inference_time', 'score']].to_string())
        
        print("\nBest configuration:")
        best_config = df.iloc[0]
        for param, value in best_config.items():
            if param not in ['val_accuracy', 'training_time', 'inference_time', 'score']:
                print(f"  {param}: {value}")
        
        return df
    
    def plot_results(self, df: pd.DataFrame):
        """
        Create visualization plots for hyperparameter tuning results.
        
        Args:
            df: Results DataFrame
        """
        print("\nCreating hyperparameter tuning plots...")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Hyperparameter Tuning Results', fontsize=16)
        
        # 1. Score distribution
        ax1 = axes[0, 0]
        ax1.hist(df['score'], bins=20, alpha=0.7, color='skyblue')
        ax1.axvline(df['score'].max(), color='red', linestyle='--', label='Best')
        ax1.set_title('Score Distribution')
        ax1.set_xlabel('Score')
        ax1.set_ylabel('Frequency')
        ax1.legend()
        
        # 2. Accuracy vs Inference Time
        ax2 = axes[0, 1]
        scatter = ax2.scatter(df['inference_time'], df['val_accuracy'], 
                             c=df['score'], cmap='viridis', alpha=0.7)
        ax2.axvline(20, color='red', linestyle='--', alpha=0.7, label='20ms target')
        ax2.set_title('Accuracy vs Inference Time')
        ax2.set_xlabel('Inference Time (ms)')
        ax2.set_ylabel('Validation Accuracy')
        ax2.legend()
        plt.colorbar(scatter, ax=ax2, label='Score')
        
        # 3. Dropout rate analysis
        ax3 = axes[0, 2]
        dropout_analysis = df.groupby('dropout_rate')['score'].mean()
        ax3.bar(dropout_analysis.index, dropout_analysis.values, color='lightcoral')
        ax3.set_title('Score by Dropout Rate')
        ax3.set_xlabel('Dropout Rate')
        ax3.set_ylabel('Mean Score')
        
        # 4. Learning rate analysis
        ax4 = axes[1, 0]
        lr_analysis = df.groupby('learning_rate')['score'].mean()
        ax4.bar(range(len(lr_analysis)), lr_analysis.values, color='lightgreen')
        ax4.set_title('Score by Learning Rate')
        ax4.set_xlabel('Learning Rate')
        ax4.set_ylabel('Mean Score')
        ax4.set_xticks(range(len(lr_analysis)))
        ax4.set_xticklabels([f'{lr:.4f}' for lr in lr_analysis.index], rotation=45)
        
        # 5. Batch size analysis
        ax5 = axes[1, 1]
        batch_analysis = df.groupby('batch_size')['score'].mean()
        ax5.bar(batch_analysis.index, batch_analysis.values, color='gold')
        ax5.set_title('Score by Batch Size')
        ax5.set_xlabel('Batch Size')
        ax5.set_ylabel('Mean Score')
        
        # 6. MFCC coefficients analysis
        ax6 = axes[1, 2]
        mfcc_analysis = df.groupby('n_mfcc')['score'].mean()
        ax6.bar(mfcc_analysis.index, mfcc_analysis.values, color='plum')
        ax6.set_title('Score by MFCC Coefficients')
        ax6.set_xlabel('Number of MFCC Coefficients')
        ax6.set_ylabel('Mean Score')
        
        plt.tight_layout()
        plt.savefig('hyperparameter_tuning_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Hyperparameter tuning plots saved to hyperparameter_tuning_results.png")
    
    def save_results(self, df: pd.DataFrame, filename: str = 'hyperparameter_tuning_results.csv'):
        """
        Save hyperparameter tuning results to file.
        
        Args:
            df: Results DataFrame
            filename: Output filename
        """
        df.to_csv(filename, index=False)
        print(f"Results saved to {filename}")
        
        # Save best configuration as JSON
        best_config = df.iloc[0].to_dict()
        with open('best_hyperparameters.json', 'w') as f:
            json.dump(best_config, f, indent=2)
        print("Best configuration saved to best_hyperparameters.json")

def main():
    """
    Main function for hyperparameter tuning.
    """
    print("=" * 80)
    print("HYPERPARAMETER TUNING FOR CNN SPOKEN DIGIT RECOGNITION")
    print("=" * 80)
    
    # Initialize tuner
    tuner = HyperparameterTuner(n_trials=30, cv_folds=3)
    
    # Load data
    data_splits = tuner.load_data()
    
    # Perform random search
    print("\n" + "="*50)
    print("RANDOM SEARCH OPTIMIZATION")
    print("="*50)
    results = tuner.random_search(data_splits)
    
    # Analyze results
    df = tuner.analyze_results()
    
    # Create plots
    tuner.plot_results(df)
    
    # Save results
    tuner.save_results(df)
    
    # Print final summary
    print("\n" + "="*80)
    print("HYPERPARAMETER TUNING COMPLETED")
    print("="*80)
    print(f"Best configuration found:")
    for param, value in tuner.best_config.items():
        print(f"  {param}: {value}")
    print(f"\nBest score: {tuner.best_score:.4f}")
    print(f"Best validation accuracy: {df.iloc[0]['val_accuracy']:.4f}")
    print(f"Best inference time: {df.iloc[0]['inference_time']:.2f}ms")
    
    return tuner.best_config, df

if __name__ == "__main__":
    main()