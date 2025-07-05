#!/usr/bin/env python3
"""
Model Comparison Framework
Handles training, evaluation, and comparison of multiple ML models
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score, 
                           f1_score, precision_score, recall_score, roc_auc_score,
                           precision_recall_curve, roc_curve, auc)
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
import time
import warnings
warnings.filterwarnings('ignore')

class ModelComparator:
    """Framework for comprehensive model comparison"""
    
    def __init__(self):
        self.models = {}
        self.results = {}
        self.trained_models = {}
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.label_encoder = None
    
    def initialize_models(self):
        """Initialize all models with their configurations"""
        self.models = {
            'logistic_regression': {
                'model': LogisticRegression(random_state=42, max_iter=2000, class_weight='balanced'),
                'name': 'Logistic Regression',
                'params': {
                    'C': [0.1, 1.0, 10.0],
                    'solver': ['liblinear', 'lbfgs']
                }
            },
            'svm_linear': {
                'model': SVC(kernel='linear', random_state=42, class_weight='balanced', probability=True),
                'name': 'SVM (Linear)',
                'params': {
                    'C': [0.1, 1.0, 10.0],
                    'gamma': ['scale', 'auto']
                }
            },
            'svm_rbf': {
                'model': SVC(kernel='rbf', random_state=42, class_weight='balanced', probability=True),
                'name': 'SVM (RBF)',
                'params': {
                    'C': [0.1, 1.0, 10.0],
                    'gamma': ['scale', 'auto']
                }
            },
            'random_forest': {
                'model': RandomForestClassifier(random_state=42, class_weight='balanced'),
                'name': 'Random Forest',
                'params': {
                    'n_estimators': [100, 200],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5]
                }
            },
            'gradient_boosting': {
                'model': GradientBoostingClassifier(random_state=42),
                'name': 'Gradient Boosting',
                'params': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.1, 0.2],
                    'max_depth': [3, 5]
                }
            },
            'naive_bayes': {
                'model': MultinomialNB(),
                'name': 'Naive Bayes',
                'params': {
                    'alpha': [0.1, 1.0, 10.0]
                }
            },
            'knn': {
                'model': KNeighborsClassifier(),
                'name': 'K-Nearest Neighbors',
                'params': {
                    'n_neighbors': [3, 5, 7],
                    'weights': ['uniform', 'distance']
                }
            },
            'decision_tree': {
                'model': DecisionTreeClassifier(random_state=42, class_weight='balanced'),
                'name': 'Decision Tree',
                'params': {
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5, 10]
                }
            }
        }
    
    def train_single_model(self, model_key, X_train, X_test, y_train, y_test, use_grid_search=True):
        """Train and evaluate a single model"""
        model_info = self.models[model_key]
        model_name = model_info['name']
        
        print(f"\nTraining {model_name}...")
        start_time = time.time()
        
        if use_grid_search and len(model_info['params']) > 0:
            # Use GridSearchCV for hyperparameter tuning
            grid_search = GridSearchCV(
                model_info['model'],
                model_info['params'],
                cv=3,
                scoring='f1_macro',
                n_jobs=-1
            )
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
        else:
            # Train with default parameters
            best_model = model_info['model']
            best_model.fit(X_train, y_train)
            best_params = "Default"
        
        training_time = time.time() - start_time
        
        # Make predictions
        start_time = time.time()
        y_pred = best_model.predict(X_test)
        y_pred_proba = None
        
        if hasattr(best_model, 'predict_proba'):
            y_pred_proba = best_model.predict_proba(X_test)
        
        prediction_time = time.time() - start_time
        
        # Calculate comprehensive metrics
        metrics = self.calculate_metrics(y_test, y_pred, y_pred_proba)
        
        # Detailed classification report
        report = classification_report(
            y_test, y_pred,
            target_names=self.label_encoder.classes_ if self.label_encoder else None,
            output_dict=True,
            zero_division=0
        )
        
        # Per-class metrics
        per_class_metrics = self.extract_per_class_metrics(report)
        
        # Store results
        results = {
            'model_name': model_name,
            'best_model': best_model,
            'best_params': best_params,
            'training_time': training_time,
            'prediction_time': prediction_time,
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'classification_report': report,
            'per_class_metrics': per_class_metrics,
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            **metrics
        }
        
        # Store trained model
        self.trained_models[model_key] = best_model
        
        # Print summary
        self.print_model_summary(model_name, results)
        
        return results
    
    def calculate_metrics(self, y_true, y_pred, y_pred_proba=None):
        """Calculate comprehensive evaluation metrics"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'macro_f1': f1_score(y_true, y_pred, average='macro'),
            'weighted_f1': f1_score(y_true, y_pred, average='weighted'),
            'macro_precision': precision_score(y_true, y_pred, average='macro'),
            'macro_recall': recall_score(y_true, y_pred, average='macro'),
            'weighted_precision': precision_score(y_true, y_pred, average='weighted'),
            'weighted_recall': recall_score(y_true, y_pred, average='weighted')
        }
        
        # Calculate AUC if probabilities available
        if y_pred_proba is not None:
            try:
                # Binarize labels for multiclass AUC
                y_true_bin = label_binarize(y_true, classes=np.unique(y_true))
                if y_true_bin.shape[1] > 1:
                    metrics['macro_auc'] = roc_auc_score(y_true_bin, y_pred_proba, 
                                                        average='macro', multi_class='ovr')
                    metrics['weighted_auc'] = roc_auc_score(y_true_bin, y_pred_proba, 
                                                           average='weighted', multi_class='ovr')
            except Exception as e:
                metrics['macro_auc'] = None
                metrics['weighted_auc'] = None
        else:
            metrics['macro_auc'] = None
            metrics['weighted_auc'] = None
        
        return metrics
    
    def extract_per_class_metrics(self, report):
        """Extract per-class metrics from classification report"""
        per_class_metrics = {}
        
        if self.label_encoder:
            class_names = self.label_encoder.classes_
        else:
            class_names = [str(i) for i in range(len(report) - 3)]  # Exclude macro/weighted/accuracy
        
        for class_name in class_names:
            if class_name in report:
                per_class_metrics[class_name] = {
                    'precision': report[class_name]['precision'],
                    'recall': report[class_name]['recall'],
                    'f1_score': report[class_name]['f1-score'],
                    'support': report[class_name]['support']
                }
        
        return per_class_metrics
    
    def print_model_summary(self, model_name, results):
        """Print summary of model performance"""
        print(f"  Training completed in {results['training_time']:.2f}s")
        print(f"  Accuracy: {results['accuracy']:.4f}")
        print(f"  Macro F1: {results['macro_f1']:.4f}")
        print(f"  Weighted F1: {results['weighted_f1']:.4f}")
        if results['macro_auc']:
            print(f"  Macro AUC: {results['macro_auc']:.4f}")
    
    def compare_all_models(self, X, y, test_size=0.2, use_grid_search=True, label_encoder=None):
        """Compare all models with comprehensive evaluation"""
        print("\n" + "="*80)
        print("COMPREHENSIVE MODEL COMPARISON")
        print("="*80)
        
        self.label_encoder = label_encoder
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        
        # Store splits for ensemble use
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
        # Initialize models
        self.initialize_models()
        
        # Train and evaluate each model
        for model_key in self.models.keys():
            try:
                results = self.train_single_model(
                    model_key, X_train, X_test, y_train, y_test, use_grid_search
                )
                self.results[model_key] = results
            except Exception as e:
                print(f"Error training {self.models[model_key]['name']}: {e}")
        
        return X_test, y_test
    
    def create_ensemble_model(self):
        """Create and evaluate ensemble model from top performers"""
        print(f"\nCreating Ensemble Model...")
        
        # Check if we have enough successful models
        if len(self.results) < 2:
            print("Error: Need at least 2 trained models for ensemble.")
            return None
        
        X_train, X_test, y_train, y_test = self.X_train, self.X_test, self.y_train, self.y_test
        
        # Select top models based on macro F1 (up to 3)
        available_models = [(k, v) for k, v in self.results.items() if 'best_model' in v]
        
        top_models = sorted(
            available_models,
            key=lambda x: x[1]['macro_f1'],
            reverse=True
        )[:min(3, len(available_models))]
        
        estimators = []
        for model_key, results in top_models:
            try:
                estimators.append((model_key, results['best_model']))
            except Exception as e:
                print(f"Warning: Could not add {results['model_name']} to ensemble: {e}")
        
        if len(estimators) < 2:
            print("Error: Need at least 2 valid models for ensemble.")
            return None
        
        # Create voting classifier
        try:
            ensemble = VotingClassifier(estimators=estimators, voting='soft')
            
            start_time = time.time()
            ensemble.fit(X_train, y_train)
            training_time = time.time() - start_time
            
            # Evaluate ensemble
            start_time = time.time()
            y_pred = ensemble.predict(X_test)
            prediction_time = time.time() - start_time
            
            # Calculate metrics
            metrics = self.calculate_metrics(y_test, y_pred)
            
            ensemble_results = {
                'model_name': 'Ensemble (Top Models)',
                'training_time': training_time,
                'prediction_time': prediction_time,
                'component_models': [results['model_name'] for _, results in top_models],
                **metrics
            }
            
            self.results['ensemble'] = ensemble_results
            
            print(f"  Ensemble created with: {', '.join(ensemble_results['component_models'])}")
            print(f"  Accuracy: {ensemble_results['accuracy']:.4f}")
            print(f"  Macro F1: {ensemble_results['macro_f1']:.4f}")
            
            return ensemble
            
        except Exception as e:
            print(f"Error creating ensemble: {e}")
            return None
    
    def get_best_model(self):
        """Get the best performing model"""
        if not self.results:
            return None, None
        
        # Find best model by macro F1 (excluding ensemble for now)
        available_models = {k: v for k, v in self.results.items() 
                          if 'best_model' in v and k != 'ensemble'}
        
        if not available_models:
            return None, None
        
        best_model_key = max(available_models.keys(), 
                           key=lambda k: available_models[k]['macro_f1'])
        
        return best_model_key, available_models[best_model_key]
    
    def print_detailed_comparison(self):
        """Print detailed comparison of all models"""
        print("\n" + "="*100)
        print("DETAILED MODEL COMPARISON RESULTS")
        print("="*100)
        
        # Overall comparison table
        print(f"\n{'Model':<20} {'Accuracy':<10} {'Macro F1':<10} {'Weighted F1':<12} {'Macro AUC':<10} {'Train Time':<12}")
        print("-" * 85)
        
        sorted_results = sorted(
            self.results.items(),
            key=lambda x: x[1]['macro_f1'],
            reverse=True
        )
        
        for model_key, results in sorted_results:
            auc_str = f"{results['macro_auc']:.4f}" if results.get('macro_auc') else "N/A"
            print(f"{results['model_name']:<20} {results['accuracy']:<10.4f} "
                  f"{results['macro_f1']:<10.4f} {results['weighted_f1']:<12.4f} "
                  f"{auc_str:<10} {results['training_time']:<12.2f}s")
        
        # Best model analysis
        best_model_key, best_results = self.get_best_model()
        if best_results:
            print(f"\n🏆 BEST MODEL: {best_results['model_name']}")
            print(f"   Macro F1 Score: {best_results['macro_f1']:.4f}")
            print(f"   Accuracy: {best_results['accuracy']:.4f}")
            print(f"   Training Time: {best_results['training_time']:.2f}s")
            
            if 'best_params' in best_results and best_results['best_params'] != "Default":
                print(f"   Best Parameters: {best_results['best_params']}")
            
            # Per-class performance for best model
            if 'per_class_metrics' in best_results:
                self.print_per_class_performance(best_results['per_class_metrics'])
        
        # Model insights
        self.print_model_insights()
    
    def print_per_class_performance(self, per_class_metrics):
        """Print per-class performance metrics"""
        print(f"\nPer-Class Performance (Best Model):")
        print(f"{'Intent':<25} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")
        print("-" * 70)
        
        for class_name, metrics in per_class_metrics.items():
            print(f"{class_name:<25} {metrics['precision']:<10.3f} "
                  f"{metrics['recall']:<10.3f} {metrics['f1_score']:<10.3f} "
                  f"{int(metrics['support']):<10}")
    
    def print_model_insights(self):
        """Print model insights and comparisons"""
        print(f"\n📊 MODEL INSIGHTS:")
        
        # Best performing models by category
        best_accuracy = max(self.results.items(), key=lambda x: x[1]['accuracy'])
        best_macro_f1 = max(self.results.items(), key=lambda x: x[1]['macro_f1'])
        fastest_training = min(self.results.items(), key=lambda x: x[1]['training_time'])
        
        print(f"   Best Accuracy: {best_accuracy[1]['model_name']} ({best_accuracy[1]['accuracy']:.4f})")
        print(f"   Best Macro F1: {best_macro_f1[1]['model_name']} ({best_macro_f1[1]['macro_f1']:.4f})")
        print(f"   Fastest Training: {fastest_training[1]['model_name']} ({fastest_training[1]['training_time']:.2f}s)")
        
        # Performance gaps
        worst_model = min(self.results.items(), key=lambda x: x[1]['macro_f1'])
        performance_gap = best_macro_f1[1]['macro_f1'] - worst_model[1]['macro_f1']
        print(f"   Performance Gap: {performance_gap:.4f} (Best vs Worst Macro F1)")
    
    def plot_model_comparison(self, figsize=(15, 12)):
        """Create comprehensive visualization comparing models"""
        if not self.results:
            print("No results to plot!")
            return
        
        # Extract data for plotting (exclude ensemble for cleaner plots)
        models = []
        accuracies = []
        macro_f1s = []
        training_times = []
        
        for model_key, results in self.results.items():
            if model_key != 'ensemble':
                models.append(results['model_name'])
                accuracies.append(results['accuracy'])
                macro_f1s.append(results['macro_f1'])
                training_times.append(results['training_time'])
        
        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
        
        # Accuracy comparison
        bars1 = ax1.bar(models, accuracies, color='skyblue', alpha=0.7)
        ax1.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Accuracy')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, acc in zip(bars1, accuracies):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{acc:.3f}', ha='center', va='bottom', fontsize=8)
        
        # Macro F1 comparison
        bars2 = ax2.bar(models, macro_f1s, color='lightgreen', alpha=0.7)
        ax2.set_title('Model Macro F1 Comparison', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Macro F1 Score')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, f1 in zip(bars2, macro_f1s):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{f1:.3f}', ha='center', va='bottom', fontsize=8)
        
        # Training time comparison
        bars3 = ax3.bar(models, training_times, color='orange', alpha=0.7)
        ax3.set_title('Training Time Comparison', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Training Time (seconds)')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(axis='y', alpha=0.3)
        
        # Accuracy vs F1 scatter plot
        colors = plt.cm.Set3(np.linspace(0, 1, len(models)))
        scatter = ax4.scatter(accuracies, macro_f1s, s=100, alpha=0.7, c=colors)
        
        for i, model in enumerate(models):
            ax4.annotate(model, (accuracies[i], macro_f1s[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        ax4.set_xlabel('Accuracy')
        ax4.set_ylabel('Macro F1 Score')
        ax4.set_title('Accuracy vs Macro F1 Score', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_confusion_matrix(self, model_key=None):
        """Plot confusion matrix for specified model or best model"""
        if model_key is None:
            model_key, _ = self.get_best_model()
        
        if model_key not in self.results:
            print(f"Model {model_key} not found in results!")
            return
        
        results = self.results[model_key]
        cm = results['confusion_matrix']
        
        plt.figure(figsize=(10, 8))
        
        labels = self.label_encoder.classes_ if self.label_encoder else None
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=labels, yticklabels=labels)
        
        plt.title(f'Confusion Matrix - {results["model_name"]}', fontsize=14, fontweight='bold')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    # Example usage
    print("MODEL COMPARISON FRAMEWORK")
    print("This module provides comprehensive model comparison capabilities.")
    print("Import and use ModelComparator class in your main pipeline.")