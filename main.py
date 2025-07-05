#!/usr/bin/env python3
"""
Main Intent Detection Pipeline
Complete end-to-end pipeline for intent classification with model comparison
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import pickle
import warnings
import sys
import os

# Import our custom modules
from data_augumentation import DataBalancer, setup_nltk
from model_comparison import ModelComparator

warnings.filterwarnings('ignore')

class IntentDetectionPipeline:
    """Complete pipeline for intent detection with model comparison"""
    
    def __init__(self, use_augmentation=True, use_smote=True, target_samples=30):
        self.use_augmentation = use_augmentation
        self.use_smote = use_smote
        self.target_samples = target_samples
        
        # Initialize components
        self.data_balancer = DataBalancer()
        self.model_comparator = ModelComparator()
        self.vectorizer = None
        self.label_encoder = None
        
        # Data storage
        self.original_df = None
        self.balanced_df = None
        self.X_vectorized = None
        self.y_encoded = None
        
        print("Intent Detection Pipeline Initialized")
        print(f"Configuration:")
        print(f"  - Text Augmentation: {use_augmentation}")
        print(f"  - SMOTE Balancing: {use_smote}")
        print(f"  - Target Samples per Class: {target_samples}")
    
    def load_data(self, file_path, text_column='sentence', label_column='label'):
        """Load data from CSV file"""
        try:
            self.original_df = pd.read_csv(file_path)
            print(f"\n✅ Data loaded successfully from {file_path}")
            print(f"   Dataset shape: {self.original_df.shape}")
            print(f"   Columns: {list(self.original_df.columns)}")
            
            # Validate required columns
            if text_column not in self.original_df.columns:
                raise ValueError(f"Text column '{text_column}' not found in dataset")
            if label_column not in self.original_df.columns:
                raise ValueError(f"Label column '{label_column}' not found in dataset")
            
            # Show class distribution
            self.data_balancer.analyze_class_distribution(self.original_df, label_column)
            
            return True
            
        except FileNotFoundError:
            print(f"❌ Error: File {file_path} not found!")
            return False
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False
    
    def preprocess_and_balance_data(self, text_column='sentence', label_column='label'):
        """Preprocess text and balance classes"""
        print("\n" + "="*80)
        print("DATA PREPROCESSING AND BALANCING")
        print("="*80)
        
        if self.original_df is None:
            print("❌ No data loaded. Please load data first.")
            return False
        
        try:
            # Apply data balancing
            if self.use_augmentation:
                self.balanced_df = self.data_balancer.augment_minority_classes(
                    self.original_df, 
                    target_samples=self.target_samples,
                    label_column=label_column,
                    text_column=text_column
                )
            else:
                self.balanced_df = self.original_df.copy()
                print("Skipping text augmentation as requested.")
            
            # Preprocess text
            print("\nPreprocessing text...")
            self.balanced_df['processed_sentence'] = self.balanced_df[text_column].apply(
                self.data_balancer.preprocess_text
            )
            
            # Remove empty processed sentences
            original_len = len(self.balanced_df)
            self.balanced_df = self.balanced_df[self.balanced_df['processed_sentence'].str.len() > 0]
            final_len = len(self.balanced_df)
            
            if original_len != final_len:
                print(f"Removed {original_len - final_len} empty sentences after preprocessing")
            
            # Encode labels
            print("Encoding labels...")
            self.label_encoder = LabelEncoder()
            self.balanced_df['encoded_label'] = self.label_encoder.fit_transform(self.balanced_df[label_column])
            
            print(f"Label encoding mapping:")
            for i, label in enumerate(self.label_encoder.classes_):
                print(f"  {label} -> {i}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error in preprocessing: {e}")
            return False
    
    def vectorize_text(self, max_features=2000, ngram_range=(1, 3)):
        """Convert text to numerical features using TF-IDF"""
        print(f"\nVectorizing text with TF-IDF...")
        print(f"  Max features: {max_features}")
        print(f"  N-gram range: {ngram_range}")
        
        try:
            # Initialize TF-IDF vectorizer
            self.vectorizer = TfidfVectorizer(
                max_features=max_features,
                ngram_range=ngram_range,
                min_df=1,
                max_df=0.9,
                sublinear_tf=True
            )
            
            # Fit and transform
            self.X_vectorized = self.vectorizer.fit_transform(self.balanced_df['processed_sentence'])
            self.y_encoded = self.balanced_df['encoded_label'].values
            
            print(f"✅ Vectorization completed")
            print(f"   Feature matrix shape: {self.X_vectorized.shape}")
            print(f"   Vocabulary size: {len(self.vectorizer.vocabulary_)}")
            
            # Apply SMOTE if requested
            if self.use_smote:
                self.X_vectorized, self.y_encoded = self.data_balancer.apply_smote_balancing(
                    self.X_vectorized, self.y_encoded
                )
            
            return True
            
        except Exception as e:
            print(f"❌ Error in vectorization: {e}")
            return False
    
    def train_and_compare_models(self, use_grid_search=True, test_size=0.2):
        """Train and compare all models"""
        if self.X_vectorized is None or self.y_encoded is None:
            print("❌ Data not vectorized. Please run vectorization first.")
            return False
        
        try:
            # Run model comparison
            X_test, y_test = self.model_comparator.compare_all_models(
                self.X_vectorized, 
                self.y_encoded, 
                test_size=test_size,
                use_grid_search=use_grid_search,
                label_encoder=self.label_encoder
            )
            
            # Create ensemble model
            ensemble = self.model_comparator.create_ensemble_model()
            
            return True
            
        except Exception as e:
            print(f"❌ Error in model training: {e}")
            return False
    
    def display_results(self):
        """Display comprehensive results"""
        if not self.model_comparator.results:
            print("❌ No results to display. Please train models first.")
            return
        
        # Print detailed comparison
        self.model_comparator.print_detailed_comparison()
        
        # Create visualizations
        try:
            print("\n📊 Creating visualization plots...")
            self.model_comparator.plot_model_comparison()
        except Exception as e:
            print(f"Could not create plots: {e}")
    
    def test_predictions(self, test_queries=None):
        """Test predictions with the best model"""
        if test_queries is None:
            test_queries = [
                "I want EMI option please",
                "What is warranty period?", 
                "Cancel my order",
                "Check pincode delivery",
                "Ortho mattress features",
                "How much does it cost?",
                "Return policy details",
                "Available sizes",
                "Any current offers?",
                "Custom size mattress"
            ]
        
        best_model_key, best_results = self.model_comparator.get_best_model()
        
        if not best_results:
            print("❌ No trained models available for testing!")
            return
        
        print(f"\n🔮 TESTING PREDICTIONS WITH BEST MODEL: {best_results['model_name']}")
        print("="*80)
        
        try:
            best_model = best_results['best_model']
            
            # Preprocess queries
            processed_queries = [self.data_balancer.preprocess_text(query) for query in test_queries]
            
            # Vectorize
            X_queries = self.vectorizer.transform(processed_queries)
            
            # Predict
            predictions = best_model.predict(X_queries)
            predicted_labels = self.label_encoder.inverse_transform(predictions)
            
            # Get probabilities if available
            if hasattr(best_model, 'predict_proba'):
                probabilities = best_model.predict_proba(X_queries)
                
                print("\nPrediction Results:")
                print("-" * 70)
                for i, (query, prediction) in enumerate(zip(test_queries, predicted_labels)):
                    confidence = probabilities[i].max()
                    confidence_bar = "█" * int(confidence * 20)
                    print(f"'{query}'")
                    print(f"  → {prediction} (confidence: {confidence:.3f}) {confidence_bar}")
                    print()
            else:
                print("\nPrediction Results:")
                print("-" * 50)
                for query, prediction in zip(test_queries, predicted_labels):
                    print(f"'{query}' → {prediction}")
                    
        except Exception as e:
            print(f"❌ Error making predictions: {e}")
    
    def save_best_model(self, filename='best_intent_model.pkl'):
        """Save the best model and preprocessing components"""
        best_model_key, best_results = self.model_comparator.get_best_model()
        
        if not best_results:
            print("❌ No trained models to save!")
            return False
        
        try:
            model_data = {
                'model': best_results['best_model'],
                'vectorizer': self.vectorizer,
                'label_encoder': self.label_encoder,
                'model_name': best_results['model_name'],
                'performance_metrics': {
                    'accuracy': best_results['accuracy'],
                    'macro_f1': best_results['macro_f1'],
                    'weighted_f1': best_results['weighted_f1']
                },
                'preprocessing_config': {
                    'use_augmentation': self.use_augmentation,
                    'use_smote': self.use_smote,
                    'target_samples': self.target_samples
                }
            }
            
            with open(filename, 'wb') as f:
                pickle.dump(model_data, f)
            
            print(f"✅ Best model saved to {filename}")
            print(f"   Model: {best_results['model_name']}")
            print(f"   Macro F1: {best_results['macro_f1']:.4f}")
            print(f"   Accuracy: {best_results['accuracy']:.4f}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error saving model: {e}")
            return False
    
    def generate_report(self):
        """Generate a comprehensive report"""
        if not self.model_comparator.results:
            print("❌ No results available for report generation.")
            return
        
        print("\n" + "="*100)
        print("COMPREHENSIVE INTENT DETECTION PIPELINE REPORT")
        print("="*100)
        
        # Dataset information
        print(f"\n📊 DATASET INFORMATION:")
        if self.original_df is not None:
            print(f"   Original dataset size: {len(self.original_df)}")
            print(f"   Number of classes: {len(self.original_df['label'].unique())}")
            print(f"   Classes: {sorted(self.original_df['label'].unique())}")
        
        if self.balanced_df is not None:
            print(f"   Final dataset size: {len(self.balanced_df)}")
            if self.X_vectorized is not None:
                print(f"   Feature matrix shape: {self.X_vectorized.shape}")
        
        # Configuration
        print(f"\n⚙️  PIPELINE CONFIGURATION:")
        print(f"   Text Augmentation: {self.use_augmentation}")
        print(f"   SMOTE Balancing: {self.use_smote}")
        print(f"   Target Samples per Class: {self.target_samples}")
        
        # Best model information
        best_model_key, best_results = self.model_comparator.get_best_model()
        if best_results:
            print(f"\n🏆 BEST MODEL PERFORMANCE:")
            print(f"   Model: {best_results['model_name']}")
            print(f"   Accuracy: {best_results['accuracy']:.4f}")
            print(f"   Macro F1: {best_results['macro_f1']:.4f}")
            print(f"   Weighted F1: {best_results['weighted_f1']:.4f}")
            print(f"   Training Time: {best_results['training_time']:.2f}s")
        
        # Model ranking
        print(f"\n📈 MODEL RANKING (by Macro F1):")
        sorted_results = sorted(
            self.model_comparator.results.items(),
            key=lambda x: x[1]['macro_f1'],
            reverse=True
        )
        
        for i, (model_key, results) in enumerate(sorted_results, 1):
            print(f"   {i}. {results['model_name']}: {results['macro_f1']:.4f}")
        
        print(f"\n✅ Pipeline execution completed successfully!")

def load_model(filename='best_intent_model.pkl'):
    """Load a saved model"""
    try:
        with open(filename, 'rb') as f:
            model_data = pickle.load(f)
        
        print(f"✅ Model loaded from {filename}")
        print(f"   Model: {model_data['model_name']}")
        
        return model_data
        
    except FileNotFoundError:
        print(f"❌ Model file {filename} not found!")
        return None
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

def predict_intent(text, model_data):
    """Make prediction using loaded model"""
    try:
        # Initialize data balancer for preprocessing
        balancer = DataBalancer()
        
        # Preprocess text
        processed_text = balancer.preprocess_text(text)
        
        # Vectorize
        X_text = model_data['vectorizer'].transform([processed_text])
        
        # Predict
        prediction = model_data['model'].predict(X_text)[0]
        predicted_label = model_data['label_encoder'].inverse_transform([prediction])[0]
        
        # Get confidence if available
        if hasattr(model_data['model'], 'predict_proba'):
            confidence = model_data['model'].predict_proba(X_text)[0].max()
            return predicted_label, confidence
        else:
            return predicted_label, None
            
    except Exception as e:
        print(f"❌ Error making prediction: {e}")
        return None, None

def main():
    """Main function to run the complete pipeline"""
    
    # Setup NLTK
    setup_nltk()
    
    print("="*80)
    print("ENHANCED INTENT DETECTION WITH MODEL COMPARISON")
    print("="*80)
    
    # Initialize pipeline
    pipeline = IntentDetectionPipeline(
        use_augmentation=True,
        use_smote=True,
        target_samples=30
    )
    
    # Load data
    if not pipeline.load_data('sofmattress_train.csv'):
        print("Failed to load data. Please ensure 'sofmattress_train.csv' exists.")
        return None
    
    # Preprocess and balance data
    if not pipeline.preprocess_and_balance_data():
        print("Failed to preprocess data.")
        return None
    
    # Vectorize text
    if not pipeline.vectorize_text():
        print("Failed to vectorize text.")
        return None
    
    # Train and compare models
    if not pipeline.train_and_compare_models():
        print("Failed to train models.")
        return None
    
    # Display results
    pipeline.display_results()
    
    # Test predictions
    pipeline.test_predictions()
    
    # Save best model
    pipeline.save_best_model('best_intent_model_comparison.pkl')
    
    # Generate comprehensive report
    pipeline.generate_report()
    
    return pipeline

if __name__ == "__main__":
    print("Starting Enhanced Intent Detection Pipeline...")
    try:
        pipeline = main()
        if pipeline:
            print("\n🎉 Pipeline execution completed successfully!")
        else:
            print("\n❌ Pipeline execution failed!")
    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user.")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()